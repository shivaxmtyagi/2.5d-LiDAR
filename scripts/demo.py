#!/usr/bin/env python3
"""
One-click demonstration (spec section 27).

    python scripts/demo.py

Generates a synthetic scene, runs the full pipeline once, saves all
visualization views + a JSON metrics report to data/predictions/, and
prints a summary to the console.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import Config
from src.input.synthetic import SyntheticLidarGenerator, SyntheticSceneConfig
from src.pipeline.pipeline import AdaptiveLidarPipeline
from src.visualization.plots import render_all_views


def main():
    print("=" * 70)
    print("Adaptive Variable-Resolution 2.5D LiDAR Mapping — DEMO")
    print("MODE: BASELINE / DEMO  (heuristic geometric segmentation — no")
    print("       trained deep-learning checkpoint is loaded in this build)")
    print("=" * 70)

    cfg = Config.load(ROOT / "configs" / "default.yaml")
    generator = SyntheticLidarGenerator(SyntheticSceneConfig())
    pc = generator.generate_frame(t=0.0)
    print(f"\nGenerated synthetic frame: {len(pc)} points "
          f"(scene: road, curb, pothole, wall, pole, gantry, parked vehicle, "
          f"moving pedestrian, moving vehicle)")

    pipeline = AdaptiveLidarPipeline(cfg)
    result = pipeline.process(pc)

    print("\n--- Pipeline stage latency (ms) ---")
    for k in ["preprocessing_ms", "segmentation_ms", "terrain_ms", "objects_ms", "projection_ms", "total_ms"]:
        print(f"  {k:20s} {result.metrics[k]:8.2f}")
    print(f"  {'fps':20s} {result.metrics['fps']:8.2f}")

    print(f"\nPoints: raw={result.n_points_raw}  after preprocessing={result.n_points_processed}")
    print(f"Adaptive grid cells occupied: {result.adaptive_grid.cell_count()}")
    print(f"Semantic segmentation mode: {result.model_mode.upper()}")

    print(f"\nObjects detected: {len(result.objects)}")
    for obj in result.objects:
        speed = None
        if obj.velocity is not None:
            import numpy as np
            speed = float(np.linalg.norm(obj.velocity))
        print(f"  track#{obj.track_id:<3} class={obj.predicted_class:<16} "
              f"dist={obj.distance:5.1f}m  pts={obj.point_count:<5} "
              f"size(w x l x h)=({obj.width:.1f} x {obj.length:.1f} x {obj.height:.1f})m"
              + (f"  speed={speed:.2f} m/s" if speed is not None else ""))

    print("\nAdaptive resolution profile (near -> far):")
    for ring in result.adaptive_grid.resolution_profile():
        lo, hi = ring["range_m"]
        print(f"  {lo:6.1f} - {hi:6.1f} m  ->  {ring['resolution_m']:.2f} m cells")

    out_dir = ROOT / "data" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRendering visualization views to {out_dir} ...")
    paths = render_all_views(result.processed_pointcloud, result.semantic, result.terrain, result.adaptive_grid,
                              result.objects, str(out_dir), prefix="demo")
    for p in paths:
        print(f"  wrote {p}")

    report = {
        "n_points_raw": result.n_points_raw,
        "n_points_processed": result.n_points_processed,
        "metrics": result.metrics,
        "model_mode": result.model_mode,
        "n_objects": len(result.objects),
        "grid_cell_count": result.adaptive_grid.cell_count(),
        "resolution_profile": result.adaptive_grid.resolution_profile(),
    }
    report_path = out_dir / "demo_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  wrote {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
