#!/usr/bin/env python3
"""
Sequential / multi-frame demo (spec section 27):

    python scripts/demo_realtime.py --frames 8

Streams a sequence of synthetic frames through the same pipeline instance so
object tracking and temporal map decay run across frames (a pedestrian and
vehicle move; their dynamic-object grid cells should appear, follow them,
and fade once they've moved on) — and renders a view of the final frame.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import Config
from src.input.synthetic import SyntheticLidarGenerator, SyntheticSceneConfig
from src.pipeline.pipeline import AdaptiveLidarPipeline
from src.visualization.plots import render_all_views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.3, help="seconds between frames")
    args = parser.parse_args()

    print("=" * 70)
    print("Adaptive Variable-Resolution 2.5D LiDAR Mapping — SEQUENTIAL DEMO")
    print("MODE: BASELINE / DEMO")
    print("=" * 70)

    cfg = Config.load(ROOT / "configs" / "default.yaml")
    generator = SyntheticLidarGenerator(SyntheticSceneConfig())
    pipeline = AdaptiveLidarPipeline(cfg)

    last_result = None
    last_pc = None
    fps_values = []
    for i in range(args.frames):
        t = i * args.dt
        pc = generator.generate_frame(t=t)
        wall_t0 = time.perf_counter()
        result = pipeline.process(pc)
        wall_elapsed = time.perf_counter() - wall_t0
        fps_values.append(result.metrics["fps"])

        dynamic_ids = [o.track_id for o in result.objects if o.predicted_class == "dynamic_object"]
        static_ids = [o.track_id for o in result.objects if o.predicted_class == "static_obstacle"]
        print(f"frame {i:2d}  t={t:5.2f}s  wall={wall_elapsed:5.2f}s  "
              f"pts={result.n_points_processed:6d}  cells={result.adaptive_grid.cell_count():6d}  "
              f"dynamic={dynamic_ids}  static={static_ids}")

        last_result, last_pc = result, result.processed_pointcloud

    print(f"\nMean processing FPS across sequence: {sum(fps_values)/len(fps_values):.2f}")
    print("(Heuristic BASELINE mode is not optimized for real-time; see README > Limitations.)")

    out_dir = ROOT / "data" / "predictions"
    print(f"\nRendering final-frame views to {out_dir} ...")
    render_all_views(last_pc, last_result.semantic, last_result.terrain, last_result.adaptive_grid,
                      last_result.objects, str(out_dir), prefix="realtime_final")
    print("Done.")


if __name__ == "__main__":
    main()
