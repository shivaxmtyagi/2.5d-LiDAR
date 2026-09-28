#!/usr/bin/env python3
"""
Uniform-grid vs. adaptive-grid benchmark (Module 23/36):

    python scripts/benchmark.py

Runs src.evaluation.benchmark.run_benchmark on a synthetic frame and prints
the comparison table + writes it to data/predictions/benchmark_report.json.
All numbers here are measured from an actual run, not asserted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import Config
from src.input.synthetic import SyntheticLidarGenerator, SyntheticSceneConfig
from src.evaluation.benchmark import run_benchmark


def main():
    cfg = Config.load(ROOT / "configs" / "default.yaml")
    generator = SyntheticLidarGenerator(SyntheticSceneConfig())
    pc = generator.generate_frame(t=0.0)

    result = run_benchmark(
        pc, cfg.grid,
        uniform_resolution=cfg.uniform_grid_baseline.resolution,
        max_range=cfg.sensor.max_range,
    )

    print("=" * 70)
    print("UNIFORM vs ADAPTIVE GRID BENCHMARK (measured, same point cloud)")
    print("=" * 70)
    print(f"Input points: {result.n_points}")
    print()
    print(f"{'metric':38s} {'uniform':>15s} {'adaptive':>15s}")
    print(f"{'-'*38} {'-'*15} {'-'*15}")
    print(f"{'occupied cell count':38s} {result.uniform_cell_count_sparse:>15d} {result.adaptive_cell_count:>15d}")
    print(f"{'theoretical dense cell count':38s} {result.uniform_cell_count_dense_theoretical:>15d} {'n/a':>15s}")
    print(f"{'memory estimate (KB, sparse)':38s} {result.uniform_memory_bytes_sparse/1024:>15.1f} {result.adaptive_memory_bytes/1024:>15.1f}")
    print(f"{'projection latency (ms)':38s} {result.uniform_projection_ms:>15.2f} {result.adaptive_projection_ms:>15.2f}")
    print()
    print(f"Cell-count reduction vs. dense uniform grid: {result.cell_count_reduction_pct:.2f}%")
    print(f"Memory reduction vs. dense uniform grid:      {result.memory_reduction_pct:.2f}%")
    print(f"Near-field resolution: {result.near_resolution_m} m   Far-field resolution: {result.far_resolution_m} m")

    out_path = ROOT / "data" / "predictions" / "benchmark_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
