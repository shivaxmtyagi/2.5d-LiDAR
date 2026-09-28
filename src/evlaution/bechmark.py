"""
Uniform-grid vs. adaptive-grid benchmark (Module 23 / spec section 36).

Runs both grids over the SAME point cloud and SAME spatial coverage and
reports measured (not fabricated) cell counts, memory estimates, and
projection latency. This is the evidence for the project's core memory-
reduction claim — if these numbers don't show a reduction, the README must
say so rather than paper over it (spec section 51).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

from src.common.pointcloud import PointCloud
from src.mapping.adaptive_grid import AdaptiveGrid, build_resolution_function
from src.mapping.uniform_grid import UniformGrid
from src.mapping.projection import project_to_grid
from src.segmentation.base import SegmentationResult
from src.terrain.terrain_analyzer import TerrainResult
import numpy as np


@dataclass
class BenchmarkResult:
    n_points: int
    uniform_resolution_m: float
    uniform_cell_count_sparse: int
    uniform_cell_count_dense_theoretical: int
    uniform_memory_bytes_sparse: int
    uniform_projection_ms: float
    adaptive_cell_count: int
    adaptive_memory_bytes: int
    adaptive_projection_ms: float
    cell_count_reduction_pct: float
    memory_reduction_pct: float
    near_resolution_m: float
    far_resolution_m: float

    def to_dict(self):
        return asdict(self)


def run_benchmark(pc: PointCloud, grid_cfg, uniform_resolution: float, max_range: float) -> BenchmarkResult:
    n = len(pc)
    # Neutral inputs so the benchmark measures grid mechanics only, not
    # segmentation/terrain quality (those are measured separately in
    # evaluation/accuracy.py).
    seg = SegmentationResult(
        semantic_class=np.ones(n, dtype=np.int32),
        confidence=np.full(n, 0.8, dtype=np.float32),
        mode="baseline",
    )
    terrain = TerrainResult(
        slope_deg=np.zeros(n, dtype=np.float32),
        roughness=np.zeros(n, dtype=np.float32),
        drivable=np.ones(n, dtype=bool),
        drivable_probability=np.full(n, 0.9, dtype=np.float32),
    )

    # --- Adaptive grid ---
    adaptive = AdaptiveGrid(build_resolution_function(grid_cfg), max_range=max_range)
    t0 = time.perf_counter()
    project_to_grid(pc, seg, terrain, adaptive, timestamp=0.0)
    t_adaptive = (time.perf_counter() - t0) * 1000.0

    # --- Uniform baseline grid (same fine resolution as the adaptive grid's
    #     *nearest* ring, applied everywhere — the naive "just use the best
    #     resolution for the whole map" approach) ---
    uniform = UniformGrid(resolution=uniform_resolution, max_range=max_range)
    t0 = time.perf_counter()
    xy = pc.points[:, :2]
    z = pc.points[:, 2]
    for i in range(n):
        key = uniform.get_cell_key(float(xy[i, 0]), float(xy[i, 1]))
        if key is None:
            continue
        cell = uniform.get_or_create_cell(key)
        zi = float(z[i])
        cell.point_count += 1
        cell.min_elevation = min(cell.min_elevation, zi)
        cell.max_elevation = max(cell.max_elevation, zi)
        cell.mean_elevation += (zi - cell.mean_elevation) / cell.point_count
    t_uniform = (time.perf_counter() - t0) * 1000.0

    adaptive_mem = adaptive.memory_bytes_estimate()
    uniform_mem_sparse = uniform.memory_bytes_estimate()
    uniform_dense_theoretical = uniform.theoretical_cell_count()

    # Fair "what would a naive implementation actually allocate" comparison:
    # a dense uniform grid at the fine resolution over the full covered
    # square, vs. the adaptive grid's sparse occupied-cell footprint.
    bytes_per_uniform_cell = uniform_mem_sparse / max(uniform.cell_count(), 1)
    uniform_dense_memory_bytes = uniform_dense_theoretical * bytes_per_uniform_cell

    cell_reduction = (
        100.0 * (1 - adaptive.cell_count() / uniform_dense_theoretical)
        if uniform_dense_theoretical > 0 else float("nan")
    )
    mem_reduction = (
        100.0 * (1 - adaptive_mem / uniform_dense_memory_bytes)
        if uniform_dense_memory_bytes > 0 else float("nan")
    )

    profile = adaptive.resolution_profile()
    near_res = profile[0]["resolution_m"] if profile else float("nan")
    far_res = profile[-1]["resolution_m"] if profile else float("nan")

    return BenchmarkResult(
        n_points=n,
        uniform_resolution_m=uniform_resolution,
        uniform_cell_count_sparse=uniform.cell_count(),
        uniform_cell_count_dense_theoretical=uniform_dense_theoretical,
        uniform_memory_bytes_sparse=uniform_mem_sparse,
        uniform_projection_ms=t_uniform,
        adaptive_cell_count=adaptive.cell_count(),
        adaptive_memory_bytes=adaptive_mem,
        adaptive_projection_ms=t_adaptive,
        cell_count_reduction_pct=cell_reduction,
        memory_reduction_pct=mem_reduction,
        near_resolution_m=near_res,
        far_resolution_m=far_res,
    )
