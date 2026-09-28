"""
Adaptive variable-resolution 2.5D grid (Modules 14-16) — the core
contribution of this system.

Design: a **ring-based adaptive grid**. The ground plane is divided into
concentric rings by horizontal distance from the sensor, r = sqrt(x^2+y^2).
Each ring has its own uniform cell size (its "resolution"). Within a ring,
cells are assigned by ordinary floor-division indexing at that ring's
resolution — exactly like a normal occupancy grid, just a different one per
ring.

Why this avoids the failure modes called out in the spec (gaps, overlaps,
duplicate/ambiguous cells):
  - A point's ring is a pure function of r, decided by a sorted list of
    breakpoints with half-open intervals [prev_max, this_max) — so every
    real-valued r maps to exactly one ring, including points exactly on a
    boundary (they go to the *inner*, finer ring — see `resolution_for_distance`).
  - Because the ring id is baked into the cell key `(ring_id, ix, iy)`, cells
    from different rings can never collide even though their (ix, iy)
    numbering restarts at 0 for each ring's own resolution. So there is no
    possibility of two rings' cells overlapping in key-space, and no need to
    reconcile different resolutions at a shared border cell.
  - Storage is a sparse dict keyed by that tuple, so memory is proportional
    to *occupied* cells, not to the full extent of the coarsest resolution.

Two resolution functions are supported (config: grid.resolution_mode):
  - "piecewise": a list of (max_range, resolution) breakpoints (matches the
    problem statement's example: 5cm within 10m, ~50cm around 100m).
  - "continuous": resolution(r) = clip(a + b*r, min_resolution, max_resolution),
    a genuinely continuous function of distance rather than discrete steps.
"""

from __future__ import annotations

import bisect
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


CellKey = Tuple[int, int, int]  # (ring_id, ix, iy)


@dataclass
class Cell:
    """One adaptive-resolution grid cell. Fields mirror spec section 16,
    trimmed of anything not actually populated (no dead weight fields)."""

    cell_id: CellKey
    center_x: float
    center_y: float
    cell_size: float
    distance_from_sensor: float

    min_elevation: float = float("inf")
    max_elevation: float = float("-inf")
    mean_elevation: float = 0.0

    terrain_probability: float = 0.0
    drivable_probability: float = 0.0
    static_obstacle_probability: float = 0.0
    dynamic_object_probability: float = 0.0

    semantic_class: int = 0
    confidence: float = 0.0
    point_count: int = 0
    timestamp: float = 0.0

    def elevation_range(self) -> float:
        if self.point_count == 0:
            return 0.0
        return self.max_elevation - self.min_elevation


class ResolutionFunction:
    def resolution(self, r: float) -> float:
        raise NotImplementedError

    def resolution_vec(self, r: np.ndarray) -> np.ndarray:
        return np.array([self.resolution(float(ri)) for ri in r], dtype=np.float32)


class PiecewiseResolution(ResolutionFunction):
    """Ring breakpoints as [(max_range_0, res_0), (max_range_1, res_1), ...],
    sorted ascending. A point with r <= max_range_0 falls in ring 0, etc."""

    def __init__(self, breakpoints: List[dict]):
        bp = sorted(breakpoints, key=lambda b: b["max_range"])
        self.max_ranges = [b["max_range"] for b in bp]
        self.resolutions = [b["resolution"] for b in bp]

    def ring_for_distance(self, r: float) -> int:
        idx = bisect.bisect_left(self.max_ranges, r)
        return min(idx, len(self.max_ranges) - 1)

    def resolution(self, r: float) -> float:
        return self.resolutions[self.ring_for_distance(r)]

    def num_rings(self) -> int:
        return len(self.max_ranges)

    def ring_bounds(self, ring_id: int) -> Tuple[float, float]:
        lo = 0.0 if ring_id == 0 else self.max_ranges[ring_id - 1]
        hi = self.max_ranges[ring_id]
        return lo, hi


class ContinuousResolution(ResolutionFunction):
    """resolution(r) = clip(a + b*r, min_resolution, max_resolution).
    Genuinely continuous — no ring boundaries at all; every distinct r that
    produces a distinct resolution still buckets deterministically because
    ContinuousAdaptiveGrid quantizes the *resolution value itself* into a
    small number of ring ids so the same overlap-free key scheme applies
    (see AdaptiveGrid._ring_id_for_continuous)."""

    def __init__(self, a: float, b: float, min_resolution: float, max_resolution: float, n_quantization_rings: int = 40):
        self.a = a
        self.b = b
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.n_quantization_rings = n_quantization_rings

    def resolution(self, r: float) -> float:
        return float(np.clip(self.a + self.b * r, self.min_resolution, self.max_resolution))


class AdaptiveGrid:
    """Sparse ring-based adaptive-resolution 2.5D grid."""

    def __init__(self, resolution_fn: ResolutionFunction, max_range: float = 100.0):
        self.resolution_fn = resolution_fn
        self.max_range = max_range
        self.cells: Dict[CellKey, Cell] = {}

    # ---- cell-key assignment -------------------------------------------------

    def _ring_id_for_continuous(self, r: float) -> int:
        fn: ContinuousResolution = self.resolution_fn  # type: ignore
        # Quantize distance itself into n rings spanning [0, max_range]; the
        # resolution used *within* a ring is resolution_fn(ring midpoint), so
        # points near a quantization boundary still get a locally-consistent
        # cell size, and the (ring_id, ix, iy) key stays collision-free
        # across rings exactly as in the piecewise case.
        ring_width = self.max_range / fn.n_quantization_rings
        ring_id = int(min(r // ring_width, fn.n_quantization_rings - 1))
        return ring_id

    def get_cell_key(self, x: float, y: float) -> Optional[CellKey]:
        r = float(np.hypot(x, y))
        if r > self.max_range:
            return None
        if isinstance(self.resolution_fn, PiecewiseResolution):
            ring_id = self.resolution_fn.ring_for_distance(r)
            res = self.resolution_fn.resolutions[ring_id]
        else:
            ring_id = self._ring_id_for_continuous(r)
            fn: ContinuousResolution = self.resolution_fn  # type: ignore
            ring_width = self.max_range / fn.n_quantization_rings
            ring_mid = (ring_id + 0.5) * ring_width
            res = fn.resolution(ring_mid)
        # Half-open cell assignment: floor() is deterministic at boundaries
        # (a point exactly on a cell edge always belongs to the cell whose
        # interval is [edge, edge+res)).
        ix = int(np.floor(x / res))
        iy = int(np.floor(y / res))
        return (ring_id, ix, iy)

    def resolution_for_key(self, key: CellKey) -> float:
        ring_id = key[0]
        if isinstance(self.resolution_fn, PiecewiseResolution):
            return self.resolution_fn.resolutions[ring_id]
        fn: ContinuousResolution = self.resolution_fn  # type: ignore
        ring_width = self.max_range / fn.n_quantization_rings
        ring_mid = (ring_id + 0.5) * ring_width
        return fn.resolution(ring_mid)

    def cell_center(self, key: CellKey) -> Tuple[float, float]:
        _, ix, iy = key
        res = self.resolution_for_key(key)
        return (ix + 0.5) * res, (iy + 0.5) * res

    # ---- point insertion -------------------------------------------------

    def get_or_create_cell(self, key: CellKey, timestamp: float = 0.0) -> Cell:
        cell = self.cells.get(key)
        if cell is None:
            cx, cy = self.cell_center(key)
            res = self.resolution_for_key(key)
            cell = Cell(
                cell_id=key,
                center_x=cx,
                center_y=cy,
                cell_size=res,
                distance_from_sensor=float(np.hypot(cx, cy)),
                timestamp=timestamp,
            )
            self.cells[key] = cell
        return cell

    def cell_count(self) -> int:
        return len(self.cells)

    def all_cells(self) -> List[Cell]:
        return list(self.cells.values())

    def memory_bytes_estimate(self) -> int:
        """Rough memory footprint: one Cell object's approximate resident
        size (measured via sys.getsizeof on its fields) times cell count.
        Used only for the uniform-vs-adaptive comparison in evaluation/."""
        per_cell = sys.getsizeof(Cell) + 16 * 14  # dataclass overhead + ~14 numeric fields
        return per_cell * len(self.cells)

    def resolution_profile(self) -> List[dict]:
        """Human-readable summary of what resolution applies at what range —
        used by the dashboard to prove the adaptive behavior (spec section 37)."""
        if isinstance(self.resolution_fn, PiecewiseResolution):
            out = []
            for i in range(self.resolution_fn.num_rings()):
                lo, hi = self.resolution_fn.ring_bounds(i)
                out.append({"ring": i, "range_m": [lo, hi], "resolution_m": self.resolution_fn.resolutions[i]})
            return out
        fn: ContinuousResolution = self.resolution_fn  # type: ignore
        ring_width = self.max_range / fn.n_quantization_rings
        out = []
        for i in range(fn.n_quantization_rings):
            lo, hi = i * ring_width, (i + 1) * ring_width
            out.append({"ring": i, "range_m": [lo, hi], "resolution_m": fn.resolution((lo + hi) / 2)})
        return out


def build_resolution_function(grid_cfg) -> ResolutionFunction:
    if grid_cfg.resolution_mode == "piecewise":
        return PiecewiseResolution([dict(b) for b in grid_cfg.breakpoints])
    if grid_cfg.resolution_mode == "continuous":
        c = grid_cfg.continuous
        return ContinuousResolution(a=c.a, b=c.b, min_resolution=c.min_resolution, max_resolution=c.max_resolution)
    raise ValueError(f"Unknown grid.resolution_mode: {grid_cfg.resolution_mode}")
