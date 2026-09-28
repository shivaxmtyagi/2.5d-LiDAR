"""Naive fixed-resolution grid — the baseline the adaptive grid is measured
against (Module 23 / spec section 36). Deliberately as simple as a real
"normal occupancy grid" would be, so the comparison is fair and not a straw
man: same sparse-dict storage strategy as AdaptiveGrid, just one resolution
everywhere instead of many."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

CellKey = Tuple[int, int]


@dataclass
class UniformCell:
    cell_id: CellKey
    center_x: float
    center_y: float
    min_elevation: float = float("inf")
    max_elevation: float = float("-inf")
    mean_elevation: float = 0.0
    point_count: int = 0


class UniformGrid:
    def __init__(self, resolution: float, max_range: float = 100.0):
        self.resolution = resolution
        self.max_range = max_range
        self.cells: Dict[CellKey, UniformCell] = {}

    def get_cell_key(self, x: float, y: float):
        r = float(np.hypot(x, y))
        if r > self.max_range:
            return None
        return (int(np.floor(x / self.resolution)), int(np.floor(y / self.resolution)))

    def get_or_create_cell(self, key: CellKey) -> UniformCell:
        cell = self.cells.get(key)
        if cell is None:
            cx = (key[0] + 0.5) * self.resolution
            cy = (key[1] + 0.5) * self.resolution
            cell = UniformCell(cell_id=key, center_x=cx, center_y=cy)
            self.cells[key] = cell
        return cell

    def cell_count(self) -> int:
        return len(self.cells)

    def memory_bytes_estimate(self) -> int:
        per_cell = sys.getsizeof(UniformCell) + 16 * 6
        return per_cell * len(self.cells)

    def theoretical_cell_count(self) -> int:
        """Cell count if the ENTIRE covered square were filled at this
        resolution — i.e. what a dense (non-sparse) uniform grid would need
        to allocate up front, which is the real-world memory cost a naive
        implementation pays regardless of occupancy."""
        side_cells = int(np.ceil((2 * self.max_range) / self.resolution))
        return side_cells * side_cells
