"""
Temporal map management (Module 22).

Static geometry (terrain, walls, poles) should persist across frames even
if a given frame doesn't re-observe every cell (LiDAR sparsity, occlusion).
Dynamic-object evidence should NOT persist — a cell that says "car here"
must stop saying that once the car has moved on, or the map lies. Each
frame:
  1. Every cell not re-observed this frame has its dynamic_object_probability
     exponentially decayed by `temporal_decay`.
  2. A cell whose dynamic_object_probability has decayed below a small
     threshold, and that has gone unobserved for
     `dynamic_object_timeout_frames`, has its dynamic flag cleared entirely
     (spec section 22: "do not allow stale moving objects to permanently
     remain in the map").
Static terrain/obstacle fields are left untouched by decay — they're
expected to persist.
"""

from __future__ import annotations

from typing import Set

from src.mapping.adaptive_grid import AdaptiveGrid, CellKey


class TemporalMapManager:
    def __init__(self, decay: float = 0.9, dynamic_timeout_frames: int = 5,
                 clear_threshold: float = 0.05):
        self.decay = decay
        self.dynamic_timeout_frames = dynamic_timeout_frames
        self.clear_threshold = clear_threshold
        self._missed_frames: dict[CellKey, int] = {}

    def update(self, grid: AdaptiveGrid, observed_keys: Set[CellKey]) -> None:
        for key, cell in list(grid.cells.items()):
            if key in observed_keys:
                self._missed_frames[key] = 0
                continue
            # Not observed this frame: decay dynamic evidence, age the cell.
            missed = self._missed_frames.get(key, 0) + 1
            self._missed_frames[key] = missed
            cell.dynamic_object_probability *= self.decay
            if (cell.dynamic_object_probability < self.clear_threshold
                    and missed >= self.dynamic_timeout_frames):
                cell.dynamic_object_probability = 0.0
                # If the cell had no terrain/static evidence either, it's now
                # stale — drop it so memory doesn't grow unboundedly for a
                # sensor that's been running a long time.
                if cell.terrain_probability < 0.05 and cell.static_obstacle_probability < 0.05:
                    del grid.cells[key]
                    del self._missed_frames[key]
