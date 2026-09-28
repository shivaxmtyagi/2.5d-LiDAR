"""
3D -> 2.5D projection (Module 17) and elevation/semantic fusion (Module 21).

For every point: compute its adaptive cell, then update that cell's running
elevation stats (min/max/mean — never collapsed to a single value, which is
the whole point of 2.5D vs. a binary occupancy grid, spec section 18) and
semantic vote. After all points are inserted, `finalize_cells` converts the
running per-cell accumulators into the final probabilities/semantic class
stored on each Cell.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from src.common.pointcloud import PointCloud
from src.mapping.adaptive_grid import AdaptiveGrid, Cell, CellKey
from src.objects.clustering import DetectedObject
from src.segmentation.base import SegmentationResult
from src.segmentation.heuristic import TERRAIN, STATIC_OBSTACLE, UNKNOWN
from src.terrain.terrain_analyzer import TerrainResult

DYNAMIC_OBJECT = 3


class _CellAccumulator:
    __slots__ = ("count", "sum_z", "min_z", "max_z", "class_votes", "sum_drivable",
                 "sum_confidence", "dynamic_votes", "static_votes")

    def __init__(self):
        self.count = 0
        self.sum_z = 0.0
        self.min_z = float("inf")
        self.max_z = float("-inf")
        self.class_votes: Dict[int, int] = defaultdict(int)
        self.sum_drivable = 0.0
        self.sum_confidence = 0.0
        self.dynamic_votes = 0
        self.static_votes = 0


def project_to_grid(
    pc: PointCloud,
    seg: SegmentationResult,
    terrain: TerrainResult,
    grid: AdaptiveGrid,
    timestamp: float,
    objects: Optional[List[DetectedObject]] = None,
) -> AdaptiveGrid:
    """Insert every point of `pc` into `grid`, fusing geometry + semantics +
    confidence. `objects` (post-tracking DetectedObjects, each with
    predicted_class "static_obstacle"/"dynamic_object") lets dynamic-object
    cells be flagged even though single-frame segmentation alone can't tell
    dynamic from static (see segmentation/heuristic.py docstring)."""
    n = len(pc)
    if n == 0:
        return grid

    # Build a per-point override for dynamic/static coming from the tracker.
    point_class_override = np.array(seg.semantic_class, copy=True)
    if objects:
        for obj in objects:
            if obj.predicted_class == "dynamic_object":
                point_class_override[obj.points_idx] = DYNAMIC_OBJECT
            elif obj.predicted_class in ("static_obstacle",):
                point_class_override[obj.points_idx] = STATIC_OBSTACLE

    accumulators: Dict[CellKey, _CellAccumulator] = {}
    xy = pc.points[:, :2]
    z = pc.points[:, 2]

    for i in range(n):
        key = grid.get_cell_key(float(xy[i, 0]), float(xy[i, 1]))
        if key is None:
            continue  # out of max_range
        acc = accumulators.get(key)
        if acc is None:
            acc = _CellAccumulator()
            accumulators[key] = acc
        zi = float(z[i])
        acc.count += 1
        acc.sum_z += zi
        acc.min_z = min(acc.min_z, zi)
        acc.max_z = max(acc.max_z, zi)
        cls = int(point_class_override[i])
        acc.class_votes[cls] += 1
        acc.sum_drivable += float(terrain.drivable_probability[i])
        acc.sum_confidence += float(seg.confidence[i])
        if cls == DYNAMIC_OBJECT:
            acc.dynamic_votes += 1
        elif cls == STATIC_OBSTACLE:
            acc.static_votes += 1

    for key, acc in accumulators.items():
        cell = grid.get_or_create_cell(key, timestamp=timestamp)
        cell.point_count = acc.count
        cell.min_elevation = acc.min_z
        cell.max_elevation = acc.max_z
        cell.mean_elevation = acc.sum_z / acc.count
        cell.semantic_class = max(acc.class_votes.items(), key=lambda kv: kv[1])[0]
        cell.confidence = acc.sum_confidence / acc.count
        cell.drivable_probability = acc.sum_drivable / acc.count
        cell.terrain_probability = acc.class_votes.get(TERRAIN, 0) / acc.count
        cell.static_obstacle_probability = acc.static_votes / acc.count
        cell.dynamic_object_probability = acc.dynamic_votes / acc.count
        cell.timestamp = timestamp

    return grid
