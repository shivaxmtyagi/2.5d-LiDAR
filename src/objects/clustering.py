"""Object detection via clustering (Module 5). Operates on points already
labeled non-terrain by the segmenter — obstacle candidates only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.cluster import DBSCAN

from src.common.pointcloud import PointCloud


@dataclass
class DetectedObject:
    object_id: int
    points_idx: np.ndarray      # indices into the source (obstacle-subset) point cloud
    centroid: np.ndarray        # (3,)
    bbox_min: np.ndarray        # (3,)
    bbox_max: np.ndarray        # (3,)
    point_count: int
    predicted_class: str = "unknown"   # "static_obstacle" | "dynamic_object" (set by tracker)
    confidence: float = 0.5
    velocity: Optional[np.ndarray] = None  # (2,) m/s in xy, set by tracker once known
    track_id: Optional[int] = None

    @property
    def width(self) -> float:
        return float(self.bbox_max[0] - self.bbox_min[0])

    @property
    def length(self) -> float:
        return float(self.bbox_max[1] - self.bbox_min[1])

    @property
    def height(self) -> float:
        return float(self.bbox_max[2] - self.bbox_min[2])

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.centroid[:2]))


def cluster_obstacles(pc: PointCloud, obstacle_mask: np.ndarray, eps: float = 0.6,
                       min_samples: int = 6) -> List[DetectedObject]:
    """Run DBSCAN on obstacle-candidate points and build one DetectedObject
    per cluster (label -1 = noise, discarded)."""
    idx = np.nonzero(obstacle_mask)[0]
    if len(idx) == 0:
        return []
    pts = pc.points[idx]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)

    objects: List[DetectedObject] = []
    for cluster_label in sorted(set(labels)):
        if cluster_label == -1:
            continue
        member_mask = labels == cluster_label
        member_global_idx = idx[member_mask]
        member_pts = pc.points[member_global_idx]
        objects.append(DetectedObject(
            object_id=int(cluster_label),
            points_idx=member_global_idx,
            centroid=member_pts.mean(axis=0),
            bbox_min=member_pts.min(axis=0),
            bbox_max=member_pts.max(axis=0),
            point_count=int(member_mask.sum()),
        ))
    return objects


def classify_by_size(obj: DetectedObject, pedestrian_max_footprint: float = 1.0,
                      pedestrian_max_height: float = 2.2, vehicle_min_footprint: float = 1.6) -> str:
    """Heuristic (BASELINE/DEMO mode) size-based class hint — not a trained
    classifier. Only used to label objects for the dashboard; static/dynamic
    is still decided by the tracker, not by this function."""
    footprint = max(obj.width, obj.length)
    if footprint <= pedestrian_max_footprint and obj.height <= pedestrian_max_height:
        return "pedestrian_like"
    if footprint >= vehicle_min_footprint:
        return "vehicle_like"
    return "other_obstacle"
