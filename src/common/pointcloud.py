"""
Common internal PointCloud representation.

Every loader (real .pcd/.ply/.bin, synthetic generator, future ROS bridge)
converts its input into this one structure so the rest of the pipeline
never has to know where the data came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PointCloud:
    """Unified point cloud representation used throughout the pipeline.

    points: (N, 3) float32 array of x, y, z in the LiDAR frame (meters).
    intensity: optional (N,) float32 array, normalized 0-1 where available.
    timestamp: optional (N,) float64 array, per-point capture time (seconds).
                If the sensor only provides one timestamp per frame, that
                scalar is broadcast to all points.
    ring: optional (N,) int32 array, laser ring/channel index.
    frame_id: coordinate frame this cloud is expressed in ("lidar",
              "vehicle", "world"), used by preprocessing.transforms.
    frame_timestamp: single float, capture time of the whole frame (seconds).
                      Used by tracking/temporal modules even if per-point
                      timestamps are missing.
    """

    points: np.ndarray
    intensity: Optional[np.ndarray] = None
    timestamp: Optional[np.ndarray] = None
    ring: Optional[np.ndarray] = None
    frame_id: str = "lidar"
    frame_timestamp: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"PointCloud.points must be (N, 3), got shape {self.points.shape}"
            )
        n = self.points.shape[0]
        if self.intensity is not None:
            self.intensity = np.asarray(self.intensity, dtype=np.float32).reshape(-1)
            if self.intensity.shape[0] != n:
                raise ValueError("intensity length must match points length")
        if self.timestamp is not None:
            self.timestamp = np.asarray(self.timestamp, dtype=np.float64).reshape(-1)
            if self.timestamp.shape[0] != n:
                raise ValueError("timestamp length must match points length")
        if self.ring is not None:
            self.ring = np.asarray(self.ring, dtype=np.int32).reshape(-1)
            if self.ring.shape[0] != n:
                raise ValueError("ring length must match points length")

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def subset(self, mask: np.ndarray) -> "PointCloud":
        """Return a new PointCloud containing only points where mask is True."""
        mask = np.asarray(mask, dtype=bool)
        return PointCloud(
            points=self.points[mask],
            intensity=None if self.intensity is None else self.intensity[mask],
            timestamp=None if self.timestamp is None else self.timestamp[mask],
            ring=None if self.ring is None else self.ring[mask],
            frame_id=self.frame_id,
            frame_timestamp=self.frame_timestamp,
            meta=dict(self.meta),
        )

    def has_intensity(self) -> bool:
        return self.intensity is not None

    def has_timestamp(self) -> bool:
        return self.timestamp is not None
