"""Coordinate frame transforms (Module 2C): lidar -> vehicle -> world.

Transforms are stored as 4x4 homogeneous matrices, the standard robotics
convention, so they compose via simple matrix multiplication and this same
code path works whether the pose comes from a static extrinsic calibration
or a live localization/SLAM estimate.
"""

from __future__ import annotations

import numpy as np

from src.common.pointcloud import PointCloud


def make_transform(translation=(0.0, 0.0, 0.0), rotation_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Build a 4x4 transform from an XYZ translation and roll/pitch/yaw (degrees)."""
    roll, pitch, yaw = np.deg2rad(rotation_deg)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    r = rz @ ry @ rx

    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = translation
    return t


def apply_transform(pc: PointCloud, transform: np.ndarray, new_frame_id: str) -> PointCloud:
    """Apply a 4x4 homogeneous transform to every point in the cloud."""
    if pc.is_empty:
        result = pc.subset(np.zeros(0, dtype=bool))
        result.frame_id = new_frame_id
        return result
    homo = np.concatenate([pc.points, np.ones((len(pc), 1), dtype=np.float32)], axis=1)
    transformed = (transform @ homo.T).T[:, :3].astype(np.float32)
    return PointCloud(
        points=transformed,
        intensity=pc.intensity,
        timestamp=pc.timestamp,
        ring=pc.ring,
        frame_id=new_frame_id,
        frame_timestamp=pc.frame_timestamp,
        meta=dict(pc.meta),
    )


def lidar_to_vehicle(pc: PointCloud, extrinsic: np.ndarray) -> PointCloud:
    return apply_transform(pc, extrinsic, new_frame_id="vehicle")


def vehicle_to_world(pc: PointCloud, pose: np.ndarray) -> PointCloud:
    return apply_transform(pc, pose, new_frame_id="world")
