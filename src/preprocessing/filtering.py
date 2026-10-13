"""Point cloud filtering (Module 2A/2B). Ground points are never dropped
here — terrain/elevation is a required output, so only range and noise
filtering happen at this stage (see terrain/ for drivability analysis)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.common.pointcloud import PointCloud


def range_filter(pc: PointCloud, min_range: float, max_range: float) -> PointCloud:
    """Drop points outside [min_range, max_range] horizontal distance from sensor."""
    if pc.is_empty:
        return pc
    r = np.linalg.norm(pc.points[:, :2], axis=1)
    mask = (r >= min_range) & (r <= max_range)
    return pc.subset(mask)


def statistical_outlier_removal(pc: PointCloud, k_neighbors: int = 12, std_ratio: float = 2.0) -> PointCloud:
    """Remove points whose mean distance to their k nearest neighbors is an
    outlier relative to the cloud's global mean/std of that statistic."""
    n = len(pc)
    if n <= k_neighbors + 1:
        return pc  # too few points for a meaningful neighborhood statistic
    tree = cKDTree(pc.points)
    dists, _ = tree.query(pc.points, k=k_neighbors + 1)  # includes self at k=0
    mean_dist = dists[:, 1:].mean(axis=1)
    global_mean = mean_dist.mean()
    global_std = mean_dist.std()
    threshold = global_mean + std_ratio * global_std
    mask = mean_dist <= threshold
    return pc.subset(mask)


def radius_outlier_removal(pc: PointCloud, radius: float = 0.3, min_neighbors: int = 3) -> PointCloud:
    """Remove points with fewer than min_neighbors within `radius`."""
    n = len(pc)
    if n == 0:
        return pc
    tree = cKDTree(pc.points)
    counts = tree.query_ball_point(pc.points, r=radius, return_length=True)
    mask = counts >= (min_neighbors + 1)  # +1 because the point itself counts
    return pc.subset(mask)


def voxel_downsample(pc: PointCloud, voxel_size: float) -> PointCloud:
    """Downsample by keeping the point nearest to each voxel's centroid.
    Deterministic (order-independent) and preserves per-point attributes."""
    if pc.is_empty or voxel_size <= 0:
        return pc
    voxel_idx = np.floor(pc.points / voxel_size).astype(np.int64)
    # Hash voxel coords -> group via a structured view for a fast unique().
    keys = voxel_idx[:, 0] * 73856093 ^ voxel_idx[:, 1] * 19349663 ^ voxel_idx[:, 2] * 83492791
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    # First occurrence of each key after sorting = one representative per voxel.
    boundaries = np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]])
    keep_idx = order[boundaries]
    mask = np.zeros(len(pc), dtype=bool)
    mask[keep_idx] = True
    return pc.subset(mask)
