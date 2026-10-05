"""
Terrain analysis (Module 4).

Takes the point cloud + semantic predictions and produces drivable /
non-drivable classification. Deliberately does NOT rely on semantic class
alone (spec section 11): a point locally flat and low can be "terrain" by
semantics yet still sit on too steep a local slope, or have too much local
roughness, to be safely drivable — both are checked geometrically via a
local plane fit over each point's k nearest neighbors.

Implementation note: the local plane fit (normally one small least-squares
solve per point) is batched into a single vectorized computation using the
closed-form normal equations for all N points at once
(AtA x = Atz, solved with np.linalg.solve on a stacked (N,3,3) array) —
looping this in Python was the dominant latency cost measured in this
build. See src/segmentation/heuristic.py for the same rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from src.common.pointcloud import PointCloud
from src.segmentation.base import SegmentationResult
from src.segmentation.heuristic import TERRAIN


@dataclass
class TerrainResult:
    slope_deg: np.ndarray             # (N,) local slope estimate, degrees
    roughness: np.ndarray             # (N,) stddev of local height residuals, meters
    drivable: np.ndarray              # (N,) bool
    drivable_probability: np.ndarray  # (N,) float32 in [0,1]


def analyze_terrain(
    pc: PointCloud,
    seg: SegmentationResult,
    neighbor_radius: float = 1.0,
    max_slope_deg: float = 15.0,
    max_height_diff: float = 0.15,
    roughness_threshold: float = 0.06,
    min_neighbors: int = 4,
    k_neighbors: int = 10,
) -> TerrainResult:
    n = len(pc)
    if n == 0:
        empty = np.zeros(0, dtype=np.float32)
        return TerrainResult(empty, empty, np.zeros(0, dtype=bool), empty)

    xy = pc.points[:, :2]
    z = pc.points[:, 2]
    k = max(min(k_neighbors, n), 3)  # need >=3 points to fit a plane
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k)  # (N, k)
    if k == 1:
        idx = idx.reshape(-1, 1)

    local_xy = xy[idx]                    # (N, k, 2)
    local_z = z[idx]                      # (N, k)
    ones = np.ones((n, k, 1), dtype=np.float32)
    A = np.concatenate([local_xy, ones], axis=2)          # (N, k, 3): [x, y, 1]

    AtA = np.einsum("nki,nkj->nij", A, A)                  # (N, 3, 3)
    AtA += np.eye(3, dtype=np.float32) * 1e-6               # numerical stability
    Atz = np.einsum("nki,nk->ni", A, local_z)[:, :, None]    # (N, 3, 1) — batched solve needs a trailing dim

    coeffs = np.linalg.solve(AtA, Atz)[:, :, 0]              # (N, 3): a, b, c for z = a*x + b*y + c
    a, b = coeffs[:, 0], coeffs[:, 1]
    slope_deg = np.degrees(np.arctan(np.hypot(a, b))).astype(np.float32)

    predicted_z = np.einsum("nki,ni->nk", A, coeffs)        # (N, k)
    residual = local_z - predicted_z
    roughness = residual.std(axis=1).astype(np.float32)

    height_diff = local_z.max(axis=1) - local_z.min(axis=1)

    is_terrain_semantic = seg.semantic_class == TERRAIN
    slope_ok = slope_deg <= max_slope_deg
    height_ok = height_diff <= max_height_diff * 4  # local patch can legitimately span a curb
    rough_ok = roughness <= roughness_threshold

    drivable_probability = (
        0.4 * is_terrain_semantic.astype(np.float32)
        + 0.25 * slope_ok.astype(np.float32)
        + 0.2 * height_ok.astype(np.float32)
        + 0.15 * rough_ok.astype(np.float32)
    ).astype(np.float32)

    # Too-sparse neighborhoods (fewer real neighbors than min_neighbors within
    # a reasonable radius) get down-weighted rather than trusted outright.
    if n > min_neighbors:
        radius_counts = np.array(
            tree.query_ball_point(xy, r=neighbor_radius, return_length=True)
        )
        sparse = radius_counts < min_neighbors
        drivable_probability[sparse] = np.minimum(drivable_probability[sparse], 0.1)

    drivable = drivable_probability >= 0.6
    return TerrainResult(
        slope_deg=slope_deg,
        roughness=roughness,
        drivable=drivable,
        drivable_probability=drivable_probability,
    )
