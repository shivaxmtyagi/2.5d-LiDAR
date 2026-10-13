"""
Point-cloud augmentation for training (Module 9).

Deliberately restricted to augmentations that preserve the physical meaning
of a ground-vehicle LiDAR sweep (spec section 9: "avoid augmentations that
destroy the physical meaning of the LiDAR coordinate system"):
  - rotation: ONLY about the vertical (z) axis (yaw). Rolling or pitching
    the whole point cloud would put the "ground" out of the horizontal
    plane, which no real sensor mounted on a ground vehicle would ever see.
  - translation: small XY jitter (simulates pose/localization noise), no Z
    translation (would fake the sensor floating above/below the ground).
  - jitter: small per-point Gaussian noise, matching real range-noise
    characteristics rather than an arbitrary distribution.
  - scaling: small uniform scale (simulates calibration scale error) —
    kept close to 1.0, since a real scene's actual physical dimensions
    shouldn't change much.
  - random point dropout: simulates sparser returns (rain, dust, range).
  - noise points: a handful of random spurious returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentationConfig:
    yaw_rotation: bool = True
    max_yaw_deg: float = 180.0
    xy_translation_std: float = 0.1
    jitter_std: float = 0.01
    jitter_clip: float = 0.05
    scale_range: tuple = (0.95, 1.05)
    dropout_max_ratio: float = 0.1
    noise_points_max: int = 20
    noise_extent: float = 50.0


def augment_point_cloud(points: np.ndarray, labels: np.ndarray, cfg: AugmentationConfig,
                         rng: np.random.Generator) -> tuple:
    """points: (N,3), labels: (N,). Returns (augmented_points, augmented_labels).
    Order: rotate -> scale -> translate -> jitter -> dropout -> add noise,
    so each transform composes physically (e.g. jitter is applied in the
    already-rotated/scaled frame, matching real sensor noise being added
    after the scene geometry, not before)."""
    pts = points.copy()

    if cfg.yaw_rotation:
        yaw = np.deg2rad(rng.uniform(-cfg.max_yaw_deg, cfg.max_yaw_deg))
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts[:, :2] = pts[:, :2] @ rot.T

    scale = rng.uniform(*cfg.scale_range)
    pts *= scale

    tx, ty = rng.normal(0, cfg.xy_translation_std, size=2)
    pts[:, 0] += tx
    pts[:, 1] += ty

    jitter = np.clip(rng.normal(0, cfg.jitter_std, size=pts.shape), -cfg.jitter_clip, cfg.jitter_clip)
    pts += jitter.astype(np.float32)

    if cfg.dropout_max_ratio > 0 and len(pts) > 10:
        ratio = rng.uniform(0, cfg.dropout_max_ratio)
        keep_mask = rng.uniform(size=len(pts)) >= ratio
        pts, labels = pts[keep_mask], labels[keep_mask]

    if cfg.noise_points_max > 0:
        n_noise = rng.integers(0, cfg.noise_points_max + 1)
        if n_noise > 0:
            noise_pts = rng.uniform(-cfg.noise_extent, cfg.noise_extent, size=(n_noise, 3)).astype(np.float32)
            noise_pts[:, 2] = rng.uniform(-1, 3, size=n_noise)  # plausible height range
            pts = np.concatenate([pts, noise_pts], axis=0)
            labels = np.concatenate([labels, np.zeros(n_noise, dtype=labels.dtype)])  # "unknown"

    return pts.astype(np.float32), labels
