"""
Synthetic LiDAR scene generator (Module 30).

Builds a plausible driving scene — flat ground, road, curb, pothole, wall,
pole, a vehicle-like box, a pedestrian-like column, a slope, and an
overhanging structure — so the whole pipeline can be developed and tested
without a physical sensor or a downloaded dataset.

This is explicitly a geometric simulator, not a physically accurate LiDAR
raytracer: points are sampled directly on object surfaces plus range noise,
not simulated via ray casting against occlusion. That's a documented
limitation (see README, "Limitations").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.common.pointcloud import PointCloud

# Ground-truth semantic ids for evaluation (independent of the model's output
# class table so we can measure accuracy against something).
GT_UNKNOWN = 0
GT_TERRAIN = 1
GT_STATIC_OBSTACLE = 2
GT_DYNAMIC_OBJECT = 3


@dataclass
class MovingObjectSpec:
    name: str
    start_xy: tuple
    velocity_xy: tuple  # meters/sec
    height: float
    footprint: float  # square footprint side length, meters


@dataclass
class SyntheticSceneConfig:
    ground_extent: float = 50.0       # scene spans [-extent, extent] in x and y
    point_density: float = 3.0        # points per square meter for ground (kept modest so
                                       # the per-point neighborhood queries in the heuristic
                                       # segmenter/terrain analyzer run in seconds, not minutes —
                                       # a real sensor driver + vectorized/trained model would not
                                       # have this constraint)
    range_noise_std: float = 0.01     # meters, gaussian noise added to z
    seed: int = 42
    moving_objects: List[MovingObjectSpec] = field(default_factory=lambda: [
        MovingObjectSpec("pedestrian_1", start_xy=(8.0, 3.0), velocity_xy=(0.6, 0.0), height=1.7, footprint=0.5),
        MovingObjectSpec("vehicle_1", start_xy=(25.0, -6.0), velocity_xy=(-2.5, 0.0), height=1.6, footprint=2.0),
    ])


class SyntheticLidarGenerator:
    """Generates one or a sequence of synthetic LiDAR frames.

    Each generated point additionally carries a ground-truth semantic label
    in `PointCloud.meta["gt_labels"]` and instance id in
    `PointCloud.meta["gt_instance"]`, which the evaluation module uses to
    compute accuracy metrics without needing an external dataset.
    """

    def __init__(self, config: Optional[SyntheticSceneConfig] = None):
        self.config = config or SyntheticSceneConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def _ground_patch(self, x_range, y_range, density, z_fn, noise_std):
        area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
        n = max(int(area * density), 1)
        x = self.rng.uniform(*x_range, size=n)
        y = self.rng.uniform(*y_range, size=n)
        z = z_fn(x, y) + self.rng.normal(0, noise_std, size=n)
        return np.stack([x, y, z], axis=1)

    def _box_surface_points(self, center_xy, size_xy, z0, z1, density):
        """Sample points on the 4 vertical faces + top of a box (a wall/vehicle)."""
        cx, cy = center_xy
        sx, sy = size_xy
        pts = []
        height = z1 - z0
        for sign, axis in [(-1, "x"), (1, "x"), (-1, "y"), (1, "y")]:
            n = max(int(density * (sy if axis == "x" else sx) * height), 4)
            t = self.rng.uniform(-0.5, 0.5, size=n)
            h = self.rng.uniform(z0, z1, size=n)
            if axis == "x":
                x = np.full(n, cx + sign * sx / 2)
                y = cy + t * sy
            else:
                y = np.full(n, cy + sign * sy / 2)
                x = cx + t * sx
            pts.append(np.stack([x, y, h], axis=1))
        n_top = max(int(density * sx * sy), 4)
        x = cx + self.rng.uniform(-0.5, 0.5, size=n_top) * sx
        y = cy + self.rng.uniform(-0.5, 0.5, size=n_top) * sy
        z = np.full(n_top, z1)
        pts.append(np.stack([x, y, z], axis=1))
        return np.concatenate(pts, axis=0)

    def _column_points(self, center_xy, radius, z0, z1, density):
        cx, cy = center_xy
        n = max(int(density * 2 * np.pi * radius * (z1 - z0)), 8)
        theta = self.rng.uniform(0, 2 * np.pi, size=n)
        h = self.rng.uniform(z0, z1, size=n)
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        return np.stack([x, y, h], axis=1)

    def generate_frame(self, t: float = 0.0) -> PointCloud:
        """Generate one frame at simulation time `t` (seconds). Moving objects
        are placed at start_xy + velocity_xy * t."""
        cfg = self.config
        ext = cfg.ground_extent
        d = cfg.point_density
        noise = cfg.range_noise_std

        all_pts: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_instance: List[np.ndarray] = []
        next_instance_id = 1

        def add(pts, label, instance_id=0):
            nonlocal all_pts, all_labels, all_instance
            all_pts.append(pts)
            all_labels.append(np.full(len(pts), label, dtype=np.int32))
            all_instance.append(np.full(len(pts), instance_id, dtype=np.int32))

        # 1. Flat ground / road, with a gentle slope beyond x=35 and a curb
        #    (a 12cm step) at x=15, plus a pothole (20cm deep, 1m wide) at (5, -2).
        def ground_height(x, y):
            z = np.zeros_like(x)
            slope_mask = x > 35.0
            z = np.where(slope_mask, (x - 35.0) * np.tan(np.deg2rad(8.0)), z)
            curb_mask = x > 15.0
            z = np.where(curb_mask & ~slope_mask, 0.12, z)
            pothole_mask = (np.abs(x - 5.0) < 0.5) & (np.abs(y - (-2.0)) < 0.5)
            z = np.where(pothole_mask, z - 0.20, z)
            return z

        ground = self._ground_patch((-ext, ext), (-ext, ext), d, ground_height, noise)
        add(ground, GT_TERRAIN, 0)

        # 2. Static wall
        wall = self._box_surface_points((30.0, 12.0), (0.3, 8.0), 0.0, 2.0, density=200)
        add(wall, GT_STATIC_OBSTACLE, next_instance_id); next_instance_id += 1

        # 3. Static pole
        pole = self._column_points((18.0, -10.0), radius=0.08, z0=0.0, z1=3.0, density=300)
        add(pole, GT_STATIC_OBSTACLE, next_instance_id); next_instance_id += 1

        # 4. Overhanging structure (e.g. a sign gantry) — ground stays drivable
        #    underneath, but there is obstacle geometry above it.
        gantry = self._box_surface_points((22.0, 0.0), (0.4, 6.0), 2.4, 2.7, density=150)
        add(gantry, GT_STATIC_OBSTACLE, next_instance_id); next_instance_id += 1

        # 5. Parked (static) vehicle-like box
        parked = self._box_surface_points((10.0, 8.0), (4.2, 1.9), 0.0, 1.5, density=250)
        add(parked, GT_STATIC_OBSTACLE, next_instance_id); next_instance_id += 1

        # 6. Moving objects at time t
        for spec in cfg.moving_objects:
            cx = spec.start_xy[0] + spec.velocity_xy[0] * t
            cy = spec.start_xy[1] + spec.velocity_xy[1] * t
            if spec.footprint <= 0.8:
                pts = self._column_points((cx, cy), radius=spec.footprint / 2, z0=0.0, z1=spec.height, density=300)
            else:
                pts = self._box_surface_points((cx, cy), (spec.footprint, spec.footprint * 0.9), 0.0, spec.height, density=250)
            add(pts, GT_DYNAMIC_OBJECT, next_instance_id)
            next_instance_id += 1

        points = np.concatenate(all_pts, axis=0).astype(np.float32)
        labels = np.concatenate(all_labels, axis=0)
        instance = np.concatenate(all_instance, axis=0)

        # Range-limit to a circle so it looks like a real spinning-LiDAR sweep.
        r = np.linalg.norm(points[:, :2], axis=1)
        keep = r <= ext
        points, labels, instance = points[keep], labels[keep], instance[keep]

        intensity = self.rng.uniform(0.05, 1.0, size=len(points)).astype(np.float32)
        timestamp = np.full(len(points), t, dtype=np.float64)

        pc = PointCloud(
            points=points,
            intensity=intensity,
            timestamp=timestamp,
            frame_id="lidar",
            frame_timestamp=t,
        )
        pc.meta["gt_labels"] = labels
        pc.meta["gt_instance"] = instance
        return pc

    def generate_sequence(self, n_frames: int, dt: float = 0.2) -> List[PointCloud]:
        return [self.generate_frame(t=i * dt) for i in range(n_frames)]
