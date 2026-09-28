"""
Unified LiDAR input interface (Module 1).

Supports:
  - .bin   (KITTI-style: flat float32 array of x,y,z,intensity)
  - .npy   (Nx3 or Nx4 numpy array)
  - .ply   (ASCII or binary_little_endian, x y z [intensity])
  - .pcd   (ASCII PCD, x y z [intensity])

Real .pcd/.ply files can also be read via Open3D if it's installed
(`pip install open3d`); this loader falls back to a small dependency-free
parser for the common ASCII case so the pipeline runs without it.

The pipeline never depends on a specific format — every loader function
returns a PointCloud (src.common.pointcloud.PointCloud).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.pointcloud import PointCloud


class LidarLoadError(Exception):
    pass


def load_point_cloud(path: str) -> PointCloud:
    """Dispatch to the right parser based on file extension."""
    p = Path(path)
    if not p.exists():
        raise LidarLoadError(f"File not found: {path}")
    suffix = p.suffix.lower()
    if suffix == ".bin":
        return _load_bin(p)
    if suffix == ".npy":
        return _load_npy(p)
    if suffix == ".ply":
        return _load_ply(p)
    if suffix == ".pcd":
        return _load_pcd(p)
    raise LidarLoadError(f"Unsupported LiDAR file format: {suffix}")


def _load_bin(p: Path) -> PointCloud:
    """KITTI-style .bin: flat float32 array, 4 values per point (x,y,z,intensity)."""
    data = np.fromfile(str(p), dtype=np.float32)
    if data.size % 4 != 0:
        raise LidarLoadError(f"{p} does not look like a 4-channel .bin file (x,y,z,intensity)")
    data = data.reshape(-1, 4)
    return PointCloud(points=data[:, :3], intensity=data[:, 3], frame_id="lidar")


def _load_npy(p: Path) -> PointCloud:
    data = np.load(str(p))
    if data.ndim != 2 or data.shape[1] not in (3, 4):
        raise LidarLoadError(f"{p}: expected an (N,3) or (N,4) array, got shape {data.shape}")
    intensity = data[:, 3] if data.shape[1] == 4 else None
    return PointCloud(points=data[:, :3], intensity=intensity, frame_id="lidar")


def _load_ply(p: Path) -> PointCloud:
    try:
        import open3d as o3d  # type: ignore

        cloud = o3d.io.read_point_cloud(str(p))
        pts = np.asarray(cloud.points, dtype=np.float32)
        return PointCloud(points=pts, frame_id="lidar")
    except ImportError:
        return _load_ply_ascii(p)


def _load_ply_ascii(p: Path) -> PointCloud:
    with open(p, "r", errors="ignore") as f:
        lines = f.readlines()
    header_end = None
    n_vertices = None
    props = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
        elif line.startswith("property") and n_vertices is not None and header_end is None:
            props.append(line.split()[-1])
        elif line == "end_header":
            header_end = i
            break
    if header_end is None or n_vertices is None:
        raise LidarLoadError(f"{p}: could not parse PLY header (only ASCII PLY supported without open3d)")
    x_idx = props.index("x") if "x" in props else 0
    y_idx = props.index("y") if "y" in props else 1
    z_idx = props.index("z") if "z" in props else 2
    intensity_idx = props.index("intensity") if "intensity" in props else None

    data_lines = lines[header_end + 1: header_end + 1 + n_vertices]
    arr = np.array([[float(v) for v in line.split()] for line in data_lines], dtype=np.float32)
    points = arr[:, [x_idx, y_idx, z_idx]]
    intensity = arr[:, intensity_idx] if intensity_idx is not None else None
    return PointCloud(points=points, intensity=intensity, frame_id="lidar")


def _load_pcd(p: Path) -> PointCloud:
    try:
        import open3d as o3d  # type: ignore

        cloud = o3d.io.read_point_cloud(str(p))
        pts = np.asarray(cloud.points, dtype=np.float32)
        return PointCloud(points=pts, frame_id="lidar")
    except ImportError:
        return _load_pcd_ascii(p)


def _load_pcd_ascii(p: Path) -> PointCloud:
    with open(p, "r", errors="ignore") as f:
        lines = f.readlines()
    fields = []
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("FIELDS"):
            fields = stripped.split()[1:]
        elif stripped.startswith("DATA"):
            if "ascii" not in stripped:
                raise LidarLoadError(f"{p}: only ASCII PCD supported without open3d")
            data_start = i + 1
            break
    if data_start is None:
        raise LidarLoadError(f"{p}: could not parse PCD header")
    x_idx = fields.index("x")
    y_idx = fields.index("y")
    z_idx = fields.index("z")
    intensity_idx = fields.index("intensity") if "intensity" in fields else None

    rows = [line.split() for line in lines[data_start:] if line.strip()]
    arr = np.array(rows, dtype=np.float32)
    points = arr[:, [x_idx, y_idx, z_idx]]
    intensity = arr[:, intensity_idx] if intensity_idx is not None else None
    return PointCloud(points=points, intensity=intensity, frame_id="lidar")


def save_point_cloud_npy(pc: PointCloud, path: str) -> None:
    """Convenience writer used by the synthetic generator / tests."""
    if pc.intensity is not None:
        arr = np.concatenate([pc.points, pc.intensity.reshape(-1, 1)], axis=1)
    else:
        arr = pc.points
    np.save(path, arr)
