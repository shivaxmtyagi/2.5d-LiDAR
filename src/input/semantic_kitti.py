"""
SemanticKITTI dataset adapter (Module 29).

STATUS: real, structurally-correct parsing code for SemanticKITTI's actual
on-disk binary formats (velodyne .bin, labels .label — see below), written
against the dataset's public format spec. It has NOT been run against a
real copy of SemanticKITTI in the sandbox that built this repo (the dataset
is ~80 GB and this environment has no general internet access — only a
fixed allowlist of package-index domains). You will need to point
`root_dir` at your own local copy.

Expected directory layout (this is SemanticKITTI's actual layout, not
something invented for this project):

    <root_dir>/
      sequences/
        00/
          velodyne/000000.bin  000001.bin  ...
          labels/000000.label  000001.label  ...   (absent for the test split)
        01/
        ...
        21/

velodyne/*.bin: flat float32 array, 4 values per point (x, y, z, remission).
labels/*.label: flat uint32 array, one value per point, where the LOW 16
bits are the semantic class id and the HIGH 16 bits are the instance id
(`label = semantic | (instance << 16)`) — this bit layout is fixed by the
dataset format itself.

`raw_to_canonical`: SemanticKITTI ships ~28 raw semantic ids (see the
dataset's own `semantic-kitti.yaml` for the authoritative list). This
project's pipeline only needs the 4-class table in configs/default.yaml, so
`DEFAULT_RAW_TO_CANONICAL` below remaps the well-known raw ids accordingly.
Double-check that table against your copy of `semantic-kitti.yaml` before
training — label-id mistakes silently produce a wrong but plausible-looking
model, which is exactly the kind of unverified claim this project is
trying to avoid (see README > Limitations). Pass your own `raw_to_canonical`
dict to override it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.common.pointcloud import PointCloud
from src.input.dataset_base import BaseLidarDataset

UNKNOWN, TERRAIN, STATIC_OBSTACLE, DYNAMIC_OBJECT = 0, 1, 2, 3

# Best-effort mapping from SemanticKITTI's raw label ids to this project's
# 4-class table, based on the dataset's published label set. VERIFY against
# your own sequences/semantic-kitti.yaml (learning_map) before trusting
# training results — see module docstring.
DEFAULT_RAW_TO_CANONICAL: Dict[int, int] = {
    0: UNKNOWN,     # unlabeled
    1: UNKNOWN,     # outlier
    # -- ground / terrain --
    40: TERRAIN,    # road
    44: TERRAIN,    # parking
    48: TERRAIN,    # sidewalk
    49: TERRAIN,    # other-ground
    72: TERRAIN,    # terrain
    # -- static structures --
    50: STATIC_OBSTACLE,   # building
    51: STATIC_OBSTACLE,   # fence
    52: STATIC_OBSTACLE,   # other-structure
    70: STATIC_OBSTACLE,   # vegetation
    71: STATIC_OBSTACLE,   # trunk
    80: STATIC_OBSTACLE,   # pole
    81: STATIC_OBSTACLE,   # traffic-sign
    99: STATIC_OBSTACLE,   # other-object
    # -- parked / stationary vehicles & people (static at capture time) --
    10: STATIC_OBSTACLE,   # car
    11: STATIC_OBSTACLE,   # bicycle
    13: STATIC_OBSTACLE,   # bus
    15: STATIC_OBSTACLE,   # motorcycle
    16: STATIC_OBSTACLE,   # on-rails
    18: STATIC_OBSTACLE,   # truck
    20: STATIC_OBSTACLE,   # other-vehicle
    30: STATIC_OBSTACLE,   # person
    31: STATIC_OBSTACLE,   # bicyclist
    32: STATIC_OBSTACLE,   # motorcyclist
    # -- explicitly moving instances (SemanticKITTI's own "moving-*" classes) --
    252: DYNAMIC_OBJECT,   # moving-car
    253: DYNAMIC_OBJECT,   # moving-bicyclist
    254: DYNAMIC_OBJECT,   # moving-person
    255: DYNAMIC_OBJECT,   # moving-motorcyclist
    256: DYNAMIC_OBJECT,   # moving-on-rails
    257: DYNAMIC_OBJECT,   # moving-bus
    258: DYNAMIC_OBJECT,   # moving-truck
    259: DYNAMIC_OBJECT,   # moving-other-vehicle
}

DEFAULT_TRAIN_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
DEFAULT_VALID_SEQUENCES = ["08"]


class SemanticKITTIDataset(BaseLidarDataset):
    def __init__(self, root_dir: str, sequences: Optional[List[str]] = None,
                 raw_to_canonical: Optional[Dict[int, int]] = None, require_labels: bool = True):
        self.root_dir = Path(root_dir)
        self.sequences = sequences or DEFAULT_TRAIN_SEQUENCES
        self.raw_to_canonical = raw_to_canonical or DEFAULT_RAW_TO_CANONICAL
        self.require_labels = require_labels
        self.frames: List[Tuple[Path, Optional[Path]]] = self._index()

    def _index(self) -> List[Tuple[Path, Optional[Path]]]:
        frames = []
        for seq in self.sequences:
            seq_dir = self.root_dir / "sequences" / seq
            velodyne_dir = seq_dir / "velodyne"
            labels_dir = seq_dir / "labels"
            if not velodyne_dir.exists():
                raise FileNotFoundError(
                    f"SemanticKITTI sequence {seq} not found at {velodyne_dir} — "
                    f"check root_dir points at the folder that CONTAINS 'sequences/'."
                )
            for bin_path in sorted(velodyne_dir.glob("*.bin")):
                label_path = labels_dir / (bin_path.stem + ".label")
                if self.require_labels and not label_path.exists():
                    continue  # e.g. the official test split ships no labels
                frames.append((bin_path, label_path if label_path.exists() else None))
        if not frames:
            raise FileNotFoundError(
                f"No SemanticKITTI frames found under {self.root_dir} for sequences "
                f"{self.sequences}. Confirm the dataset is actually downloaded/extracted there."
            )
        return frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[PointCloud, np.ndarray]:
        bin_path, label_path = self.frames[idx]
        raw = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
        points, intensity = raw[:, :3], raw[:, 3]

        if label_path is not None:
            raw_labels = np.fromfile(str(label_path), dtype=np.uint32)
            semantic_raw = (raw_labels & 0xFFFF).astype(np.int32)  # low 16 bits, per format spec
            canonical = np.array(
                [self.raw_to_canonical.get(int(c), UNKNOWN) for c in semantic_raw], dtype=np.int32
            )
        else:
            canonical = np.zeros(len(points), dtype=np.int32)

        pc = PointCloud(points=points, intensity=intensity, frame_id="lidar")
        return pc, canonical


def build_semantic_kitti_splits(root_dir: str, raw_to_canonical: Optional[Dict[int, int]] = None):
    """Convenience factory returning (train_dataset, valid_dataset) using the
    dataset's own published train/valid sequence split (test sequences 11-21
    ship without labels and aren't usable for supervised training/eval)."""
    train = SemanticKITTIDataset(root_dir, sequences=DEFAULT_TRAIN_SEQUENCES, raw_to_canonical=raw_to_canonical)
    valid = SemanticKITTIDataset(root_dir, sequences=DEFAULT_VALID_SEQUENCES, raw_to_canonical=raw_to_canonical)
    return train, valid
