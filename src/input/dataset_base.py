"""
Dataset abstraction (Module 29). Any concrete dataset (SemanticKITTI,
nuScenes, a custom capture) implements `BaseLidarDataset`, so
scripts/train.py and the evaluation code never import a dataset-specific
module directly — only this interface. src.input.synthetic's generator is
deliberately NOT wrapped in this interface: it's an infinite generator for
demos/tests, not an indexed, split-able training dataset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

from src.common.pointcloud import PointCloud


class BaseLidarDataset(ABC):
    """A dataset item is (PointCloud, point_labels) where point_labels is an
    (N,) int array already remapped into this project's canonical class
    table (configs/default.yaml: classes)."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[PointCloud, np.ndarray]:
        ...

    def class_table(self) -> Dict[int, str]:
        """Override if a dataset ships its own class names; defaults to
        this project's canonical table."""
        return {0: "unknown", 1: "terrain", 2: "static_obstacle", 3: "dynamic_object"}
