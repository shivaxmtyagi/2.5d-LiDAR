"""
Abstract segmentation interface (Module 3).

Any real model — PointNet++, a sparse-conv network, MinkowskiEngine, etc. —
plugs in here by implementing `SegmentationModel.predict`. The rest of the
pipeline (terrain analysis, object detection, projection) only ever talks to
this interface, never to a concrete architecture, so swapping in a trained
checkpoint later requires zero changes elsewhere.

`SegmentationResult` deliberately mirrors the spec's required output shape:
per-point semantic_class + confidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.common.pointcloud import PointCloud


@dataclass
class SegmentationResult:
    semantic_class: np.ndarray   # (N,) int32, indexes into configs classes table
    confidence: np.ndarray       # (N,) float32 in [0, 1]
    mode: str                    # "baseline" or "trained" — surfaced to the UI,
                                  # per spec section 28: never present heuristic
                                  # output as if it came from a trained model.


class SegmentationModel(ABC):
    mode: str = "baseline"

    @abstractmethod
    def predict(self, pc: PointCloud) -> SegmentationResult:
        ...


class TrainedModelUnavailable(RuntimeError):
    pass
