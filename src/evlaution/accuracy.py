"""ML accuracy metrics (Module 24) computed against the synthetic
generator's ground-truth labels (src.input.synthetic sets pc.meta["gt_labels"]).
For a real dataset (KITTI/SemanticKITTI/nuScenes) this same function works
unchanged as long as the dataset adapter populates the same meta key."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class AccuracyReport:
    per_class_iou: Dict[int, float]
    mean_iou: float
    precision: Dict[int, float]
    recall: Dict[int, float]
    f1: Dict[int, float]
    overall_accuracy: float


def compute_accuracy(pred_labels: np.ndarray, gt_labels: np.ndarray, class_ids) -> AccuracyReport:
    assert pred_labels.shape == gt_labels.shape, "prediction/ground-truth length mismatch"
    n = len(gt_labels)
    per_class_iou, precision, recall, f1 = {}, {}, {}, {}
    for c in class_ids:
        pred_mask = pred_labels == c
        gt_mask = gt_labels == c
        tp = int(np.logical_and(pred_mask, gt_mask).sum())
        fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
        fn = int(np.logical_and(~pred_mask, gt_mask).sum())
        union = tp + fp + fn
        per_class_iou[c] = tp / union if union > 0 else float("nan")
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        p, r = precision[c], recall[c]
        f1[c] = (2 * p * r / (p + r)) if (p + r) > 0 and not np.isnan(p) and not np.isnan(r) else float("nan")

    valid_ious = [v for v in per_class_iou.values() if not np.isnan(v)]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else float("nan")
    overall_accuracy = float((pred_labels == gt_labels).sum() / n) if n > 0 else float("nan")

    return AccuracyReport(
        per_class_iou=per_class_iou,
        mean_iou=mean_iou,
        precision=precision,
        recall=recall,
        f1=f1,
        overall_accuracy=overall_accuracy,
    )
