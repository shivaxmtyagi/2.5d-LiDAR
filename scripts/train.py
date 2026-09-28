#!/usr/bin/env python3
"""
Training pipeline (Module 9):

    python scripts/train.py --config configs/model.yaml

STATUS: real training code, NOT executed in the sandbox that built this
repo — no GPU there, no local copy of SemanticKITTI (it's ~80 GB and this
project's network access is restricted to package-index domains, not
general internet), and no disk budget for `torch`'s default CUDA
dependencies (see src/segmentation/architectures/pointnet2.py docstring).
Run this on your own machine, with SemanticKITTI downloaded and
`configs/model.yaml: dataset.root_dir` pointed at it.

Implements: dataset -> loader -> train/valid split -> preprocessing ->
augmentation -> model -> loss -> optimizer -> validation -> checkpoint ->
resume, per spec section 9, with checkpointing so a training run can be
killed and resumed without starting over.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import Config

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("This script requires PyTorch. Run `pip install torch` first "
          "(see src/segmentation/architectures/pointnet2.py for CPU/CUDA wheel notes).")
    sys.exit(1)

import numpy as np

from src.input.semantic_kitti import SemanticKITTIDataset, DEFAULT_TRAIN_SEQUENCES, DEFAULT_VALID_SEQUENCES
from src.preprocessing.augmentation import AugmentationConfig, augment_point_cloud
from src.segmentation.architectures.pointnet2 import PointNet2SemSeg
from src.segmentation.losses import build_loss
from src.evaluation.accuracy import compute_accuracy


class _TorchSemanticKITTIWrapper(Dataset):
    """Wraps SemanticKITTIDataset (which returns a variable-length PointCloud
    per frame) into fixed-size (points_per_sample) tensors PointNet++ needs,
    applying augmentation only in training mode."""

    def __init__(self, base: SemanticKITTIDataset, points_per_sample: int,
                 augment_cfg: AugmentationConfig | None, seed: int = 0):
        self.base = base
        self.points_per_sample = points_per_sample
        self.augment_cfg = augment_cfg
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        pc, labels = self.base[idx]
        points = pc.points
        intensity = pc.intensity if pc.intensity is not None else np.zeros(len(points), dtype=np.float32)

        if self.augment_cfg is not None:
            # Augment xyz+labels together; intensity is carried through unchanged
            # (physically, sensor reflectance doesn't change under these transforms).
            combined_idx = np.arange(len(points))
            aug_points, aug_labels = augment_point_cloud(points, labels, self.augment_cfg, self.rng)
            # augmentation may add/drop points; re-derive intensity by nearest
            # original point for any added noise points (dropout only removes).
            if len(aug_points) != len(points):
                # dropout path: labels/points already subset consistently inside
                # augment_point_cloud for the shared portion; recompute intensity
                # by matching on array identity is unsafe post-concat, so we
                # simply zero-fill intensity for any point beyond the original
                # count (only true for synthetic noise points).
                n_orig_kept = min(len(aug_points), len(points))
                intensity_used = np.zeros(len(aug_points), dtype=np.float32)
                intensity_used[:n_orig_kept] = intensity[:n_orig_kept]
            else:
                intensity_used = intensity
            points, labels, intensity = aug_points, aug_labels, intensity_used

        n = len(points)
        if n >= self.points_per_sample:
            idx_sample = self.rng.choice(n, size=self.points_per_sample, replace=False)
        else:
            pad = self.rng.choice(n, size=self.points_per_sample - n, replace=True)
            idx_sample = np.concatenate([np.arange(n), pad])

        xyz = points[idx_sample].astype(np.float32)
        feat = intensity[idx_sample].reshape(-1, 1).astype(np.float32)
        lbl = labels[idx_sample].astype(np.int64)
        return torch.from_numpy(xyz), torch.from_numpy(feat), torch.from_numpy(lbl)


def compute_class_counts(dataset: SemanticKITTIDataset, num_classes: int, max_frames_to_scan: int = 200) -> dict:
    """Estimate class frequencies from a sample of frames (scanning the
    entire dataset just for counts is wasteful for a multi-GB dataset)."""
    counts = {c: 0 for c in range(num_classes)}
    n_scan = min(len(dataset), max_frames_to_scan)
    step = max(len(dataset) // n_scan, 1)
    for i in range(0, len(dataset), step):
        _, labels = dataset[i]
        for c in range(num_classes):
            counts[c] += int((labels == c).sum())
    return counts


def train(cfg_path: str):
    cfg = Config.load(cfg_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() or cfg.training.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    train_base = SemanticKITTIDataset(cfg.dataset.root_dir, sequences=list(cfg.dataset.train_sequences))
    valid_base = SemanticKITTIDataset(cfg.dataset.root_dir, sequences=list(cfg.dataset.valid_sequences))
    print(f"Train frames: {len(train_base)}  Valid frames: {len(valid_base)}")

    num_classes = cfg.architecture.num_classes
    print("Estimating class frequencies for loss weighting...")
    class_counts = compute_class_counts(train_base, num_classes)
    print("Class counts (sampled):", class_counts)

    aug_cfg = AugmentationConfig(
        yaw_rotation=cfg.augmentation.yaw_rotation,
        max_yaw_deg=cfg.augmentation.max_yaw_deg,
        xy_translation_std=cfg.augmentation.xy_translation_std,
        jitter_std=cfg.augmentation.jitter_std,
        jitter_clip=cfg.augmentation.jitter_clip,
        scale_range=tuple(cfg.augmentation.scale_range),
        dropout_max_ratio=cfg.augmentation.dropout_max_ratio,
        noise_points_max=cfg.augmentation.noise_points_max,
    )

    train_ds = _TorchSemanticKITTIWrapper(train_base, cfg.training.points_per_sample, aug_cfg, seed=0)
    valid_ds = _TorchSemanticKITTIWrapper(valid_base, cfg.training.points_per_sample, None, seed=1)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,
                               num_workers=cfg.training.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.training.batch_size, shuffle=False,
                               num_workers=cfg.training.num_workers)

    model = PointNet2SemSeg(num_classes=num_classes, in_channels=cfg.architecture.in_channels).to(device)
    criterion = build_loss(cfg.loss, class_counts, num_classes)
    if hasattr(criterion, "weights"):
        criterion.weights = criterion.weights.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.training.lr_decay_step, gamma=cfg.training.lr_decay_gamma)

    start_epoch = 0
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if cfg.training.resume_from:
        print(f"Resuming from {cfg.training.resume_from}")
        state = torch.load(cfg.training.resume_from, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for xyz, feat, labels in train_loader:
            xyz, feat, labels = xyz.to(device), feat.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(xyz, feat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        mean_loss = epoch_loss / max(len(train_loader), 1)
        print(f"[epoch {epoch}] train_loss={mean_loss:.4f}  time={time.time()-t0:.1f}s  lr={scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % cfg.training.checkpoint_every_epochs == 0 or epoch == cfg.training.epochs - 1:
            ckpt_path = checkpoint_dir / f"segmentation_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "num_classes": num_classes,
                "in_channels": cfg.architecture.in_channels,
            }, ckpt_path)
            print(f"  saved checkpoint: {ckpt_path}")
            validate(model, valid_loader, device, num_classes)

    print("Training complete.")


def validate(model, valid_loader, device, num_classes):
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for xyz, feat, labels in valid_loader:
            xyz, feat = xyz.to(device), feat.to(device)
            logits = model(xyz, feat)
            pred = logits.argmax(dim=-1).cpu().numpy()
            all_pred.append(pred.reshape(-1))
            all_gt.append(labels.numpy().reshape(-1))
    if not all_pred:
        print("  (empty validation set)")
        return
    pred = np.concatenate(all_pred)
    gt = np.concatenate(all_gt)
    report = compute_accuracy(pred, gt, class_ids=list(range(num_classes)))
    print(f"  validation: overall_acc={report.overall_accuracy:.4f}  mIoU={report.mean_iou:.4f}")
    print(f"  per-class IoU: {report.per_class_iou}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "model.yaml"))
    args = parser.parse_args()
    train(args.config)
