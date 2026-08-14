"""
Evaluation metrics (Stage 15).

Image-quality metrics:
    PSNR  — pixel-level fidelity (higher = closer to ground truth)
    SSIM  — structural similarity (higher = better preserved structure)
    LPIPS — learned perceptual similarity (lower = more perceptually similar;
            correlates better with human judgment than PSNR/SSIM alone)
    FID   — Frechet Inception Distance between generated/real distributions
            (lower = generated images are distributionally closer to real ones;
            requires a reasonably large sample, so only compute at the end)

Semantic-usefulness metrics (computed by running a segmentation network over
generated vs. real EO and comparing to land-cover labels):
    IoU / mIoU — intersection-over-union per class / mean over classes
    F1         — harmonic mean of precision/recall per class
    Pixel accuracy — fraction of correctly classified pixels

IMPORTANT: none of these functions invent or assume results. They only
compute a number from tensors you pass in. Actual reported numbers must
come from running these against real model outputs on Kaggle (Stage 14 —
ablation studies — explicitly forbids fabricated numbers).
"""

from __future__ import annotations
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_numpy_hwc(img: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) tensor in [-1, 1] to an (H, W, C) numpy array in [0, 1]."""
    img = (img.detach().cpu().float() + 1.0) / 2.0
    img = img.clamp(0, 1).numpy()
    return np.transpose(img, (1, 2, 0))


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = to_numpy_hwc(pred)
    target_np = to_numpy_hwc(target)
    return float(peak_signal_noise_ratio(target_np, pred_np, data_range=1.0))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = to_numpy_hwc(pred)
    target_np = to_numpy_hwc(target)
    return float(structural_similarity(target_np, pred_np, channel_axis=-1, data_range=1.0))


class LPIPSMetric:
    """Lazily-initialized LPIPS wrapper (avoids importing/loading the LPIPS
    backbone network unless it's actually used)."""

    def __init__(self, net: str = "alex", device: str = "cpu"):
        import lpips
        self.model = lpips.LPIPS(net=net).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        # lpips expects (B, C, H, W) in [-1, 1], which matches our data prep
        pred_b = pred.unsqueeze(0).to(self.device)
        target_b = target.unsqueeze(0).to(self.device)
        return float(self.model(pred_b, target_b).item())


def compute_confusion_stats(pred_labels: np.ndarray, true_labels: np.ndarray, num_classes: int):
    """Accumulate a confusion matrix for IoU/F1/pixel-accuracy computation."""
    mask = (true_labels >= 0) & (true_labels < num_classes)
    conf = np.bincount(
        num_classes * true_labels[mask].astype(int) + pred_labels[mask].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return conf


def iou_from_confusion(conf: np.ndarray) -> np.ndarray:
    intersection = np.diag(conf)
    union = conf.sum(axis=1) + conf.sum(axis=0) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = intersection / union
    return iou  # per-class; caller takes nanmean for mIoU


def pixel_accuracy_from_confusion(conf: np.ndarray) -> float:
    return float(np.diag(conf).sum() / conf.sum())
