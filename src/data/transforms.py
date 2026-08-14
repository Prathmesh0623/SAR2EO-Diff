"""
Lightweight, dependency-free augmentations for paired SAR/EO patches
(Stage 4 / Stage 5).

Design rule: any spatial augmentation (flip, rotate) MUST be applied
identically to the SAR patch and its paired EO patch, or the model will
learn a broken (misaligned) mapping. We therefore implement augmentations
as functions that take (sar, eo) together and return (sar, eo) together —
never augment them independently.
"""

from __future__ import annotations
import random
import numpy as np


def random_hflip(sar: np.ndarray, eo: np.ndarray, p: float = 0.5):
    if random.random() < p:
        sar = sar[:, :, ::-1].copy()
        eo = eo[:, :, ::-1].copy()
    return sar, eo


def random_vflip(sar: np.ndarray, eo: np.ndarray, p: float = 0.5):
    if random.random() < p:
        sar = sar[:, ::-1, :].copy()
        eo = eo[:, ::-1, :].copy()
    return sar, eo


def random_rot90(sar: np.ndarray, eo: np.ndarray, p: float = 0.5):
    if random.random() < p:
        k = random.choice([1, 2, 3])
        sar = np.rot90(sar, k=k, axes=(1, 2)).copy()
        eo = np.rot90(eo, k=k, axes=(1, 2)).copy()
    return sar, eo


class PairedAugment:
    """Compose the paired augmentations above. Use for training only —
    validation/test should use identity (no augmentation) so metrics are
    comparable across epochs and models."""

    def __init__(self, hflip=True, vflip=True, rot90=True):
        self.hflip = hflip
        self.vflip = vflip
        self.rot90 = rot90

    def __call__(self, sar: np.ndarray, eo: np.ndarray):
        if self.hflip:
            sar, eo = random_hflip(sar, eo)
        if self.vflip:
            sar, eo = random_vflip(sar, eo)
        if self.rot90:
            sar, eo = random_rot90(sar, eo)
        return sar, eo
