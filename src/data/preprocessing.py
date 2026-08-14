"""
Preprocessing utilities for SAR and EO imagery (Stage 4).

SAR (Sentinel-1, VV/VH):
    Raw values are backscatter intensities, typically stored in linear power
    or already converted to dB. They span a very large dynamic range and are
    affected by speckle noise. We convert to dB (if not already) and clip
    extreme outliers before normalizing to [-1, 1] for the network.

EO (Sentinel-2, optical bands):
    Reflectance values are typically stored as 16-bit integers (e.g. 0-10000
    for Sentinel-2 L2A). We clip to a reasonable reflectance range and scale
    to [-1, 1] to match the SAR normalization range (helps stable training).

These functions are dataset-agnostic: they operate on numpy arrays, so they
can be reused whether the source is SEN12MS .tif/.h5 patches or another
SAR/EO paired dataset.
"""

from __future__ import annotations
import numpy as np

# ---- SAR ----------------------------------------------------------------

SAR_DB_CLIP_MIN = -25.0
SAR_DB_CLIP_MAX = 5.0


def sar_to_db(sar_linear: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert linear-power SAR backscatter to decibels (dB).

    Skip this step if your source data is already in dB (check dataset docs
    for SEN12MS — some releases distribute SAR already in dB).
    """
    return 10.0 * np.log10(np.clip(sar_linear, eps, None))


def normalize_sar(sar_db: np.ndarray,
                   clip_min: float = SAR_DB_CLIP_MIN,
                   clip_max: float = SAR_DB_CLIP_MAX) -> np.ndarray:
    """Clip SAR dB values to a sane range and rescale to [-1, 1].

    The clip range (-25 dB to +5 dB) is a common working range for
    Sentinel-1 VV/VH over land; revisit this after Stage 5 (data
    exploration) once you've plotted the actual value histograms for your
    subset of SEN12MS.
    """
    clipped = np.clip(sar_db, clip_min, clip_max)
    normed = (clipped - clip_min) / (clip_max - clip_min)  # -> [0, 1]
    return normed * 2.0 - 1.0                               # -> [-1, 1]


def denormalize_sar(sar_norm: np.ndarray,
                     clip_min: float = SAR_DB_CLIP_MIN,
                     clip_max: float = SAR_DB_CLIP_MAX) -> np.ndarray:
    """Inverse of normalize_sar, for visualization."""
    zero_one = (sar_norm + 1.0) / 2.0
    return zero_one * (clip_max - clip_min) + clip_min


# ---- EO -------------------------------------------------------------------

EO_REFLECTANCE_CLIP_MAX = 4000.0  # generous upper bound for typical land scenes


def normalize_eo(eo_raw: np.ndarray,
                  clip_max: float = EO_REFLECTANCE_CLIP_MAX) -> np.ndarray:
    """Clip Sentinel-2 reflectance and rescale to [-1, 1].

    NOTE: verify the actual value range for your SEN12MS subset in Stage 5
    before committing to clip_max — bright targets (snow, clouds, urban)
    can exceed 4000; adjust accordingly.
    """
    clipped = np.clip(eo_raw, 0.0, clip_max)
    zero_one = clipped / clip_max
    return zero_one * 2.0 - 1.0


def denormalize_eo(eo_norm: np.ndarray,
                    clip_max: float = EO_REFLECTANCE_CLIP_MAX) -> np.ndarray:
    """Inverse of normalize_eo, for visualization (returns 0-1 range,
    suitable for direct matplotlib imshow)."""
    zero_one = (eo_norm + 1.0) / 2.0
    return np.clip(zero_one, 0.0, 1.0)


# ---- Patch extraction -------------------------------------------------------

def extract_patches(image: np.ndarray, patch_size: int, stride: int | None = None):
    """Extract non-overlapping (or strided) square patches from a CHW image.

    Args:
        image: array of shape (C, H, W)
        patch_size: side length of each square patch
        stride: step between patches; defaults to patch_size (non-overlapping)

    Returns:
        list of (C, patch_size, patch_size) arrays
    """
    stride = stride or patch_size
    c, h, w = image.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(image[:, y:y + patch_size, x:x + patch_size])
    return patches
