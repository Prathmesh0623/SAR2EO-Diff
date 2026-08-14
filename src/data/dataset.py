"""
PyTorch Dataset / DataLoader for paired SAR-EO patches (Stage 5).

This is written against the SEN12MS directory layout, where scenes are
organized as:

    ROOT/
      ROIsXXXX_<season>/
        s1_<scene_id>/   <- Sentinel-1 GeoTIFFs (VV, VH bands)
        s2_<scene_id>/   <- Sentinel-2 GeoTIFFs (13 bands)
        lc_<scene_id>/   <- MODIS land-cover GeoTIFFs (optional, for
                             semantic consistency in Stage 9)

Exact folder naming varies by SEN12MS release/mirror — verify against your
downloaded copy in Stage 2/3 and adjust `_discover_pairs` if needed. This
class is deliberately kept dataset-format-agnostic at the interface level
(returns tensors, not file paths) so later stages don't need to change.

Do NOT run this against the full SEN12MS dataset on first try — build a
tiny `subset_size` first (Stage 27, Phase 1: "does the pipeline work?").
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError:  # rasterio is only needed when actually reading .tif files
    rasterio = None

from .preprocessing import normalize_sar, normalize_eo, sar_to_db, extract_patches


# Sentinel-2 band order used by SEN12MS (13 bands). We only need a subset
# for the RGB-first configuration described in the brief.
S2_BAND_INDEX = {
    "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
    "B08": 7, "B8A": 8, "B09": 9, "B10": 10, "B11": 11, "B12": 12,
}
# RGB = B04 (Red), B03 (Green), B02 (Blue)
RGB_BAND_ORDER = ["B04", "B03", "B02"]


class SEN12MSPairedDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: int = 128,
        sar_channels: tuple[str, ...] = ("VV", "VH"),
        eo_bands: tuple[str, ...] = tuple(RGB_BAND_ORDER),
        transform: Optional[Callable] = None,
        seed: int = 42,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        subset_size: Optional[int] = None,
        sar_already_db: bool = False,
    ):
        """
        Args:
            root: path to SEN12MS root directory (e.g. Kaggle dataset input path)
            split: one of "train", "val", "test"
            patch_size: side length for cropped patches
            sar_channels: which SAR polarizations to load (VV, VH, or both)
            eo_bands: which Sentinel-2 bands to load as output channels
            transform: optional PairedAugment-style callable(sar, eo) -> (sar, eo)
            subset_size: if set, only use this many scene-pairs (useful for
                Phase 1 "pipeline works?" smoke tests per Stage 27)
            sar_already_db: set True if your SEN12MS mirror already stores
                SAR in dB (skip the linear->dB conversion)
        """
        if rasterio is None:
            raise ImportError(
                "rasterio is required to read SEN12MS GeoTIFFs. "
                "Install with `pip install rasterio` (already in requirements.txt)."
            )

        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.sar_channels = sar_channels
        self.eo_bands = eo_bands
        self.transform = transform
        self.sar_already_db = sar_already_db

        pairs = self._discover_pairs()
        if subset_size is not None:
            pairs = pairs[:subset_size]

        self.pairs = self._split_pairs(pairs, split, train_frac, val_frac, seed)

    # ---- discovery -------------------------------------------------------

    def _discover_pairs(self) -> list[tuple[Path, Path]]:
        """Walk the SEN12MS root and pair up each s1_* scene with its
        matching s2_* scene by scene id.

        Returns a list of (sar_path, eo_path) GeoTIFF file pairs.
        Adjust the glob patterns here if your SEN12MS copy uses a different
        folder/file naming convention — verify in Stage 2 first.
        """
        pairs = []
        for roi_dir in sorted(self.root.glob("ROIs*")):
            s1_dirs = sorted(roi_dir.glob("s1_*"))
            for s1_dir in s1_dirs:
                s2_dir = Path(str(s1_dir).replace("s1_", "s2_"))
                if not s2_dir.exists():
                    continue
                for s1_file in sorted(s1_dir.glob("*.tif")):
                    s2_file = s2_dir / s1_file.name.replace("s1_", "s2_")
                    if s2_file.exists():
                        pairs.append((s1_file, s2_file))
        return pairs

    def _split_pairs(self, pairs, split, train_frac, val_frac, seed):
        rng = np.random.RandomState(seed)
        idx = np.arange(len(pairs))
        rng.shuffle(idx)
        n_train = int(len(idx) * train_frac)
        n_val = int(len(idx) * val_frac)
        if split == "train":
            sel = idx[:n_train]
        elif split == "val":
            sel = idx[n_train:n_train + n_val]
        elif split == "test":
            sel = idx[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
        return [pairs[i] for i in sel]

    # ---- loading -----------------------------------------------------------

    def _load_sar(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            bands = []
            band_names = ["VV", "VH"]
            for ch in self.sar_channels:
                band_idx = band_names.index(ch) + 1  # rasterio is 1-indexed
                bands.append(src.read(band_idx))
            arr = np.stack(bands, axis=0).astype(np.float32)
        if not self.sar_already_db:
            arr = sar_to_db(arr)
        return normalize_sar(arr)

    def _load_eo(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            bands = []
            for band_name in self.eo_bands:
                band_idx = S2_BAND_INDEX[band_name] + 1
                bands.append(src.read(band_idx))
            arr = np.stack(bands, axis=0).astype(np.float32)
        return normalize_eo(arr)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sar_path, eo_path = self.pairs[idx]
        sar = self._load_sar(sar_path)   # (C_sar, H, W), normalized [-1, 1]
        eo = self._load_eo(eo_path)      # (C_eo, H, W), normalized [-1, 1]

        # Center-crop / random-crop to patch_size (assumes source tiles are
        # >= patch_size; SEN12MS tiles are typically 256x256).
        h, w = sar.shape[1], sar.shape[2]
        if h > self.patch_size or w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size + 1) if self.split == "train" else (h - self.patch_size) // 2
            left = np.random.randint(0, w - self.patch_size + 1) if self.split == "train" else (w - self.patch_size) // 2
            sar = sar[:, top:top + self.patch_size, left:left + self.patch_size]
            eo = eo[:, top:top + self.patch_size, left:left + self.patch_size]

        if self.transform is not None and self.split == "train":
            sar, eo = self.transform(sar, eo)

        return {
            "sar": torch.from_numpy(sar.copy()).float(),
            "eo": torch.from_numpy(eo.copy()).float(),
            "sar_path": str(sar_path),
            "eo_path": str(eo_path),
        }
