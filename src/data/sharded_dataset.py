"""
Dataset loader for the "sen12ms-asia" Kaggle mirror by krishnaanchal, which
packages SEN12MS as pre-cropped 256x256 PyTorch tensor shards rather than
the original GeoTIFF/ROI folder layout that SEN12MSPairedDataset (in
dataset.py) expects.

Each shard_XXX.pt file is a dict:
    "sar":   (N, 3, 256, 256) float16 -- SAR channels, empirically observed
             value range roughly [-81, 1522] on this mirror (NOT the -25..5 dB
             range assumed elsewhere in this repo -- this dataset stores SAR
             on its own scale, so we normalize off the observed range instead
             of the physical dB formula in preprocessing.py).
    "opt":   (N, 3, 256, 256) float16 -- EO/RGB channels, observed range
             roughly [136, 22304] (Sentinel-2-like reflectance, just scaled
             differently than the standard 0-10000 range).
    "label": (N,) uint8 -- one IGBP land-cover class per patch (scene-level
             label, not a pixel-wise mask). Values observed: a subset of
             {1..17}. Useful later for Stage 12 (semantic consistency) as a
             classification target, but is NOT a segmentation mask, so
             src/models/segmentation.py's per-pixel design does not directly
             apply to this dataset without adaptation.

IMPORTANT: the normalization ranges below are set from ONE inspected shard,
not the full dataset. Before trusting them for a real training run, check a
few more shards (or all of them) for min/max to make sure no shard has
outlier values outside this range -- see the __main__ block at the bottom
of this file for a ready-to-run check.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset

# Empirically observed on shard_000.pt -- see docstring above.
SAR_MIN, SAR_MAX = -81.0, 1522.0
OPT_MIN, OPT_MAX = 136.0, 22304.0


def _normalize(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clip to [lo, hi] then rescale to [-1, 1], matching the convention
    used by normalize_sar/normalize_eo in preprocessing.py."""
    x = x.float().clamp(lo, hi)
    x = (x - lo) / (hi - lo)      # -> [0, 1]
    return x * 2.0 - 1.0            # -> [-1, 1]


class ShardedSEN12MSDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
        transform: Optional[Callable] = None,
        subset_size: Optional[int] = None,
    ):
        """
        Args:
            root: directory containing shard_XXX.pt files
                  (e.g. .../asian_sen12ms_shards)
            split: "train", "val", or "test"
            transform: optional PairedAugment-style callable(sar, eo) -> (sar, eo).
                       NOTE: transforms.py's functions expect numpy arrays;
                       this dataset works in torch tensors, so a transform
                       passed here must accept/return tensors, or convert
                       internally. Left as None by default until Stage 5
                       augmentation is re-validated against this format.
            subset_size: if set, only index the first N samples total
                         (across shards) -- use for a fast pipeline smoke
                         test before pointing at the full dataset.
        """
        self.root = Path(root)
        self.transform = transform

        shard_paths = sorted(self.root.glob("shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.pt files found under {root}")

        # Build a flat index of (shard_path, index_within_shard) without
        # loading every shard into memory up front -- we only need each
        # shard's sample COUNT here, so open once, record length, and let
        # __getitem__ load the actual tensors lazily per shard.
        self._shard_paths = []
        self._shard_lengths = []
        for p in shard_paths:
            try:
                n = torch.load(p, map_location="cpu")["sar"].shape[0]
            except Exception as e:
                print(f"[ShardedSEN12MSDataset] Skipping unreadable shard {p.name}: {e}")
                continue
            self._shard_paths.append(p)
            self._shard_lengths.append(n)

        if not self._shard_paths:
            raise RuntimeError("No readable shards found -- all shard files failed to load.")

        total = sum(self._shard_lengths)
        all_indices = list(range(total))

        if subset_size is not None:
            all_indices = all_indices[:subset_size]
            total = len(all_indices)

        # Reproducible split
        g = torch.Generator().manual_seed(seed)
        shuffled = torch.randperm(total, generator=g).tolist()
        n_train = int(total * train_frac)
        n_val = int(total * val_frac)

        if split == "train":
            sel = shuffled[:n_train]
        elif split == "val":
            sel = shuffled[n_train:n_train + n_val]
        elif split == "test":
            sel = shuffled[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self._global_indices = [all_indices[i] for i in sel]

        # Simple LRU-ish cache: keep the most recently used shard's tensors
        # in memory so consecutive samples from the same shard don't
        # re-read from disk every __getitem__ call.
        self._cache_shard_idx = None
        self._cache_data = None

    def __len__(self):
        return len(self._global_indices)

    def _locate(self, global_idx: int):
        """Map a flat global index to (shard_index, local_index_in_shard)."""
        cumulative = 0
        for shard_idx, length in enumerate(self._shard_lengths):
            if global_idx < cumulative + length:
                return shard_idx, global_idx - cumulative
            cumulative += length
        raise IndexError(global_idx)

    def _load_shard(self, shard_idx: int):
        if self._cache_shard_idx != shard_idx:
            self._cache_data = torch.load(self._shard_paths[shard_idx], map_location="cpu")
            self._cache_shard_idx = shard_idx
        return self._cache_data

    def __getitem__(self, idx):
        global_idx = self._global_indices[idx]
        shard_idx, local_idx = self._locate(global_idx)
        data = self._load_shard(shard_idx)

        sar = _normalize(data["sar"][local_idx], SAR_MIN, SAR_MAX)
        eo = _normalize(data["opt"][local_idx], OPT_MIN, OPT_MAX)
        label = int(data["label"][local_idx].item())

        if self.transform is not None:
            sar, eo = self.transform(sar, eo)

        return {
            "sar": sar,          # (3, 256, 256) float32, [-1, 1]
            "eo": eo,            # (3, 256, 256) float32, [-1, 1]
            "label": label,       # IGBP land-cover class, scene-level
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smoke-test the sharded dataset loader.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--check_all_shards_range", action="store_true",
                         help="Scan every shard's min/max to verify SAR_MIN/MAX, "
                              "OPT_MIN/MAX above cover the full dataset, not just shard_000.")
    args = parser.parse_args()

    ds = ShardedSEN12MSDataset(root=args.root, split="train", subset_size=20)
    print(f"Train samples (subset): {len(ds)}")
    sample = ds[0]
    print("SAR shape:", sample["sar"].shape, "range:", sample["sar"].min().item(), sample["sar"].max().item())
    print("EO shape:", sample["eo"].shape, "range:", sample["eo"].min().item(), sample["eo"].max().item())
    print("Label:", sample["label"])

    if args.check_all_shards_range:
        root = Path(args.root)
        sar_min, sar_max = float("inf"), float("-inf")
        opt_min, opt_max = float("inf"), float("-inf")
        for p in sorted(root.glob("shard_*.pt")):
            d = torch.load(p, map_location="cpu")
            sar_min = min(sar_min, d["sar"].min().item())
            sar_max = max(sar_max, d["sar"].max().item())
            opt_min = min(opt_min, d["opt"].min().item())
            opt_max = max(opt_max, d["opt"].max().item())
            print(f"{p.name}: sar[{d['sar'].min().item():.1f},{d['sar'].max().item():.1f}] "
                  f"opt[{d['opt'].min().item():.1f},{d['opt'].max().item():.1f}]")
        print(f"\nGLOBAL: sar[{sar_min:.1f},{sar_max:.1f}] opt[{opt_min:.1f},{opt_max:.1f}]")
        print("Compare against SAR_MIN/MAX and OPT_MIN/MAX constants in this file "
              "and update them if these global values differ.")


class ShardAwareSampler(torch.utils.data.Sampler):
    """A sampler that avoids constant shard-swapping.

    Default DataLoader shuffle=True picks a uniformly random global index
    each step, which -- with 13+ shards of 1.5GB each and only one shard
    cached in memory at a time (see ShardedSEN12MSDataset._load_shard) --
    forces a full shard reload from disk on nearly every sample. That's
    extremely slow.

    This sampler instead: groups the dataset's indices by which shard they
    belong to, shuffles the ORDER of shards each epoch, and shuffles
    samples WITHIN each shard -- so consecutive batches mostly hit the
    shard already cached in memory, while still giving a different (and
    still random) sample order each epoch.
    """

    def __init__(self, dataset: "ShardedSEN12MSDataset", seed: int = 42):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        # Group this dataset's (already split-filtered) indices by shard.
        groups: dict[int, list[int]] = {}
        for pos, global_idx in enumerate(self.dataset._global_indices):
            shard_idx, _ = self.dataset._locate(global_idx)
            groups.setdefault(shard_idx, []).append(pos)

        g = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(len(groups), generator=g).tolist()

        order = []
        for shard_idx in shard_order:
            positions = groups[shard_idx]
            perm = torch.randperm(len(positions), generator=g).tolist()
            order.extend(positions[i] for i in perm)
        return iter(order)

    def __len__(self):
        return len(self.dataset)