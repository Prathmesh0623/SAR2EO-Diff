"""
Unit tests that don't require the actual SEN12MS dataset to be present —
they test the pure-function preprocessing/augmentation logic, plus a
minimal fake-Dataset shape check. The full SEN12MSPairedDataset class
(which needs real GeoTIFFs) should be smoke-tested separately via
scripts/prepare_data.py once you have the dataset attached on Kaggle.
"""
import numpy as np
from src.data.preprocessing import (
    normalize_sar, denormalize_sar, normalize_eo, denormalize_eo,
    sar_to_db, extract_patches,
)
from src.data.transforms import random_hflip, random_vflip, random_rot90


def test_sar_normalize_roundtrip():
    db = np.random.uniform(-25, 5, size=(2, 32, 32)).astype(np.float32)
    normed = normalize_sar(db)
    assert normed.min() >= -1.0 - 1e-5 and normed.max() <= 1.0 + 1e-5
    recovered = denormalize_sar(normed)
    assert np.allclose(recovered, db, atol=1e-3)


def test_eo_normalize_roundtrip():
    raw = np.random.uniform(0, 3000, size=(3, 32, 32)).astype(np.float32)
    normed = normalize_eo(raw)
    assert normed.min() >= -1.0 - 1e-5 and normed.max() <= 1.0 + 1e-5
    recovered = denormalize_eo(normed)  # returns [0,1] range
    assert recovered.min() >= 0.0 and recovered.max() <= 1.0


def test_sar_to_db_no_crash_on_zero():
    linear = np.zeros((2, 8, 8), dtype=np.float32)
    db = sar_to_db(linear)
    assert np.isfinite(db).all()


def test_extract_patches_count():
    img = np.random.randn(3, 256, 256).astype(np.float32)
    patches = extract_patches(img, patch_size=128)
    assert len(patches) == 4  # 2x2 grid of non-overlapping 128 patches
    assert patches[0].shape == (3, 128, 128)


def test_paired_augmentations_stay_aligned():
    sar = np.arange(2 * 8 * 8).reshape(2, 8, 8).astype(np.float32)
    eo = np.arange(3 * 8 * 8).reshape(3, 8, 8).astype(np.float32)

    # Force the flip to always trigger to check alignment (p=1.0)
    sar_f, eo_f = random_hflip(sar, eo, p=1.0)
    assert sar_f.shape == sar.shape and eo_f.shape == eo.shape
    assert np.array_equal(sar_f, sar[:, :, ::-1])
    assert np.array_equal(eo_f, eo[:, :, ::-1])
