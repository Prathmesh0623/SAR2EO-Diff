"""Tests for evaluation metrics using synthetic tensors (no dataset needed)."""
import torch
from src.evaluation.metrics import compute_psnr, compute_ssim


def test_psnr_identical_images_is_high():
    img = torch.rand(3, 32, 32) * 2 - 1
    psnr = compute_psnr(img, img)
    assert psnr > 40 or psnr == float("inf")


def test_ssim_identical_images_is_one():
    img = torch.rand(3, 32, 32) * 2 - 1
    ssim = compute_ssim(img, img)
    assert abs(ssim - 1.0) < 1e-3


def test_psnr_random_images_is_lower_than_identical():
    img1 = torch.rand(3, 32, 32) * 2 - 1
    img2 = torch.rand(3, 32, 32) * 2 - 1
    psnr_diff = compute_psnr(img1, img2)
    psnr_same = compute_psnr(img1, img1)
    assert psnr_diff < psnr_same
