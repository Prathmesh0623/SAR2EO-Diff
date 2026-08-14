"""Shape/sanity tests for all four network architectures. These do NOT
validate that training converges (that requires real data + GPU time on
Kaggle) — they only guarantee the forward pass runs and produces the
expected tensor shapes, so a code change doesn't silently break the model."""
import torch
from src.models.unet import UNet
from src.models.pix2pix import Pix2PixGenerator, PatchGANDiscriminator
from src.models.diffusion import ConditionalDiffusionUNet, GaussianDiffusion
from src.models.segmentation import SimpleSegmentationNet


def test_unet_shapes():
    model = UNet(in_channels=2, out_channels=3, base_channels=16, depth=2)
    x = torch.randn(2, 2, 64, 64)
    out = model(x)
    assert out.shape == (2, 3, 64, 64)
    assert out.min() >= -1.0 - 1e-4 and out.max() <= 1.0 + 1e-4  # tanh output


def test_pix2pix_shapes():
    g = Pix2PixGenerator(in_channels=2, out_channels=3, base_channels=16, depth=2)
    d = PatchGANDiscriminator(in_channels=5, base_channels=16)
    sar = torch.randn(2, 2, 64, 64)
    fake_eo = g(sar)
    assert fake_eo.shape == (2, 3, 64, 64)
    patch_out = d(sar, fake_eo)
    assert patch_out.dim() == 4


def test_diffusion_training_loss_is_finite():
    model = ConditionalDiffusionUNet(out_channels=3, cond_channels=2, base_channels=16,
                                      channel_mults=(1, 2), num_res_blocks=1)
    diffusion = GaussianDiffusion(timesteps=50)
    x0 = torch.randn(2, 3, 32, 32)
    sar = torch.randn(2, 2, 32, 32)
    t = torch.randint(0, 50, (2,))
    loss = diffusion.training_loss(model, x0, sar, t)
    assert torch.isfinite(loss)


def test_segmentation_shapes():
    model = SimpleSegmentationNet(in_channels=3, num_classes=5, base_channels=8)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 5, 64, 64)
