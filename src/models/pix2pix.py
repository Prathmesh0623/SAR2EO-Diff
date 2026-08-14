"""
Pix2Pix baseline: conditional GAN for SAR -> EO translation (Stage 10).

Generator: reuses a U-Net-style encoder-decoder (same shape contract as
src/models/unet.py) so comparisons are apples-to-apples.

Discriminator: PatchGAN — classifies overlapping NxN patches of the image
as real/fake rather than the whole image at once. This is the key idea
from the original Pix2Pix paper (Isola et al., 2017): it encourages
high-frequency detail/texture realism while the L1 loss handles low-frequency
(overall structure) correctness.

Losses (combined in src/training/train_pix2pix.py):
    L_total = L_adversarial(G, D) + lambda_l1 * L1(G(sar), eo_real)
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .unet import UNet  # reuse as the generator backbone


class Pix2PixGenerator(UNet):
    """Identical architecture to the U-Net baseline. Kept as a distinct
    class name for clarity in configs/checkpoints, and to make it easy to
    diverge (e.g. add dropout for noise injection) without touching the
    baseline U-Net."""
    pass


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (standard Pix2Pix configuration).

    Input: concatenation of (condition, image) along the channel dim, i.e.
    concat(SAR, EO) — the discriminator judges whether the EO image is a
    *plausible pairing* for the given SAR input, not just realistic in
    isolation.
    """

    def __init__(self, in_channels=2 + 3, base_channels=64):
        super().__init__()

        def block(in_ch, out_ch, stride=2, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, base_channels, normalize=False),
            *block(base_channels, base_channels * 2),
            *block(base_channels * 2, base_channels * 4),
            *block(base_channels * 4, base_channels * 8, stride=1),
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, condition, image):
        x = torch.cat([condition, image], dim=1)
        return self.model(x)  # raw logits, shape (B, 1, H', W') — patch predictions


if __name__ == "__main__":
    g = Pix2PixGenerator(in_channels=2, out_channels=3)
    d = PatchGANDiscriminator(in_channels=2 + 3)

    sar = torch.randn(2, 2, 128, 128)
    eo_fake = g(sar)
    print("Generator output:", eo_fake.shape)

    patch_pred = d(sar, eo_fake)
    print("Discriminator output:", patch_pred.shape)
