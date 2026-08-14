"""
Baseline U-Net for direct SAR -> EO regression (Stage 9 in the roadmap:
"First model — U-Net").

This is a standard encoder-decoder with skip connections. No adversarial
or diffusion component — trained with a simple pixel-wise loss (L1). Its
purpose is to (a) validate the whole data pipeline end-to-end and (b) give
a baseline number that Pix2Pix and diffusion must beat to justify their
extra complexity.
"""

from __future__ import annotations
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, base_channels=64, depth=4):
        super().__init__()
        self.depth = depth

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        out_ch = base_channels
        for i in range(depth):
            self.enc_blocks.append(conv_block(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch
            out_ch *= 2

        # Bottleneck
        self.bottleneck = conv_block(ch, out_ch)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(out_ch, ch, 2, stride=2))
            self.dec_blocks.append(conv_block(out_ch, ch))
            out_ch = ch
            ch //= 2

        self.final_conv = nn.Conv2d(out_ch, out_channels, 1)
        self.out_activation = nn.Tanh()  # output normalized to [-1, 1], matches data prep

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):
            x = upconv(x)
            # Handle any off-by-one size mismatch from odd input dims
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.out_activation(self.final_conv(x))


if __name__ == "__main__":
    # Quick shape sanity check — run with `python -m src.models.unet`
    model = UNet(in_channels=2, out_channels=3)
    dummy = torch.randn(2, 2, 128, 128)
    out = model(dummy)
    print("Input:", dummy.shape, "Output:", out.shape)
    assert out.shape == (2, 3, 128, 128)
    print("OK")
