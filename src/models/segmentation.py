"""
Simple segmentation network for semantic consistency (Stage 12).

Purpose: predict a land-cover class map from an EO image (real or
generated). During training of the diffusion model with semantic
consistency enabled, we run the GENERATED EO image through this frozen,
pretrained segmentation network and compare its predicted land-cover map
to the land-cover map predicted for the REAL EO image (or to MODIS labels
if available and aligned). This penalizes generations that look plausible
but destroy semantic structure (e.g. turning a field into water).

Deliberately small (a shallow U-Net-style encoder-decoder) — per the
brief, this must NOT be an "extremely complicated segmentation
architecture." Pretrain this separately on real EO + land-cover label
pairs (script: src/training/train_segmentation.py — add if you pursue
Stage 12 in full) before using it as a frozen semantic loss network.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class SimpleSegmentationNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, base_channels=32):
        super().__init__()

        def conv_bn_relu(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_bn_relu(in_channels, base_channels)
        self.enc2 = conv_bn_relu(base_channels, base_channels * 2)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_bn_relu(base_channels * 2, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = conv_bn_relu(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = conv_bn_relu(base_channels * 2, base_channels)

        self.classifier = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.classifier(d1)  # logits, shape (B, num_classes, H, W)


def semantic_consistency_loss(seg_net: nn.Module, generated_eo: torch.Tensor,
                               real_eo: torch.Tensor) -> torch.Tensor:
    """KL-divergence between the segmentation network's softmax predictions
    on the generated EO vs. the real EO. seg_net should be frozen (eval mode,
    requires_grad=False on its parameters) when used inside diffusion training.
    """
    with torch.no_grad():
        real_logits = seg_net(real_eo)
        real_probs = torch.softmax(real_logits, dim=1)

    gen_logits = seg_net(generated_eo)
    gen_log_probs = torch.log_softmax(gen_logits, dim=1)

    return nn.functional.kl_div(gen_log_probs, real_probs, reduction="batchmean")


if __name__ == "__main__":
    model = SimpleSegmentationNet(in_channels=3, num_classes=10)
    dummy = torch.randn(2, 3, 128, 128)
    out = model(dummy)
    print("Segmentation output:", out.shape)
    assert out.shape == (2, 10, 128, 128)
    print("OK")
