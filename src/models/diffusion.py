"""
Lightweight conditional diffusion model for SAR -> EO translation
(Stage 11: "Main model — conditional diffusion").

This is a simplified DDPM (Denoising Diffusion Probabilistic Model), NOT a
Stable-Diffusion-scale latent diffusion model — deliberately so, per the
brief's constraint of running on a single Kaggle GPU.

Conceptual pipeline:

    EO_real --(forward diffusion: add noise over T steps)--> pure noise
    SAR --(conditioning)--> concatenated as extra input channels
    Noisy_EO_t, SAR, t --(denoising U-Net)--> predicted noise
    Training objective: MSE(predicted_noise, actual_noise)

At sampling time we reverse the process: start from random noise and
iteratively denoise, conditioned on the SAR image at every step, to
produce a generated EO image.

Conditioning strategy used here: SIMPLE CHANNEL-CONCATENATION. The SAR
image is concatenated to the noisy EO image at every timestep before
feeding the denoising U-Net (this mirrors how Pix2Pix conditions its
generator, extended into the diffusion setting). This is the "beginner
friendly" conditioning approach mentioned in the brief; cross-attention
conditioning is listed as future work.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


# ---- Timestep embedding ---------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding for the diffusion
    timestep t, so the network knows "how noisy" the input currently is."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ---- Residual block with timestep conditioning ----------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


# ---- Denoising U-Net (the epsilon-predictor network) -----------------------

class ConditionalDiffusionUNet(nn.Module):
    """Predicts the noise added to a (SAR-conditioned) noisy EO patch.

    Input channel layout: concat(noisy_EO [out_channels], SAR [cond_channels])
    Output: predicted noise, same shape as the EO target (out_channels).
    """

    def __init__(self, out_channels=3, cond_channels=2, base_channels=64,
                 channel_mults=(1, 2, 2, 4), num_res_blocks=2, time_emb_dim=256):
        super().__init__()
        in_channels = out_channels + cond_channels

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        chs = [ch]
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_emb_dim))
                ch = out_ch
                chs.append(ch)
            if i != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                chs.append(ch)
            else:
                self.downsamples.append(None)

        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResBlock(ch + chs.pop(), out_ch, time_emb_dim))
                ch = out_ch
            if i != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(None)

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, noisy_eo, sar_cond, t):
        x = torch.cat([noisy_eo, sar_cond], dim=1)
        t_emb = self.time_embed(t)
        h = self.in_conv(x)

        hs = [h]
        idx = 0
        for i in range(len(self.downsamples)):
            for _ in range(len(self.down_blocks) // len(self.downsamples)):
                h = self.down_blocks[idx](h, t_emb)
                hs.append(h)
                idx += 1
            if self.downsamples[i] is not None:
                h = self.downsamples[i](h)
                hs.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        idx = 0
        up_i = 0
        for i in range(len(self.upsamples)):
            n_blocks = len(self.up_blocks) // len(self.upsamples)
            for _ in range(n_blocks):
                skip = hs.pop()
                if h.shape[-2:] != skip.shape[-2:]:
                    h = nn.functional.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[idx](h, t_emb)
                idx += 1
            if self.upsamples[i] is not None:
                h = self.upsamples[i](h)

        h = self.act(self.out_norm(h))
        return self.out_conv(h)


# ---- Gaussian diffusion process (forward + reverse) ------------------------

class GaussianDiffusion:
    """Implements the forward noising process and the reverse sampling loop
    for a standard DDPM with a linear beta schedule.

    This class is intentionally separate from the network (ConditionalDiffusionUNet)
    so the same network could later be swapped for a different schedule/sampler
    (e.g. DDIM) without changing the model code.
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        """Forward process: sample x_t from x_0 in closed form."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_1m_ac * noise, noise

    def training_loss(self, model, x0, sar_cond, t):
        """Standard DDPM epsilon-prediction MSE loss (Ho et al., 2020)."""
        x_t, noise = self.q_sample(x0, t)
        predicted_noise = model(x_t, sar_cond, t)
        return nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample_loop(self, model, sar_cond, shape, device):
        """Reverse process: iteratively denoise from pure noise, conditioned
        on the SAR image, to produce a generated EO image. Runs the full
        `timesteps` steps — for faster sampling, implement DDIM (future work,
        see docs/research_notes.md)."""
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            predicted_noise = model(x, sar_cond, t)

            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConditionalDiffusionUNet(out_channels=3, cond_channels=2, base_channels=32,
                                      channel_mults=(1, 2), num_res_blocks=1).to(device)
    diffusion = GaussianDiffusion(timesteps=100, device=device)

    x0 = torch.randn(2, 3, 64, 64, device=device)
    sar = torch.randn(2, 2, 64, 64, device=device)
    t = torch.randint(0, 100, (2,), device=device)

    loss = diffusion.training_loss(model, x0, sar, t)
    print("Training loss (sanity check):", loss.item())
    assert torch.isfinite(loss)
    print("OK")
