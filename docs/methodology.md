# Methodology

## Problem formulation

Given a Sentinel-1 SAR image patch `x_SAR` (VV and/or VH polarization,
2 channels), learn a mapping `f: x_SAR -> x_EO` that produces an image
resembling the corresponding Sentinel-2 optical (EO) patch `x_EO` (3-channel
RGB in the first implementation), while preserving the geographic/semantic
content of the scene.

## Why this problem is hard

- SAR and EO capture fundamentally different physical phenomena (radar
  backscatter vs. reflected sunlight) — there is no exact deterministic
  mapping, only a statistical/learned correlation.
- SAR is affected by speckle noise, layover, and shadow effects that have
  no analog in optical imagery.
- A generative model can produce a *visually plausible* EO image that is
  nevertheless *semantically wrong* (e.g. hallucinating a road where there
  is a river) — hence the semantic-consistency component (Stage 12).

## Models compared

1. **U-Net (baseline)** — deterministic regression, L1 pixel loss only.
   Establishes a floor: what's achievable with a purely reconstructive
   objective and no notion of realism.
2. **Pix2Pix** — U-Net-style generator + PatchGAN discriminator,
   adversarial + L1 loss. Tests whether adversarial training improves
   perceptual realism over pure regression.
3. **Conditional Diffusion** — DDPM-style denoising model conditioned on
   SAR via channel concatenation. Tests whether the iterative
   noise-to-image generative process produces better fidelity/diversity
   than a single-shot GAN generator.
4. **Diffusion + Semantic Consistency** — adds a segmentation-based
   consistency loss during (or as a fine-tuning stage after) diffusion
   training, penalizing generations whose predicted land-cover map
   diverges from the real EO's land-cover map.

## Evaluation protocol

All models are evaluated on the same held-out **test** split (never seen
during training or hyperparameter selection). Metrics:

- **PSNR, SSIM** — pixel/structural fidelity vs. ground-truth EO.
- **LPIPS** — perceptual similarity (deep-feature space), more aligned
  with human judgment of "does this look right" than PSNR/SSIM.
- **FID** — distributional realism of generated images as a set (not
  per-pair), computed once a large enough sample of generations exists.
- **mIoU / F1 / pixel accuracy** — semantic usefulness, computed by
  running the segmentation network (Stage 12) over generated vs. real EO
  and comparing to land-cover labels.

## Reproducibility

- Fixed random seed (`seed: 42` in every config) for data split, model
  init, and augmentation.
- Config-driven experiments (`configs/*.yaml`) — no hard-coded
  hyperparameters in training scripts.
- Every experiment logged to Weights & Biases (or console fallback) with
  full config + environment info (`src/utils/seed.py::log_environment`).
- Checkpointing supports resume, so a Kaggle session timeout doesn't
  invalidate a run.
