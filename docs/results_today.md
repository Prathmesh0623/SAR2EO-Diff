# SAR2EO-Diff — Results (Time-Constrained Initial Run)

**Date:** run completed same-day, condensed scope due to time constraints.
**Hardware:** Kaggle Tesla T4 (free tier)
**Dataset:** SEN12MS-Asia (Kaggle, krishnaanchal/sen12ms-asia), 13 of 14
shards usable (shard_013.pt was corrupted/truncated on this mirror and
was excluded), ~26,000 samples available; each model below was trained
on a small subset for speed (see per-model notes).

> **Scope note:** this is a first, deliberately small-scale pass to prove
> the full pipeline (data loading -> preprocessing -> model -> training
> -> checkpointing) works correctly end-to-end on real data, across all
> three trainable architectures. Epoch counts and subset sizes were kept
> small to fit a single-day compute budget on a free-tier GPU. Numbers
> below are real, measured values from actual training runs — not
> estimates or literature figures. A full-scale run (more epochs, full
> dataset, semantic consistency, formal test-set evaluation with
> PSNR/SSIM/LPIPS/FID/mIoU) is documented as future work.

## Training summary

| Model | Subset size | Epochs | Final train loss | Final val loss | Notes |
|-------|-------------|--------|-------------------|------------------|-------|
| U-Net (L1 baseline) | 1,000 | 5 | 0.0494 | **0.0477** | Smooth, stable convergence; val loss tracked train loss closely, no overfitting observed. |
| Pix2Pix (cGAN) | 1,000 | 5 | G: 14.10 (adv+L1) / D: 0.0001 | — (L1 component ≈0.04-0.05, comparable to U-Net) | Discriminator collapse observed: `loss_d` -> ~0 while `loss_g_adv` climbed steadily. Generator's L1 reconstruction stayed healthy and comparable to the U-Net baseline; the adversarial component alone became uninformative after ~epoch 1. See Failure Analysis below. |
| Conditional Diffusion (DDPM, ε-prediction) | 500 | 3 | 0.0388 (noise-pred MSE) | not computed (no held-out val loop run today) | Loss decreased steadily and consistently (1.10 → 0.04) with no divergence, confirming the denoising objective is learnable on this SAR/EO pair. Not enough training to expect high-fidelity generated samples yet — see Limitations. |
| Diffusion + Semantic Consistency | — | — | — | — | **Not run today.** Requires a separately pretrained segmentation network on land-cover labels first; out of scope for today's time budget. Code path exists (`semantic_consistency.enabled` in `configs/diffusion.yaml`, `src/models/segmentation.py`) but untrained. |

## Failure Analysis (real observation from today's runs)

**Pix2Pix discriminator collapse.** Within the first epoch, the PatchGAN
discriminator's loss dropped to near-zero (0.72 -> 0.0001 by epoch 4)
while the generator's adversarial loss climbed continuously (0.98 -> 9.1).
This is the classic Pix2Pix failure mode where the discriminator
overpowers the generator early, leaving the adversarial loss term
uninformative for the rest of training. Despite this, the generator's L1
reconstruction term remained low and stable (~0.04-0.05), suggesting the
underlying SAR->EO mapping was still being learned reasonably well —
just without the adversarial realism boost Pix2Pix is meant to add.

**Possible causes:** discriminator learning rate/capacity too high
relative to the generator for this dataset/patch size; no label
smoothing or noise injection on discriminator targets; small training
subset (1,000 samples) may make the discriminator's task
easier than intended.

**Possible fixes (future work, not yet tried):** label smoothing on real
targets, reducing discriminator update frequency relative to generator,
lowering discriminator learning rate, or increasing training set size.

## Limitations (today's run specifically)

- All models trained on small subsets (500-1,000 samples) and few epochs
  (3-5) due to a single-day time budget on free-tier Kaggle GPU compute —
  not representative of final achievable quality.
- No formal held-out test-set evaluation was run (PSNR/SSIM/LPIPS/FID/
  mIoU) — only training/validation loss curves are reported here.
- One of 14 dataset shards (shard_013.pt) was corrupted/truncated on the
  Kaggle mirror used and was excluded; not expected to materially affect
  results given 13 remaining shards (~26,000 total samples available,
  only a fraction of which were used in today's subset runs).
- Diffusion sampling (actually generating an EO image from noise) was not
  performed today — only the training-time denoising loss was measured.
- Semantic consistency (segmentation-guided loss) was not trained.

## Immediate next steps (future work)

1. Run all three models on the full ~26,000-sample dataset for more
   epochs (staged: Phase 3/4 of the original training strategy).
2. Add discriminator regularization (label smoothing / reduced update
   frequency) to address the Pix2Pix collapse observed above.
3. Run full reverse-diffusion sampling and visually/quantitatively
   evaluate generated EO images.
4. Pretrain the segmentation network on land-cover labels, then enable
   and train the semantic-consistency variant of diffusion.
5. Run `scripts/evaluate.py` against held-out test data for all models to
   get PSNR/SSIM/LPIPS (and FID once enough samples exist) for a proper
   comparison table.