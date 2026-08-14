# Experiment Log

Record every training run here as you actually execute it on Kaggle —
this file is the source of truth for the numbers that eventually go into
`docs/research_report.md`, the README, and the resume bullets. **Do not
add a row until the run has actually completed and you've read the
resulting metrics** (see the NO FABRICATION rule in the original project
brief).

## Log format

| Date | Model | Config | Dataset subset | Epochs | Batch size | GPU | Train time | Final train loss | Final val loss | Notes |
|------|-------|--------|-----------------|--------|------------|-----|-------------|-------------------|------------------|-------|
| _(fill in)_ | | | | | | | | | | |

## Ablation runs

### Polarization ablation (VV only / VH only / VV+VH)

| Run | Model | Polarization | PSNR | SSIM | LPIPS |
|-----|-------|---------------|------|------|-------|
| _(fill in)_ | | | | | |

### Model ablation (U-Net / Pix2Pix / Diffusion / Diffusion+Semantic)

| Model | PSNR | SSIM | LPIPS | FID | mIoU |
|-------|------|------|-------|-----|------|
| _(fill in)_ | | | | | |

### Semantic-loss ablation (diffusion with vs. without semantic consistency)

| Variant | PSNR | SSIM | mIoU | F1 | Pixel Acc |
|---------|------|------|------|----|----|
| _(fill in)_ | | | | | |

## Failure cases identified

| Scene type | Failure mode | Likely cause | Possible fix |
|------------|---------------|---------------|----------------|
| _(fill in)_ | | | |
