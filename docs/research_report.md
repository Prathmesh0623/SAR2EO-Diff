# SAR2EO-Diff: Conditional Diffusion for SAR-to-EO Satellite Image Translation with Semantic Consistency

**Status: TEMPLATE — sections marked `[TODO]` must be completed after running
real experiments on Kaggle. Do not fill in numbers, claims, or conclusions
that are not backed by an actual completed run recorded in `docs/experiments.md`.**

## 1. Abstract
`[TODO — 150-200 words summarizing problem, method, key findings once available]`

## 2. Introduction
`[TODO]`

## 3. Problem Statement
Translate Sentinel-1 SAR imagery into Sentinel-2-like optical (EO) imagery
while preserving geographically/semantically meaningful structure.

## 4. Motivation
SAR is available day/night and through cloud cover, but is harder to
visually interpret than optical imagery. A reliable SAR-to-EO translation
model could support rapid situational awareness (e.g. disaster response)
when optical imagery is unavailable due to clouds or nighttime.

## 5. Related Work
`[TODO — populate from docs/research_notes.md literature review]`

## 6. Dataset
SEN12MS: paired Sentinel-1 SAR / Sentinel-2 EO / MODIS land-cover patches.
`[TODO — dataset statistics actually used: number of scenes, patches, splits]`

## 7. Data Preprocessing
SAR: converted to dB (if needed), clipped to `[-25, 5]` dB, normalized to
`[-1, 1]`. EO: clipped to `[0, 4000]` reflectance, normalized to `[-1, 1]`.
`[TODO — note any changes made after inspecting real value distributions
in notebook 01]`

## 8. Methodology
See `docs/methodology.md` for the full description of all four models and
the evaluation protocol.

## 9. Baseline Models
U-Net (pixel regression) and Pix2Pix (conditional GAN).
`[TODO — final baseline results]`

## 10. Proposed Conditional Diffusion Model
Lightweight DDPM conditioned on SAR via channel concatenation.
`[TODO — final diffusion results]`

## 11. Semantic Consistency
`[TODO — segmentation network setup, semantic loss results]`

## 12. Experimental Setup
`[TODO — hardware, hyperparameters actually used, training time per model]`

## 13. Evaluation Metrics
PSNR, SSIM, LPIPS, FID (image quality); mIoU, F1, pixel accuracy (semantic
usefulness). See `docs/methodology.md` for definitions.

## 14. Results
`[TODO — final comparison table from results/tables/final_comparison.csv]`

## 15. Ablation Studies
`[TODO — polarization, model, and semantic-loss ablations from docs/experiments.md]`

## 16. Failure Analysis
`[TODO — from notebook 08]`

## 17. Discussion
`[TODO]`

## 18. Limitations
- Single-GPU (Kaggle), no ability to train at full Stable-Diffusion scale.
- Patch-level training (128x128 or smaller), not full-scene.
- RGB-only EO output in the first implementation (not full 13-band
  Sentinel-2 multispectral).
- SAR-to-optical correspondence is fundamentally imperfect (different
  physical sensing modalities).
- `[TODO — add any limitations discovered during actual experimentation]`

## 19. Future Work
- Full multispectral Sentinel-2 generation (all 13 bands).
- Latent diffusion (encode to a smaller latent space before diffusing) for
  larger effective resolution within the same compute budget.
- Cross-attention-based conditioning instead of channel concatenation.
- Larger-scale / higher-resolution imagery.
- Temporal information (multi-date SAR stacks).
- Downstream task evaluation (e.g. does better SAR-to-EO translation help
  a downstream object detector trained on the generated images?).

## 20. Conclusion
`[TODO]`

## 21. References
`[TODO — populate from docs/research_notes.md]`
