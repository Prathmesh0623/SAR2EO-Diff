# SAR2EO-Diff

### Conditional Diffusion for SAR-to-EO Satellite Image Translation with Semantic Consistency

![status](https://img.shields.io/badge/status-in--progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.x-red)
![license](https://img.shields.io/badge/license-MIT-green)

> **Note on project status:** this repository is a complete, runnable
> scaffold — every module has passing unit tests and shape-verified
> forward passes — but it has **not yet been trained end-to-end on the
> full SEN12MS dataset**. All quantitative results in this README and in
> `docs/research_report.md` are placeholders until real Kaggle GPU
> experiments are run and logged in `docs/experiments.md`. See "Project
> Status" below.

## Overview

SAR2EO-Diff investigates whether a deep generative model can translate a
Sentinel-1 SAR satellite image into a corresponding optical/EO-like image
while preserving meaningful geographic structure. The project compares
four increasingly sophisticated approaches — U-Net regression, Pix2Pix
(conditional GAN), a lightweight conditional diffusion model, and
diffusion augmented with a segmentation-based semantic-consistency loss —
under a shared, reproducible evaluation protocol.

## Research Problem

Can a generative model produce EO-like imagery from SAR that is (a)
visually/perceptually realistic **and** (b) semantically faithful (i.e.
downstream land-cover interpretation of the generated image matches the
real EO image)? These two properties don't automatically align — a model
can look convincing while misrepresenting the underlying land cover, which
is the central research question motivating the semantic-consistency
component.

## Key Contributions (research-oriented, not novel-method claims)

- A from-scratch, from-first-principles lightweight conditional DDPM for
  SAR-to-EO translation, sized to run on a single Kaggle GPU.
- A controlled comparison against two standard baselines (U-Net, Pix2Pix)
  under identical data/eval protocol.
- An **experimental semantic-consistency enhancement**: a segmentation-based
  auxiliary loss during diffusion training, evaluated for whether it
  improves downstream semantic usefulness (not just pixel/perceptual
  metrics).
- Explicit ablations (polarization choice, model architecture, semantic
  loss on/off) and a mandatory failure-analysis section — see
  `docs/research_report.md`.

## Architecture

```text
             SAR (VV, VH)
                  │
                  ▼
            SAR conditioning
                  │
                  ▼
Noise ──►  Diffusion U-Net  ──►  Generated EO ──► Segmentation Net ──► Semantic loss
                                       │
                                       ▼
                              Compared against real EO
                          (PSNR / SSIM / LPIPS / FID / mIoU)
```

Baselines (U-Net, Pix2Pix) follow the same SAR-in / EO-out contract so all
four models are directly comparable. Full architecture details in
`docs/methodology.md`.

## Dataset

[SEN12MS](https://mediatum.ub.tum.de/1474000) — paired Sentinel-1 SAR,
Sentinel-2 EO, and MODIS land-cover patches. First implementation uses
2-channel SAR input (VV + VH) and 3-channel RGB EO output; extending to
the full 13-band Sentinel-2 output is listed as future work.

## Methodology

See [`docs/methodology.md`](docs/methodology.md) for the full writeup of
each model, the training strategy, and the evaluation protocol.

## Experiments

All experiments are config-driven (`configs/*.yaml`) and logged in
[`docs/experiments.md`](docs/experiments.md). Training follows a staged
strategy to avoid wasting limited Kaggle GPU hours: pipeline smoke test on
a tiny subset → overfit-a-tiny-batch sanity check → larger-subset
generalization check → final full run.

## Results

`[TODO — final comparison table, populated from
results/tables/final_comparison.csv once real experiments are complete]`

| Model | PSNR | SSIM | LPIPS | FID | mIoU |
|-------|------|------|-------|-----|------|
| U-Net | — | — | — | — | — |
| Pix2Pix | — | — | — | — | — |
| Conditional Diffusion | — | — | — | — | — |
| Diffusion + Semantic Consistency | — | — | — | — | — |

## Ablation Study

- **Polarization**: VV only vs. VH only vs. VV+VH
- **Architecture**: U-Net vs. Pix2Pix vs. Diffusion vs. Diffusion+Semantic
- **Semantic loss**: diffusion with vs. without the semantic-consistency term

Full tables in `docs/experiments.md` and `docs/research_report.md` §15.

## Failure Analysis

A dedicated failure-analysis pass (blurry regions, incorrect colors,
missing/false structures, vegetation/water confusion, urban reconstruction
errors, SAR ambiguity) is required before the project is considered
complete — see `docs/research_report.md` §16 and `notebooks/08_final_results.ipynb`.

## Project Structure

```text
SAR2EO-Diff/
├── configs/            # YAML configs for each model (no hard-coded hyperparameters)
├── src/
│   ├── data/            # dataset, preprocessing, paired augmentations
│   ├── models/           # unet, pix2pix, diffusion, segmentation
│   ├── training/         # per-model training loops
│   ├── evaluation/        # metrics, evaluation loop, visualization
│   └── utils/             # seed, checkpoint, experiment logger
├── notebooks/            # 01-08, Kaggle-run, import from src/
├── scripts/              # thin CLI entry points (prepare_data / train / evaluate)
├── tests/                # pytest unit tests (data, models, metrics)
├── results/               # figures / tables / qualitative outputs (gitignored binaries)
├── checkpoints/            # model weights (gitignored)
└── docs/                    # methodology, experiment log, research notes, final report
```

## Installation

```bash
git clone <your-repo-url>
cd SAR2EO-Diff
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the unit tests (no dataset required) to confirm the environment is set up correctly:

```bash
pytest tests/ -v
```

## Kaggle Setup

1. Create a Kaggle Notebook and attach the SEN12MS dataset as an input.
2. Upload/clone this repository into `/kaggle/working/SAR2EO-Diff`.
3. `!pip install -r requirements.txt`
4. Update `data.dataset_root` in the relevant `configs/*.yaml` to match the
   Kaggle input path (e.g. `/kaggle/input/sen12ms`).
5. Run `scripts/prepare_data.py` first to confirm the dataset pipeline
   discovers pairs correctly before launching any training.

## Training

```bash
python scripts/train.py --config configs/baseline_unet.yaml
python scripts/train.py --config configs/pix2pix.yaml
python scripts/train.py --config configs/diffusion.yaml
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/baseline_unet.yaml \
    --checkpoint checkpoints/unet/best.pt \
    --output results/tables/unet_test_metrics.json
```

## Example Results

`[TODO — qualitative comparison grids, populated in
notebooks/08_final_results.ipynb / results/qualitative/ once experiments
are complete]`

## Limitations

Single-GPU/Kaggle-scale training, patch-level (not full-scene) modeling,
RGB-only EO output in the first implementation, and the inherent imperfect
correspondence between SAR and optical imagery. Full discussion in
`docs/research_report.md` §18.

## Future Work

Full multispectral Sentinel-2 output, latent diffusion for higher
effective resolution, cross-attention conditioning, temporal SAR stacks,
and downstream-task evaluation. Full list in `docs/research_report.md` §19.

## Project Status

- [x] Repository structure, configs, and all source modules created
- [x] Unit tests passing for data preprocessing, all four model
      architectures (shape/forward-pass), and evaluation metrics
- [ ] Dataset pipeline validated against real SEN12MS on Kaggle
- [ ] U-Net baseline trained
- [ ] Pix2Pix baseline trained
- [ ] Conditional diffusion trained
- [ ] Semantic consistency experiment run
- [ ] Full evaluation + ablations + failure analysis + final report completed

## References

`[TODO — populate from docs/research_notes.md literature review]`

## Author

Prathmesh
