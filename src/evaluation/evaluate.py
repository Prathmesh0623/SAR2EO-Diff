"""
Evaluation loop (Stage 15) — runs a trained model over a dataset split and
computes the full metric suite. Used by scripts/evaluate.py and by
notebooks/07_evaluation.ipynb on Kaggle.

Writes a CSV/JSON to results/tables/ with per-sample and aggregate metrics.
No numbers here are ever hard-coded — everything is computed from the
model's actual predictions on the given DataLoader.
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_psnr, compute_ssim, LPIPSMetric


@torch.no_grad()
def evaluate_model(model, dataloader: DataLoader, device: str,
                    compute_lpips: bool = True, generate_fn=None,
                    output_path: str | None = None) -> dict:
    """
    Args:
        model: trained model (U-Net, Pix2Pix generator, or diffusion model)
        dataloader: validation/test DataLoader yielding {"sar":, "eo":}
        generate_fn: optional callable(model, sar_batch) -> eo_pred_batch.
            Needed for diffusion models (which require a sampling loop
            rather than a single forward pass). Defaults to model(sar).
        output_path: if given, dump per-sample + aggregate results as JSON
    """
    model.eval()
    generate_fn = generate_fn or (lambda m, sar: m(sar))

    lpips_metric = LPIPSMetric(device=device) if compute_lpips else None

    psnr_scores, ssim_scores, lpips_scores = [], [], []
    per_sample_records = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        sar = batch["sar"].to(device)
        eo_real = batch["eo"].to(device)
        eo_pred = generate_fn(model, sar)

        for i in range(sar.shape[0]):
            psnr = compute_psnr(eo_pred[i], eo_real[i])
            ssim = compute_ssim(eo_pred[i], eo_real[i])
            record = {"psnr": psnr, "ssim": ssim}

            if lpips_metric is not None:
                lp = lpips_metric(eo_pred[i], eo_real[i])
                record["lpips"] = lp
                lpips_scores.append(lp)

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            per_sample_records.append(record)

    results = {
        "num_samples": len(psnr_scores),
        "psnr_mean": float(sum(psnr_scores) / len(psnr_scores)) if psnr_scores else None,
        "ssim_mean": float(sum(ssim_scores) / len(ssim_scores)) if ssim_scores else None,
        "lpips_mean": float(sum(lpips_scores) / len(lpips_scores)) if lpips_scores else None,
        "per_sample": per_sample_records,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
