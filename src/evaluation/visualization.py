"""
Qualitative visualization helpers (Stages 5, 16, 17, 33).

Produces the side-by-side comparison grids the brief asks for:
    SAR | Real EO | Model output(s) | Error map
so results are easy to eyeball during development and to drop into the
final report / README as figures.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.data.preprocessing import denormalize_eo, denormalize_sar


def _percentile_stretch(img: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """Percentile-stretch an image to [0,1] for display, then apply gamma
    correction to brighten dim scenes (a genuinely dark real patch --
    e.g. water, shadow, dense forest -- can look almost black even after
    a linear stretch; gamma < 1 lifts midtones for visibility without
    lying about the data). Guards against near-zero-variance input (e.g.
    an undertrained model outputting a nearly flat/constant image) --
    without this guard, dividing by a near-zero (hi - lo) range collapses
    everything to black."""
    lo, hi = np.percentile(img, [1, 99])
    if hi - lo < 1e-3:
        stretched = np.clip(img, 0, 1)
    else:
        stretched = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return np.power(stretched, gamma)


def _sar_to_display(sar_channel: np.ndarray) -> np.ndarray:
    """Take one normalized SAR channel (e.g. VV) and rescale to [0,1] for
    grayscale display."""
    db = denormalize_sar(sar_channel)
    return _percentile_stretch(db)


def plot_sar_eo_comparison(sar: torch.Tensor, eo_real: torch.Tensor,
                            model_outputs: dict[str, torch.Tensor],
                            save_path: str | None = None):
    """
    Args:
        sar: (C_sar, H, W) normalized SAR patch (expects at least VV in channel 0)
        eo_real: (3, H, W) normalized real EO patch
        model_outputs: dict of {model_name: (3, H, W) tensor} generated EO images
        save_path: if given, saves the figure instead of / in addition to showing it
    """
    sar_np = sar.detach().cpu().numpy()
    eo_real_np = _percentile_stretch(denormalize_eo(eo_real.detach().cpu().numpy()))

    n_cols = 2 + len(model_outputs) + 1  # SAR + real EO + each model + error map (for best model)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(_sar_to_display(sar_np[0]), cmap="gray")
    axes[0].set_title("SAR (VV)")
    axes[0].axis("off")

    axes[1].imshow(np.transpose(eo_real_np, (1, 2, 0)))
    axes[1].set_title("Real EO")
    axes[1].axis("off")

    last_pred = None
    for i, (name, pred) in enumerate(model_outputs.items()):
        pred_np = _percentile_stretch(denormalize_eo(pred.detach().cpu().numpy()))
        axes[2 + i].imshow(np.transpose(pred_np, (1, 2, 0)))
        axes[2 + i].set_title(name)
        axes[2 + i].axis("off")
        last_pred = pred_np

    if last_pred is not None:
        error_map = np.abs(last_pred - eo_real_np).mean(axis=0)
        im = axes[-1].imshow(error_map, cmap="inferno", vmin=0, vmax=1)
        axes[-1].set_title(f"|Error| ({list(model_outputs.keys())[-1]})")
        axes[-1].axis("off")
        fig.colorbar(im, ax=axes[-1], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()