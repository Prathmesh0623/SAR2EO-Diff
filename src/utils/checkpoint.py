"""Checkpointing utilities (Stage 28): save/resume training so a Kaggle
session timing out doesn't cost you the whole run."""
from __future__ import annotations
import os
import torch


def save_checkpoint(path: str, model, optimizer=None, epoch: int = 0,
                     best_metric: float | None = None, extra: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state  # caller can read state["epoch"], state["best_metric"], etc.
