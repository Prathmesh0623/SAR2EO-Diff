"""Reproducibility utilities (Stage 20)."""
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set all relevant random seeds for reproducible experiments.

    Note: full bitwise reproducibility on GPU also requires setting
    torch.backends.cudnn.deterministic = True, which can slow training.
    We leave that off by default (benchmark mode) for speed on Kaggle GPU;
    turn it on only when debugging a specific reproducibility issue.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_environment():
    """Return a dict of environment info to log alongside every experiment
    (Stage 20 requirement: document seed, Python, PyTorch, CUDA versions)."""
    import sys
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info
