"""Thin wrapper around Weights & Biases so training scripts don't hard-code
wandb calls everywhere (Stage 19: experiment tracking). Falls back to
plain print() if wandb isn't installed/configured or use_wandb=False, so
scripts still run for quick local smoke tests."""
from __future__ import annotations


class ExperimentLogger:
    def __init__(self, use_wandb: bool, project: str = "sar2eo-diff",
                 run_name: str | None = None, config: dict | None = None):
        self.use_wandb = use_wandb
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project=project, name=run_name, config=config or {})
                self._wandb = wandb
            except Exception as e:
                print(f"[ExperimentLogger] wandb init failed ({e}); "
                      f"falling back to console logging.")
                self.use_wandb = False

    def log(self, metrics: dict, step: int | None = None):
        if self.use_wandb and self._wandb is not None:
            self._wandb.log(metrics, step=step)
        else:
            step_str = f"step={step} " if step is not None else ""
            print(f"[log] {step_str}{metrics}")

    def log_images(self, key: str, images, step: int | None = None):
        if self.use_wandb and self._wandb is not None:
            self._wandb.log({key: [self._wandb.Image(img) for img in images]}, step=step)
        # no-op fallback for console mode; save images to disk separately if needed

    def finish(self):
        if self.use_wandb and self._wandb is not None:
            self._wandb.finish()
