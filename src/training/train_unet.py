"""
Training script for the U-Net baseline (Stage 9).

Run from the project root (locally for a smoke test, or on Kaggle for a
real run):

    python -m src.training.train_unet --config configs/baseline_unet.yaml

Follows the staged strategy from Stage 27 of the brief:
    Phase 1: small subset_size, few epochs -> confirm the pipeline runs
    Phase 2: overfit a tiny batch -> confirm the model CAN learn the mapping
    Phase 3: larger subset, more epochs -> check generalization
    Phase 4: full run -> final baseline number for comparison table
"""
from __future__ import annotations
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.sharded_dataset import ShardedSEN12MSDataset
from src.data.transforms import PairedAugment
from src.models.unet import UNet
from src.utils.seed import set_seed, log_environment
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logger import ExperimentLogger


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Environment:", log_environment())

    train_ds = ShardedSEN12MSDataset(
        root=cfg["data"]["dataset_root"], split="train",
        seed=cfg["data"]["seed"],
        train_frac=cfg["data"]["train_split"],
        val_frac=cfg["data"]["val_split"],
        subset_size=200,   # smoke-test cap -- remove this line for a full run later
    )
    val_ds = ShardedSEN12MSDataset(
        root=cfg["data"]["dataset_root"], split="val",
        seed=cfg["data"]["seed"],
        train_frac=cfg["data"]["train_split"],
        val_frac=cfg["data"]["val_split"],
        subset_size=200,   # smoke-test cap -- remove this line for a full run later
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                               shuffle=True, num_workers=cfg["training"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                             shuffle=False, num_workers=cfg["training"]["num_workers"])

    model = UNet(**cfg["model_params"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["mixed_precision"])

    logger = ExperimentLogger(
        use_wandb=cfg["logging"]["use_wandb"],
        project=cfg["logging"]["project"],
        run_name=cfg["logging"]["run_name"],
        config=cfg,
    )

    start_epoch = 0
    best_val_loss = float("inf")
    if cfg["training"]["resume_from"]:
        state = load_checkpoint(cfg["training"]["resume_from"], model, optimizer, map_location=device)
        start_epoch = state["epoch"] + 1
        best_val_loss = state.get("best_metric", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            sar = batch["sar"].to(device)
            eo = batch["eo"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg["training"]["mixed_precision"]):
                pred = model(sar)
                loss = torch.nn.functional.l1_loss(pred, eo)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if step % 50 == 0:
                logger.log({"train/l1_loss": loss.item(), "epoch": epoch})

        avg_train_loss = running_loss / max(1, len(train_loader))

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sar = batch["sar"].to(device)
                eo = batch["eo"].to(device)
                pred = model(sar)
                val_loss += torch.nn.functional.l1_loss(pred, eo).item()
        avg_val_loss = val_loss / max(1, len(val_loader))

        logger.log({"train/epoch_l1_loss": avg_train_loss,
                     "val/l1_loss": avg_val_loss, "epoch": epoch})
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        if epoch % cfg["training"]["save_every_n_epochs"] == 0:
            save_checkpoint(f"{cfg['training']['checkpoint_dir']}/last.pt", model, optimizer, epoch, best_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(f"{cfg['training']['checkpoint_dir']}/best.pt", model, optimizer, epoch, best_val_loss)

    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
