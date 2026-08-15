"""
Training script for the Pix2Pix baseline (Stage 10) -- FULL SCALE, resumable.

    python -m src.training.train_pix2pix --config configs/pix2pix.yaml

Resumable across multiple Kaggle sessions: if both generator_last.pt and
discriminator_last.pt exist in the checkpoint_dir, training resumes from
the saved epoch instead of starting over.
"""
from __future__ import annotations
import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.sharded_dataset import ShardedSEN12MSDataset, ShardAwareSampler
from src.models.pix2pix import Pix2PixGenerator, PatchGANDiscriminator
from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logger import ExperimentLogger


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ShardedSEN12MSDataset(
        root=cfg["data"]["dataset_root"], split="train",
        seed=cfg["data"]["seed"],
        train_frac=cfg["data"]["train_split"], val_frac=cfg["data"]["val_split"],
    )
    print(f"Train samples: {len(train_ds)}")
    train_sampler = ShardAwareSampler(train_ds, seed=cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                               sampler=train_sampler, num_workers=cfg["training"]["num_workers"])

    mp = cfg["model_params"]
    G = Pix2PixGenerator(in_channels=mp["in_channels"], out_channels=mp["out_channels"],
                          base_channels=mp["generator_base_channels"]).to(device)
    D = PatchGANDiscriminator(in_channels=mp["in_channels"] + mp["out_channels"],
                               base_channels=mp["discriminator_base_channels"]).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["training"]["learning_rate"],
                              betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["training"]["learning_rate"],
                              betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]))

    bce = torch.nn.BCEWithLogitsLoss()
    l1 = torch.nn.L1Loss()
    lambda_l1 = mp["lambda_l1"]

    # Label smoothing on the discriminator's real-image target -- a standard,
    # simple fix for the discriminator-collapse behavior observed in earlier
    # runs (loss_d -> ~0 while loss_g_adv climbs unboundedly). Real targets
    # are 0.9 instead of 1.0, softening the discriminator's confidence.
    REAL_LABEL = 0.9

    logger = ExperimentLogger(cfg["logging"]["use_wandb"], cfg["logging"]["project"],
                               cfg["logging"]["run_name"], cfg)

    start_epoch = 0
    g_ckpt = f"{cfg['training']['checkpoint_dir']}/generator_last.pt"
    d_ckpt = f"{cfg['training']['checkpoint_dir']}/discriminator_last.pt"
    if os.path.exists(g_ckpt) and os.path.exists(d_ckpt):
        state_g = load_checkpoint(g_ckpt, G, opt_g, map_location=device)
        state_d = load_checkpoint(d_ckpt, D, opt_d, map_location=device)
        start_epoch = state_g["epoch"] + 1
        print(f"Resumed Pix2Pix from epoch {start_epoch}")
    else:
        print("No Pix2Pix checkpoint found -- starting fresh from epoch 0.")

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        G.train(); D.train()
        for step, batch in enumerate(train_loader):
            sar = batch["sar"].to(device)
            eo_real = batch["eo"].to(device)

            # ---- Discriminator step ----
            with torch.no_grad():
                eo_fake = G(sar)
            pred_real = D(sar, eo_real)
            pred_fake = D(sar, eo_fake)
            loss_d = 0.5 * (bce(pred_real, torch.full_like(pred_real, REAL_LABEL)) +
                             bce(pred_fake, torch.zeros_like(pred_fake)))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # ---- Generator step ----
            eo_fake = G(sar)
            pred_fake_for_g = D(sar, eo_fake)
            loss_g_adv = bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_g_l1 = l1(eo_fake, eo_real)
            loss_g = loss_g_adv + lambda_l1 * loss_g_l1
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            if step % 50 == 0:
                logger.log({"loss_d": loss_d.item(), "loss_g_adv": loss_g_adv.item(),
                             "loss_g_l1": loss_g_l1.item(), "epoch": epoch})

        print(f"Epoch {epoch}: loss_d={loss_d.item():.4f} loss_g={loss_g.item():.4f}")

        if epoch % cfg["training"]["save_every_n_epochs"] == 0:
            save_checkpoint(g_ckpt, G, opt_g, epoch)
            save_checkpoint(d_ckpt, D, opt_d, epoch)

    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
