"""
Training script for the conditional diffusion model (Stage 11), with an
optional semantic-consistency term (Stage 12, enabled via config).

    python -m src.training.train_diffusion --config configs/diffusion.yaml

Each step:
    1. Sample a random timestep t per example
    2. Add noise to the real EO image according to t (forward diffusion)
    3. Ask the network to predict the noise, conditioned on SAR + t
    4. MSE(predicted_noise, actual_noise) is the base diffusion loss
    5. (optional) periodically run full sampling on a few examples and add
       the semantic-consistency loss between generated and real EO

Note: full reverse sampling (p_sample_loop) is expensive (up to `timesteps`
network calls per image), so the semantic loss is only evaluated every
`semantic_eval_every` steps, not every step, to keep training tractable on
a single Kaggle GPU.
"""
from __future__ import annotations
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.sharded_dataset import ShardedSEN12MSDataset, ShardAwareSampler
from src.data.transforms import PairedAugment
from src.models.diffusion import ConditionalDiffusionUNet, GaussianDiffusion
from src.models.segmentation import SimpleSegmentationNet, semantic_consistency_loss
from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import ExperimentLogger

SEMANTIC_EVAL_EVERY = 200  # steps


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ShardedSEN12MSDataset(
        root=cfg["data"]["dataset_root"], split="train",
        seed=cfg["data"]["seed"],
        train_frac=cfg["data"]["train_split"], val_frac=cfg["data"]["val_split"],
        subset_size=500,
    )
    train_sampler = ShardAwareSampler(train_ds, seed=cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                               sampler=train_sampler, num_workers=cfg["training"]["num_workers"])

    mp = cfg["model_params"]
    model = ConditionalDiffusionUNet(
        out_channels=mp["out_channels"], cond_channels=mp["cond_channels"],
        base_channels=mp["base_channels"], channel_mults=tuple(mp["channel_mults"]),
        num_res_blocks=mp["num_res_blocks"],
    ).to(device)

    diffusion = GaussianDiffusion(timesteps=mp["timesteps"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])

    seg_net = None
    sc_cfg = cfg.get("semantic_consistency", {})
    if sc_cfg.get("enabled", False):
        seg_net = SimpleSegmentationNet().to(device)
        if sc_cfg.get("segmentation_ckpt"):
            seg_net.load_state_dict(torch.load(sc_cfg["segmentation_ckpt"], map_location=device))
        seg_net.eval()
        for p in seg_net.parameters():
            p.requires_grad = False

    logger = ExperimentLogger(cfg["logging"]["use_wandb"], cfg["logging"]["project"],
                               cfg["logging"]["run_name"], cfg)

    global_step = 0
    grad_accum = cfg["training"]["grad_accum_steps"]

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            sar = batch["sar"].to(device)
            eo = batch["eo"].to(device)
            t = torch.randint(0, mp["timesteps"], (sar.shape[0],), device=device).long()

            loss = diffusion.training_loss(model, eo, sar, t)

            # Optional semantic-consistency term — evaluated sparsely because
            # it requires a full reverse-diffusion sampling pass.
            if seg_net is not None and global_step % SEMANTIC_EVAL_EVERY == 0:
                with torch.no_grad():
                    generated = diffusion.p_sample_loop(model, sar, eo.shape, device)
                sem_loss = semantic_consistency_loss(seg_net, generated, eo)
                loss = loss + sc_cfg["lambda_semantic"] * sem_loss
                logger.log({"train/semantic_loss": sem_loss.item()}, step=global_step)

            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % 50 == 0:
                logger.log({"train/diffusion_loss": loss.item(), "epoch": epoch}, step=global_step)

            global_step += 1

        print(f"Epoch {epoch}: last_loss={loss.item():.4f}")

        if epoch % cfg["training"]["save_every_n_epochs"] == 0:
            save_checkpoint(f"{cfg['training']['checkpoint_dir']}/last.pt", model, optimizer, epoch)

    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
