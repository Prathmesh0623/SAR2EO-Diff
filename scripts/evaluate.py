"""
Unified evaluation entry point (Stage 15).

    python scripts/evaluate.py --config configs/baseline_unet.yaml \
        --checkpoint checkpoints/unet/best.pt \
        --output results/tables/unet_test_metrics.json
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SEN12MSPairedDataset
from src.evaluation.evaluate import evaluate_model


def build_model(cfg):
    model_type = cfg["model"]
    if model_type == "unet":
        from src.models.unet import UNet
        return UNet(**cfg["model_params"])
    elif model_type == "pix2pix":
        from src.models.pix2pix import Pix2PixGenerator
        mp = cfg["model_params"]
        return Pix2PixGenerator(in_channels=mp["in_channels"], out_channels=mp["out_channels"],
                                 base_channels=mp["generator_base_channels"])
    else:
        raise ValueError(f"evaluate.py currently supports unet/pix2pix directly; "
                          f"for diffusion, write a custom generate_fn using GaussianDiffusion.p_sample_loop.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    ds = SEN12MSPairedDataset(
        root=cfg["data"]["dataset_root"], split=args.split,
        patch_size=cfg["data"]["patch_size"],
        sar_channels=tuple(cfg["data"]["sar_channels"]),
        seed=cfg["data"]["seed"],
        train_frac=cfg["data"]["train_split"], val_frac=cfg["data"]["val_split"],
    )
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    results = evaluate_model(model, loader, device, output_path=args.output)
    print(f"PSNR: {results['psnr_mean']:.3f}  SSIM: {results['ssim_mean']:.3f}  "
          f"LPIPS: {results['lpips_mean']:.3f}  (n={results['num_samples']})")
    print(f"Full results written to {args.output}")


if __name__ == "__main__":
    main()
