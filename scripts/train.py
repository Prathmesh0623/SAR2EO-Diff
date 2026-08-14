"""
Unified training entry point — dispatches to the right training script
based on the `model` field in the config file.

    python scripts/train.py --config configs/baseline_unet.yaml
    python scripts/train.py --config configs/pix2pix.yaml
    python scripts/train.py --config configs/diffusion.yaml
"""
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_type = cfg["model"]
    if model_type == "unet":
        from src.training.train_unet import main as run
    elif model_type == "pix2pix":
        from src.training.train_pix2pix import main as run
    elif model_type == "diffusion":
        from src.training.train_diffusion import main as run
    else:
        raise ValueError(f"Unknown model type in config: {model_type}")

    run(args.config)


if __name__ == "__main__":
    main()
