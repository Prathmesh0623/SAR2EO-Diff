"""
Data preparation entry point (Stage 4 / Stage 26 Notebook 2 equivalent as a
script). On Kaggle, run this once after attaching the SEN12MS dataset to
verify the pipeline can discover and load pairs before launching any
training job.

    python scripts/prepare_data.py --root /kaggle/input/sen12ms --subset 20
"""
import argparse
from src.data.dataset import SEN12MSPairedDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--subset", type=int, default=None,
                         help="Limit to N scene pairs for a quick smoke test.")
    args = parser.parse_args()

    ds = SEN12MSPairedDataset(root=args.root, split="train",
                               patch_size=args.patch_size, subset_size=args.subset)
    print(f"Discovered {len(ds)} training pairs.")
    if len(ds) == 0:
        print("No pairs found — check --root path and folder naming "
              "(see docstring in src/data/dataset.py::_discover_pairs).")
        return

    sample = ds[0]
    print("Sample SAR shape:", sample["sar"].shape)
    print("Sample EO shape:", sample["eo"].shape)
    print("SAR path:", sample["sar_path"])
    print("EO path:", sample["eo_path"])


if __name__ == "__main__":
    main()
