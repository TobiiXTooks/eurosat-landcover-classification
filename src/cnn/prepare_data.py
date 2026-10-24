"""CLI: extract EuroSAT and build the shared split + normalization artifacts.

Usage::

    python -m cnn.prepare_data --zip data/EuroSAT_RGB.zip

This is a one-time step. It extracts the archive (only when missing), discovers
the class order, builds the stratified 70/15/15 split (seed 42), computes the
RGB normalization from the training split, and persists everything under
``data/processed/`` for both tasks to reuse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import data as data_mod
from .utils import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the EuroSAT dataset.")
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/EuroSAT_RGB.zip"),
        help="Path to the EuroSAT_RGB.zip archive.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory holding the dataset (extraction target).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Where to write artifacts (default: <data-root>/processed).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the normalization progress bar.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    processed_dir = args.processed_dir or (args.data_root / data_mod.PROCESSED_DIRNAME)

    print(f"Extracting dataset (if needed) from {args.zip} ...")
    extracted_dir = data_mod.extract_dataset(args.zip, args.data_root)

    class_order = data_mod.discover_classes(extracted_dir)
    print(f"Discovered {len(class_order)} classes: {class_order}")

    samples = data_mod.list_samples(extracted_dir, class_order)
    print(f"Found {len(samples)} images.")

    splits = data_mod.stratified_split(samples, seed=args.seed)
    for name in ("train", "val", "test"):
        print(f"  {name:>5}: {len(splits[name])} images")

    mean, std = data_mod.compute_normalization(
        extracted_dir, splits["train"], show_progress=not args.no_progress
    )
    print(f"Normalization (train only): mean={mean}, std={std}")

    # Image size is read from the first training image for the saved config.
    from PIL import Image

    with Image.open(extracted_dir / splits["train"][0][0]) as img:
        image_size = img.size[0]

    data_mod.save_artifacts(
        processed_dir=processed_dir,
        extracted_dir=extracted_dir,
        class_order=class_order,
        splits=splits,
        mean=mean,
        std=std,
        seed=args.seed,
        image_size=image_size,
    )
    print(f"Saved artifacts to {processed_dir}/")


if __name__ == "__main__":
    main()
