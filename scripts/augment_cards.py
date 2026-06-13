#!/usr/bin/env python3
"""
Augment YOLO card images for live-dealer-style training data.

Applies rotation, brightness jitter, motion blur, perspective warp, and
casino-felt background compositing while preserving label integrity.

Example:

    python scripts/augment_cards.py \\
        --input-images data/training/raw/train/images \\
        --input-labels data/training/raw/train/labels \\
        --output-images data/training/augmented/images \\
        --output-labels data/training/augmented/labels \\
        --variants 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from riverrater.training.augmentation import CardAugmentor
from riverrater.training.paths import TrainingPaths, ensure_training_layout


def _default_raw_split(split: str) -> tuple[Path, Path]:
    paths = TrainingPaths.from_root()
    base = paths.raw / split
    return base / "images", base / "labels"


def build_parser() -> argparse.ArgumentParser:
    default_images, default_labels = _default_raw_split("train")
    paths = TrainingPaths.from_root()

    parser = argparse.ArgumentParser(
        description="Augment YOLO card training images for live-dealer conditions.",
    )
    parser.add_argument(
        "--input-images",
        type=Path,
        default=default_images,
        help="Directory of source images",
    )
    parser.add_argument(
        "--input-labels",
        type=Path,
        default=default_labels,
        help="Directory of YOLO label .txt files",
    )
    parser.add_argument(
        "--output-images",
        type=Path,
        default=paths.augmented_images,
        help="Output image directory",
    )
    parser.add_argument(
        "--output-labels",
        type=Path,
        default=paths.augmented_labels,
        help="Output label directory",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=3,
        help="Augmented variants per source image (default: 3)",
    )
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="Skip copying unmodified source images to output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentation",
    )
    parser.add_argument(
        "--backgrounds",
        type=Path,
        default=None,
        help="Optional directory of felt background images (.jpg/.png)",
    )
    return parser


def _load_backgrounds(directory: Path | None) -> list:
    import cv2

    if directory is None or not directory.is_dir():
        return []

    backgrounds = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        image = cv2.imread(str(path))
        if image is not None:
            backgrounds.append(image)
    return backgrounds


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_training_layout()

    if not args.input_images.is_dir():
        print(
            f"Input images directory not found: {args.input_images}\n"
            "Download data first: python scripts/download_roboflow.py --manual",
            file=sys.stderr,
        )
        return 1

    if not args.input_labels.is_dir():
        print(f"Input labels directory not found: {args.input_labels}", file=sys.stderr)
        return 1

    backgrounds = _load_backgrounds(args.backgrounds)
    augmentor = CardAugmentor(seed=args.seed, felt_backgrounds=backgrounds or None)

    stats = augmentor.augment_dataset(
        args.input_images,
        args.input_labels,
        args.output_images,
        args.output_labels,
        variants_per_image=max(0, args.variants),
        include_original=not args.no_original,
    )

    print("Augmentation complete")
    print(f"  Source images:      {stats.source_images}")
    print(f"  Variants/image:     {stats.variants_per_image}")
    print(f"  Include original:   {stats.include_original}")
    print(f"  Output images:      {stats.output_images}")
    print(f"  Output label files: {stats.output_labels}")
    if stats.skipped_empty_labels:
        print(f"  Skipped (no labels): {stats.skipped_empty_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())