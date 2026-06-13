#!/usr/bin/env python3
"""
Download the Roboflow playing-cards dataset (YOLOv8 format).

Requires ``riverrater[train]`` and a Roboflow API key:

    export ROBOFLOW_API_KEY=your_key_here
    pip install -e ".[train]"
    python scripts/download_roboflow.py

Manual alternative (no API key):

    https://universe.roboflow.com/augmented-startups/playing-cards-ow27d/dataset/4/download/yolov8

Extract the ZIP into ``data/training/raw/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from riverrater.training.paths import TrainingPaths, ensure_training_layout
from riverrater.training.roboflow import (
    RoboflowDownloadError,
    download_playing_cards_dataset,
    manual_download_instructions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Roboflow playing-cards dataset for YOLOv8 training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: data/training/raw)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Roboflow API key (default: ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Print manual download instructions and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ensure_training_layout()

    if args.manual:
        print(manual_download_instructions(paths))
        return 0

    output_dir = args.output or paths.raw
    try:
        location = download_playing_cards_dataset(
            output_dir,
            api_key=args.api_key,
        )
    except RoboflowDownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(file=sys.stderr)
        print(manual_download_instructions(TrainingPaths.from_root(output_dir.parent)), file=sys.stderr)
        return 1

    print(f"Dataset downloaded to: {location}")
    print("Next step: python scripts/augment_cards.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())