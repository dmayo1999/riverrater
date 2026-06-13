#!/usr/bin/env python3
"""
Train YOLOv8n card detector — scaffold for Google Colab (T4 GPU).

Local macOS training is not recommended (PyTorch MPS training is slow/buggy).
Use ``--colab-instructions`` to print the Colab workflow, or ``--dry-run`` to
validate ``cards.yaml`` and dataset layout without invoking Ultralytics.

Example (Colab or machine with GPU + data):

    python scripts/train_yolo.py --epochs 100 --imgsz 640

Example (local validation only):

    python scripts/train_yolo.py --dry-run
    python scripts/train_yolo.py --colab-instructions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from riverrater.training.yolo_train import (  # noqa: E402
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    MAP50_TARGET,
    TrainingDataError,
    colab_training_instructions,
    resolve_training_config,
    run_training,
    validate_class_alignment,
    validate_training_data,
)
from riverrater.vision.yolo_engine import YOLOEngine  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLOv8n on playing-card data. "
            "Recommended: Google Colab with T4 GPU."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to cards.yaml (default: data/training/cards.yaml)",
    )
    parser.add_argument(
        "--base-model",
        default="yolov8n.pt",
        help="Ultralytics base weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Input image size (default: {DEFAULT_IMGSZ})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Batch size (default: {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Copy best.pt here after training (default: models/yolov8n_cards/best.pt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CLASS_MAP alignment and dataset paths; do not train",
    )
    parser.add_argument(
        "--colab-instructions",
        action="store_true",
        help="Print Google Colab training workflow and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.colab_instructions:
        print(colab_training_instructions(data_yaml=args.data))
        return 0

    config = resolve_training_config(
        data_yaml=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        export_path=args.export,
    )

    try:
        names = validate_class_alignment(config.data_yaml, YOLOEngine.CLASS_MAP)
        summary = validate_training_data(
            config.data_yaml,
            require_train_images=not args.dry_run,
        )
    except (TrainingDataError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("YOLO training scaffold")
    print(f"  data.yaml:     {config.data_yaml}")
    print(f"  classes:       {len(names)} (aligned with YOLOEngine.CLASS_MAP)")
    print(f"  train images:  {summary['train_images']}")
    print(f"  export path:   {config.export_path}")
    print(f"  mAP@0.5 target: {MAP50_TARGET:.0%}")

    if args.dry_run:
        print("Dry run complete — dataset ready for Colab training.")
        return 0

    try:
        run_training(config)
    except ImportError:
        print(
            "ultralytics not installed. Install with: pip install -e '.[train]'\n"
            "For GPU training, use --colab-instructions and run on Google Colab.",
            file=sys.stderr,
        )
        return 1
    except TrainingDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Training complete. Weights exported to {config.export_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())