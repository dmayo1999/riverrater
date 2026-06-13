"""
Training data utilities for YOLOv8 card detection (Phase 3).

Provides label I/O, augmentation, and dataset path helpers used by
``scripts/augment_cards.py`` and ``scripts/download_roboflow.py``.
"""

from riverrater.training.augmentation import AugmentStats, CardAugmentor
from riverrater.training.labels import YoloLabel, parse_label_file, write_label_file
from riverrater.training.paths import TrainingPaths, ensure_training_layout
from riverrater.training.yolo_train import (
    MAP50_TARGET,
    TrainingConfig,
    colab_training_instructions,
    default_weights_path,
    resolve_training_config,
    run_training,
    validate_class_alignment,
)

__all__ = [
    "AugmentStats",
    "CardAugmentor",
    "MAP50_TARGET",
    "TrainingConfig",
    "TrainingPaths",
    "YoloLabel",
    "colab_training_instructions",
    "default_weights_path",
    "ensure_training_layout",
    "parse_label_file",
    "resolve_training_config",
    "run_training",
    "validate_class_alignment",
    "write_label_file",
]