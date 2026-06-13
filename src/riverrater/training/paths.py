"""
Standard directory layout for YOLO training data under ``data/training/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAINING_ROOT = PROJECT_ROOT / "data" / "training"


@dataclass(frozen=True)
class TrainingPaths:
    """Resolved paths for the Phase 3 training dataset."""

    root: Path
    raw: Path
    augmented_images: Path
    augmented_labels: Path
    live_dealer_images: Path
    live_dealer_labels: Path
    backgrounds: Path
    cards_yaml: Path

    @classmethod
    def from_root(cls, root: Path | str | None = None) -> TrainingPaths:
        base = Path(root) if root is not None else DEFAULT_TRAINING_ROOT
        return cls(
            root=base,
            raw=base / "raw",
            augmented_images=base / "augmented" / "images",
            augmented_labels=base / "augmented" / "labels",
            live_dealer_images=base / "live_dealer" / "images",
            live_dealer_labels=base / "live_dealer" / "labels",
            backgrounds=base / "backgrounds",
            cards_yaml=base / "cards.yaml",
        )


def ensure_training_layout(root: Path | str | None = None) -> TrainingPaths:
    """Create the standard ``data/training/`` directory tree if missing."""
    paths = TrainingPaths.from_root(root)
    for directory in (
        paths.raw,
        paths.augmented_images,
        paths.augmented_labels,
        paths.live_dealer_images,
        paths.live_dealer_labels,
        paths.backgrounds,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def discover_image_label_pairs(
    images_dir: Path,
    labels_dir: Path,
) -> list[tuple[Path, Path]]:
    """
    Pair images in *images_dir* with matching ``.txt`` labels in *labels_dir*.

    Images without labels are skipped. Supported extensions: ``.jpg``, ``.jpeg``,
    ``.png``, ``.webp``.
    """
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    pairs: list[tuple[Path, Path]] = []

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in extensions:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            pairs.append((image_path, label_path))

    return pairs