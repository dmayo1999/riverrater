"""
YOLOv8n training scaffold for RiverRater card detection (Phase 3).

Provides Colab-oriented helpers and a local CLI entry point via
``scripts/train_yolo.py``.  Actual GPU training is expected on Google Colab;
this module validates dataset config and documents the export path for
``models/yolov8n_cards/best.pt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2

from riverrater.game.state import Rank, Suit
from riverrater.training.augmentation import build_class_names
from riverrater.training.paths import PROJECT_ROOT, TrainingPaths


# mAP@0.5 target from docs/BUILD_PLAN.md Phase 3 exit criteria.
MAP50_TARGET = 0.85

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "yolov8n_cards"
DEFAULT_WEIGHTS_PATH = DEFAULT_MODEL_DIR / "best.pt"
DEFAULT_BASE_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16


@dataclass(frozen=True)
class TrainingConfig:
    """Resolved YOLO training parameters."""

    data_yaml: Path
    base_model: str
    epochs: int
    imgsz: int
    batch: int
    export_path: Path
    project: str = "runs/detect"
    name: str = "train"


class TrainingDataError(ValueError):
    """Raised when required training images or labels are missing."""


class ClassAlignmentError(ValueError):
    """Raised when ``cards.yaml`` class order diverges from ``CLASS_MAP``."""


def default_weights_path() -> Path:
    """Return the canonical path for exported YOLO weights."""
    return DEFAULT_WEIGHTS_PATH


def parse_cards_yaml_names(cards_yaml: Path) -> list[str]:
    """
    Parse the ``names:`` list from ``cards.yaml`` without requiring PyYAML.

    Parameters
    ----------
    cards_yaml:
        Path to the YOLO dataset configuration file.

    Returns
    -------
    list[str]
        Class names in YOLO index order (IDs 0–51).
    """
    if not cards_yaml.is_file():
        raise FileNotFoundError(f"cards.yaml not found: {cards_yaml}")

    names: list[str] = []
    in_names = False
    for line in cards_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        if stripped.startswith("- "):
            names.append(stripped[2:].strip())
            continue
        if stripped and not stripped.startswith("#"):
            break
    return names


def class_map_to_names(class_map: dict[int, tuple[Rank, Suit]]) -> list[str]:
    """Convert a YOLO class-index map to two-character card name strings."""
    return [
        f"{class_map[idx][0].value}{class_map[idx][1].value}"
        for idx in sorted(class_map)
    ]


def validate_class_alignment(
    cards_yaml: Path,
    class_map: dict[int, tuple[Rank, Suit]],
) -> list[str]:
    """
    Verify ``cards.yaml`` names match ``YOLOEngine.CLASS_MAP`` ordering.

    Parameters
    ----------
    cards_yaml:
        Dataset config path (typically ``data/training/cards.yaml``).
    class_map:
        Mapping from YOLO class ID to ``(Rank, Suit)`` tuples.

    Returns
    -------
    list[str]
        The validated 52-class name list.

    Raises
    ------
    ClassAlignmentError
        If class count or ordering does not match.
    """
    yaml_names = parse_cards_yaml_names(cards_yaml)
    map_names = class_map_to_names(class_map)
    canonical = build_class_names()

    if len(yaml_names) != 52:
        raise ClassAlignmentError(
            f"cards.yaml defines {len(yaml_names)} classes; expected 52."
        )
    if yaml_names != map_names:
        first = next(
            (i for i, (a, b) in enumerate(zip(yaml_names, map_names)) if a != b),
            None,
        )
        raise ClassAlignmentError(
            f"cards.yaml class order diverges from CLASS_MAP at index {first}: "
            f"yaml={yaml_names[first]!r}, CLASS_MAP={map_names[first]!r}."
        )
    if yaml_names != canonical:
        raise ClassAlignmentError(
            "cards.yaml names do not match build_class_names() canonical order."
        )
    return yaml_names


def resolve_training_config(
    *,
    data_yaml: Path | str | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    export_path: Path | str | None = None,
    project: str | None = None,
    name: str = "train",
) -> TrainingConfig:
    """Build a :class:`TrainingConfig` using project defaults."""
    paths = TrainingPaths.from_root()
    resolved_data = Path(data_yaml) if data_yaml is not None else paths.cards_yaml
    resolved_export = (
        Path(export_path) if export_path is not None else default_weights_path()
    )
    return TrainingConfig(
        data_yaml=resolved_data.resolve(),
        base_model=base_model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        export_path=resolved_export.resolve(),
        project=project or "runs/detect",
        name=name,
    )


def validate_training_data(
    data_yaml: Path,
    *,
    require_train_images: bool = True,
) -> dict[str, str | int | bool]:
    """
    Check that ``cards.yaml`` exists and referenced train images are present.

    Returns a summary dict suitable for ``--dry-run`` output.  Raises
    :class:`TrainingDataError` when required paths are missing.
    """
    if not data_yaml.is_file():
        raise TrainingDataError(f"Dataset config not found: {data_yaml}")

    data_root = data_yaml.parent
    train_rel = _parse_yaml_scalar(data_yaml, "train")
    val_rel = _parse_yaml_scalar(data_yaml, "val")
    train_dir = (data_root / train_rel).resolve() if train_rel else None

    train_images = 0
    if train_dir is not None and train_dir.is_dir():
        train_images = _count_images(train_dir)

    if require_train_images and train_images == 0:
        raise TrainingDataError(
            f"No training images found under {train_dir}. "
            "Download Roboflow data (scripts/download_roboflow.py) and/or "
            "run augmentation (scripts/augment_cards.py) before training."
        )

    return {
        "data_yaml": str(data_yaml),
        "data_root": str(data_root),
        "train_dir": str(train_dir) if train_dir else "",
        "val_dir": str((data_root / val_rel).resolve()) if val_rel else "",
        "train_images": train_images,
        "ready": train_images > 0,
    }


def _parse_yaml_scalar(path: Path, key: str) -> str | None:
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            value = stripped[len(prefix) :].strip()
            return value or None
    return None


def _count_images(directory: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(
        1
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in extensions
    )


def colab_training_instructions(
    *,
    repo_url: str = "https://github.com/dmayo1999/riverrater",
    data_yaml: Path | str | None = None,
) -> str:
    """
    Return copy-paste Google Colab setup and training cells.

    Human operators run these steps on a Colab T4 GPU; local MPS training is
    not supported due to known PyTorch MPS loss-calculation slowdowns.
    """
    paths = TrainingPaths.from_root()
    resolved_yaml = Path(data_yaml) if data_yaml is not None else paths.cards_yaml
    colab_yaml = f"/content/RR/{resolved_yaml.relative_to(PROJECT_ROOT)}"
    export_hint = f"/content/RR/{DEFAULT_WEIGHTS_PATH.relative_to(PROJECT_ROOT)}"

    return f"""\
# RiverRater YOLOv8n — Google Colab training (T4 GPU)

## 1. Open Colab and enable GPU
Runtime → Change runtime type → T4 GPU

## 2. Clone repo and install dependencies
```python
!git clone {repo_url}.git /content/RR
%cd /content/RR
!pip install -e ".[train]"
```

## 3. Prepare dataset (on your Mac, before Colab)
```bash
export ROBOFLOW_API_KEY=your_key   # optional
python scripts/download_roboflow.py
python scripts/augment_cards.py --variants 3
# Commit/push augmented data or upload data/training/ to Colab Drive
```

## 4. Point cards.yaml at the Colab dataset root
Edit ``path:`` in ``{colab_yaml}`` to the folder containing ``augmented/`` and ``raw/``
(absolute path, e.g. ``/content/RR/data/training``).

## 5. Train
```python
from pathlib import Path
from riverrater.training.yolo_train import (
    MAP50_TARGET,
    resolve_training_config,
    run_training,
    validate_class_alignment,
)
from riverrater.vision.yolo_engine import YOLOEngine

cards_yaml = Path("{colab_yaml}")
validate_class_alignment(cards_yaml, YOLOEngine.CLASS_MAP)

config = resolve_training_config(data_yaml=cards_yaml)
results = run_training(config)
print("mAP50 target:", MAP50_TARGET)
```

Or minimal Ultralytics-only cell:
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="{colab_yaml}", epochs=100, imgsz=640, batch=16)
```

## 6. Export weights to the repo
```python
from pathlib import Path
from shutil import copy2
src = Path("runs/detect/train/weights/best.pt")
dst = Path("{export_hint}")
dst.parent.mkdir(parents=True, exist_ok=True)
copy2(src, dst)
print("Copied to", dst)
```

Download ``best.pt`` and place at ``models/yolov8n_cards/best.pt`` locally, or
``scp`` from Colab.  Target: **mAP@0.5 ≥ {MAP50_TARGET:.0%}** on held-out live-dealer frames.

## 7. Verify locally
```bash
pip install -e ".[ml]"
python -m pytest tests/test_yolo_smoke.py -v
```
"""


def run_training(config: TrainingConfig) -> object:
    """
    Run YOLOv8 training via Ultralytics and copy ``best.pt`` to *export_path*.

    Parameters
    ----------
    config:
        Resolved training configuration.

    Returns
    -------
    object
        Ultralytics training results object.

    Raises
    ------
    ImportError
        If ``ultralytics`` is not installed.
    TrainingDataError
        If the dataset is not ready.
    """
    validate_training_data(config.data_yaml)
    from ultralytics import YOLO  # type: ignore[import-untyped]

    model = YOLO(config.base_model)
    results = model.train(
        data=str(config.data_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        project=config.project,
        name=config.name,
    )

    weights_src = Path(config.project) / config.name / "weights" / "best.pt"
    if weights_src.is_file():
        config.export_path.parent.mkdir(parents=True, exist_ok=True)
        copy2(weights_src, config.export_path)

    return results