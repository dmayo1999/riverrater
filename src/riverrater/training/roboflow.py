"""
Roboflow dataset download helpers for the playing-cards bootstrap dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

from riverrater.training.paths import TrainingPaths, ensure_training_layout


ROBOFLOW_WORKSPACE = "augmented-startups"
ROBOFLOW_PROJECT = "playing-cards-ow27d"
ROBOFLOW_VERSION = 4
ROBOFLOW_FORMAT = "yolov8"

MANUAL_DOWNLOAD_URL = (
    "https://universe.roboflow.com/augmented-startups/"
    "playing-cards-ow27d/dataset/4/download/yolov8"
)


class RoboflowDownloadError(RuntimeError):
    """Raised when the Roboflow dataset cannot be downloaded."""


def resolve_api_key(explicit_key: str | None = None) -> str | None:
    """Return API key from argument or ``ROBOFLOW_API_KEY`` env var."""
    if explicit_key:
        return explicit_key.strip() or None
    env_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    return env_key or None


def download_playing_cards_dataset(
    output_dir: Path | str | None = None,
    *,
    api_key: str | None = None,
    workspace: str = ROBOFLOW_WORKSPACE,
    project: str = ROBOFLOW_PROJECT,
    version: int = ROBOFLOW_VERSION,
    yolo_format: str = ROBOFLOW_FORMAT,
) -> Path:
    """
    Download the Roboflow playing-cards dataset in YOLOv8 format.

    Parameters
    ----------
    output_dir:
        Destination directory (defaults to ``data/training/raw``).
    api_key:
        Roboflow API key.  Falls back to ``ROBOFLOW_API_KEY`` environment
        variable.

    Returns
    -------
    Path
        Directory containing the downloaded dataset.

    Raises
    ------
    RoboflowDownloadError
        If no API key is available or the Roboflow client fails.
    """
    paths = ensure_training_layout()
    destination = Path(output_dir) if output_dir is not None else paths.raw
    destination.mkdir(parents=True, exist_ok=True)

    key = resolve_api_key(api_key)
    if not key:
        raise RoboflowDownloadError(
            "Roboflow API key required. Set ROBOFLOW_API_KEY or pass --api-key. "
            f"Manual download: {MANUAL_DOWNLOAD_URL}"
        )

    try:
        from roboflow import Roboflow  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RoboflowDownloadError(
            'Install train extras: pip install "riverrater[train]"'
        ) from exc

    try:
        rf = Roboflow(api_key=key)
        rf_project = rf.workspace(workspace).project(project)
        dataset = rf_project.version(version).download(
            yolo_format,
            location=str(destination),
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise RoboflowDownloadError(f"Roboflow download failed: {exc}") from exc

    # Roboflow may nest under a subfolder; return the path it reports.
    if hasattr(dataset, "location"):
        return Path(dataset.location)
    return destination


def manual_download_instructions(paths: TrainingPaths | None = None) -> str:
    """Human-readable steps when an API key is unavailable."""
    layout = paths or TrainingPaths.from_root()
    return (
        "Manual Roboflow download\n"
        "------------------------\n"
        f"1. Open {MANUAL_DOWNLOAD_URL}\n"
        "2. Sign in to Roboflow Universe (free account).\n"
        f"3. Extract the ZIP into: {layout.raw}\n"
        "4. Verify the tree contains train/images, train/labels, valid/, test/.\n"
        "5. Run: python scripts/augment_cards.py\n"
    )