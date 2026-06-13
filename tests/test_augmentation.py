"""
Tests for YOLO training data augmentation (Phase 3 pipeline).

Covers label parsing, geometric label integrity, output counts, and Roboflow
download helpers (mocked — no API key required).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from riverrater.training.augmentation import CardAugmentor, build_class_names
from riverrater.training.labels import (
    YoloLabel,
    parse_label_file,
    parse_label_line,
    pixel_boxes_to_labels,
    write_label_file,
)
from riverrater.training.paths import TrainingPaths, discover_image_label_pairs, ensure_training_layout
from riverrater.training.roboflow import (
    RoboflowDownloadError,
    download_playing_cards_dataset,
    manual_download_instructions,
    resolve_api_key,
)


# ---------------------------------------------------------------------------
# Label I/O
# ---------------------------------------------------------------------------


class TestYoloLabels:
    def test_parse_label_line(self) -> None:
        label = parse_label_line("3 0.512000 0.481000 0.220000 0.310000")
        assert label is not None
        assert label.class_id == 3
        assert label.cx == pytest.approx(0.512)
        assert label.w == pytest.approx(0.22)

    def test_parse_label_file_roundtrip(self, tmp_path: Path) -> None:
        labels = [
            YoloLabel(class_id=0, cx=0.5, cy=0.5, w=0.2, h=0.3),
            YoloLabel(class_id=12, cx=0.25, cy=0.75, w=0.1, h=0.15),
        ]
        path = tmp_path / "sample.txt"
        write_label_file(path, labels)
        loaded = parse_label_file(path)
        assert len(loaded) == 2
        assert loaded[0].class_id == 0
        assert loaded[1].class_id == 12

    def test_pixel_boxes_to_labels_clips_to_image(self) -> None:
        boxes = [(1, (-10.0, -10.0, 50.0, 60.0))]
        labels = pixel_boxes_to_labels(boxes, width=100, height=100)
        assert len(labels) == 1
        assert labels[0].cx >= 0.0
        assert labels[0].cy >= 0.0


# ---------------------------------------------------------------------------
# Augmentation — label integrity
# ---------------------------------------------------------------------------


def _synthetic_card_scene() -> tuple[np.ndarray, list[YoloLabel]]:
    """White card rectangle on dark background with a centred YOLO label."""
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    x1, y1, x2, y2 = 100, 50, 200, 150
    cv2.rectangle(image, (x1, y1), (x2, y2), (240, 240, 240), thickness=-1)
    labels = [
        YoloLabel(
            class_id=7,
            cx=(x1 + x2) / 2 / 300,
            cy=(y1 + y2) / 2 / 200,
            w=(x2 - x1) / 300,
            h=(y2 - y1) / 200,
        )
    ]
    return image, labels


class TestLabelIntegrity:
    def test_rotation_preserves_label_count(self) -> None:
        image, labels = _synthetic_card_scene()
        augmentor = CardAugmentor(seed=123)
        _, out_labels = augmentor.augment_single(image, labels, variant=0)
        assert len(out_labels) == 1
        assert out_labels[0].class_id == 7

    def test_rotation_label_stays_in_bounds(self) -> None:
        image, labels = _synthetic_card_scene()
        augmentor = CardAugmentor(seed=99)
        _, out_labels = augmentor.augment_single(image, labels, variant=0)
        label = out_labels[0]
        assert 0.0 <= label.cx <= 1.0
        assert 0.0 <= label.cy <= 1.0
        assert 0.0 < label.w <= 1.0
        assert 0.0 < label.h <= 1.0

    def test_brightness_does_not_change_labels(self) -> None:
        image, labels = _synthetic_card_scene()
        augmentor = CardAugmentor(seed=1)
        out_image, out_labels = augmentor.augment_single(image, labels, variant=1)
        assert out_labels == labels
        assert not np.array_equal(out_image, image)

    def test_blur_does_not_change_labels(self) -> None:
        image, labels = _synthetic_card_scene()
        augmentor = CardAugmentor(seed=2)
        _, out_labels = augmentor.augment_single(image, labels, variant=1)
        assert out_labels == labels

    def test_perspective_preserves_label_count(self) -> None:
        image, labels = _synthetic_card_scene()
        augmentor = CardAugmentor(seed=55)
        _, out_labels = augmentor.augment_single(image, labels, variant=2)
        assert len(out_labels) == 1

    def test_felt_background_updates_coordinates(self) -> None:
        image, labels = _synthetic_card_scene()
        augmentor = CardAugmentor(seed=77)
        out_image, out_labels = augmentor.augment_single(image, labels, variant=3)
        assert out_image.shape[0] >= image.shape[0]
        assert out_image.shape[1] >= image.shape[1]
        assert len(out_labels) == 1
        assert out_labels[0].cx != pytest.approx(labels[0].cx)

    def test_felt_background_with_custom_texture(self) -> None:
        image, labels = _synthetic_card_scene()
        felt = CardAugmentor(seed=10).generate_felt_background(640, 480)
        augmentor = CardAugmentor(seed=10, felt_backgrounds=[felt])
        out_image, out_labels = augmentor.augment_single(image, labels, variant=3)
        assert len(out_labels) == 1
        # Output canvas is resized to at least the source dimensions.
        assert out_image.shape == (360, 480, 3)


# ---------------------------------------------------------------------------
# Augmentation — output count
# ---------------------------------------------------------------------------


class TestOutputCount:
    def _write_sample_dataset(self, root: Path, count: int = 2) -> tuple[Path, Path]:
        images_dir = root / "images"
        labels_dir = root / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        for idx in range(count):
            image, labels = _synthetic_card_scene()
            image_path = images_dir / f"card_{idx:02d}.png"
            label_path = labels_dir / f"card_{idx:02d}.txt"
            cv2.imwrite(str(image_path), image)
            write_label_file(label_path, labels)

        return images_dir, labels_dir

    def test_augment_dataset_output_count_with_original(self, tmp_path: Path) -> None:
        images_dir, labels_dir = self._write_sample_dataset(tmp_path / "src", count=2)
        out_images = tmp_path / "out" / "images"
        out_labels = tmp_path / "out" / "labels"

        augmentor = CardAugmentor(seed=42)
        stats = augmentor.augment_dataset(
            images_dir,
            labels_dir,
            out_images,
            out_labels,
            variants_per_image=3,
            include_original=True,
        )

        # 2 sources × (1 original + 3 variants) = 8
        assert stats.source_images == 2
        assert stats.output_images == 8
        assert stats.output_labels == 8
        assert len(list(out_images.iterdir())) == 8
        assert len(list(out_labels.iterdir())) == 8

    def test_augment_dataset_output_count_without_original(self, tmp_path: Path) -> None:
        images_dir, labels_dir = self._write_sample_dataset(tmp_path / "src", count=3)
        out_images = tmp_path / "out" / "images"
        out_labels = tmp_path / "out" / "labels"

        stats = CardAugmentor(seed=7).augment_dataset(
            images_dir,
            labels_dir,
            out_images,
            out_labels,
            variants_per_image=2,
            include_original=False,
        )

        assert stats.output_images == 6
        assert stats.output_labels == 6

    def test_skips_images_without_labels(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        image, labels = _synthetic_card_scene()
        cv2.imwrite(str(images_dir / "labelled.png"), image)
        write_label_file(labels_dir / "labelled.txt", labels)
        cv2.imwrite(str(images_dir / "unlabelled.png"), image)

        stats = CardAugmentor(seed=1).augment_dataset(
            images_dir,
            labels_dir,
            tmp_path / "out_img",
            tmp_path / "out_lbl",
            variants_per_image=1,
            include_original=True,
        )

        assert stats.source_images == 1
        assert stats.output_images == 2  # original + 1 variant

    def test_every_output_label_file_is_parseable(self, tmp_path: Path) -> None:
        images_dir, labels_dir = self._write_sample_dataset(tmp_path / "src", count=1)
        out_images = tmp_path / "out" / "images"
        out_labels = tmp_path / "out" / "labels"

        CardAugmentor(seed=0).augment_dataset(
            images_dir,
            labels_dir,
            out_images,
            out_labels,
            variants_per_image=4,
            include_original=True,
        )

        for label_path in out_labels.glob("*.txt"):
            labels = parse_label_file(label_path)
            assert labels, f"{label_path.name} should contain at least one label"
            for label in labels:
                assert 0.0 <= label.cx <= 1.0
                assert 0.0 <= label.w <= 1.0


# ---------------------------------------------------------------------------
# Paths and class names
# ---------------------------------------------------------------------------


class TestTrainingPaths:
    def test_ensure_training_layout_creates_directories(self, tmp_path: Path) -> None:
        paths = ensure_training_layout(tmp_path)
        assert paths.augmented_images.is_dir()
        assert paths.live_dealer_labels.is_dir()
        assert paths.raw.is_dir()

    def test_discover_image_label_pairs(self, tmp_path: Path) -> None:
        images = tmp_path / "images"
        labels = tmp_path / "labels"
        images.mkdir()
        labels.mkdir()
        cv2.imwrite(str(images / "a.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
        write_label_file(
            labels / "a.txt",
            [YoloLabel(class_id=0, cx=0.5, cy=0.5, w=0.2, h=0.2)],
        )
        cv2.imwrite(str(images / "b.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))

        pairs = discover_image_label_pairs(images, labels)
        assert len(pairs) == 1
        assert pairs[0][0].stem == "a"


class TestClassNames:
    def test_build_class_names_count_and_order(self) -> None:
        names = build_class_names()
        assert len(names) == 52
        assert names[0] == "2c"
        assert names[3] == "2s"
        assert names[-1] == "As"


# ---------------------------------------------------------------------------
# Roboflow download (mocked)
# ---------------------------------------------------------------------------


class TestRoboflowDownload:
    def test_resolve_api_key_prefers_explicit(self) -> None:
        with patch.dict("os.environ", {"ROBOFLOW_API_KEY": "env-key"}, clear=False):
            assert resolve_api_key("cli-key") == "cli-key"

    def test_resolve_api_key_from_env(self) -> None:
        with patch.dict("os.environ", {"ROBOFLOW_API_KEY": "env-key"}, clear=True):
            assert resolve_api_key(None) == "env-key"

    def test_download_requires_api_key(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RoboflowDownloadError, match="API key"):
                download_playing_cards_dataset(tmp_path / "raw", api_key=None)

    def test_download_mocked_success(self, tmp_path: Path) -> None:
        destination = tmp_path / "raw"
        mock_dataset = MagicMock()
        mock_dataset.location = str(destination)

        mock_version = MagicMock()
        mock_version.download.return_value = mock_dataset

        mock_project = MagicMock()
        mock_project.version.return_value = mock_version

        mock_workspace = MagicMock()
        mock_workspace.project.return_value = mock_project

        mock_rf = MagicMock()
        mock_rf.workspace.return_value = mock_workspace

        mock_roboflow_module = MagicMock()
        mock_roboflow_module.Roboflow.return_value = mock_rf

        with patch.dict(sys.modules, {"roboflow": mock_roboflow_module}):
            result = download_playing_cards_dataset(
                destination,
                api_key="test-key",
            )

        assert result == destination
        mock_version.download.assert_called_once()

    def test_manual_download_instructions_mentions_url(self) -> None:
        text = manual_download_instructions(TrainingPaths.from_root("/tmp/training"))
        assert "universe.roboflow.com" in text
        assert "/tmp/training/raw" in text


# ---------------------------------------------------------------------------
# cards.yaml template
# ---------------------------------------------------------------------------


class TestCardsYamlTemplate:
    def test_cards_yaml_exists_with_52_classes(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cards_yaml = project_root / "data" / "training" / "cards.yaml"
        assert cards_yaml.is_file()
        content = cards_yaml.read_text(encoding="utf-8")
        assert "nc: 52" in content
        assert "augmented/images" in content
        assert "\n  - As\n" in content