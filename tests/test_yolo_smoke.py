"""
Smoke tests for YOLO training scaffold and YOLOEngine integration.

Weights are optional: tests that need ``best.pt`` skip when absent.
Ultralytics is mocked for code-path coverage without GPU or trained weights.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from riverrater.game.state import Card, Rank, Suit
from riverrater.training.augmentation import build_class_names
from riverrater.training.yolo_train import (
    MAP50_TARGET,
    ClassAlignmentError,
    TrainingDataError,
    class_map_to_names,
    colab_training_instructions,
    default_weights_path,
    parse_cards_yaml_names,
    resolve_training_config,
    run_training,
    validate_class_alignment,
    validate_training_data,
)
from riverrater.vision.yolo_engine import YOLOEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARDS_YAML = PROJECT_ROOT / "data" / "training" / "cards.yaml"
WEIGHTS_PATH = default_weights_path()
WEIGHTS_AVAILABLE = WEIGHTS_PATH.is_file()
MODELS_README = PROJECT_ROOT / "models" / "yolov8n_cards" / "README.md"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_yolo.py"


def _load_train_script():
    spec = importlib.util.spec_from_file_location("train_yolo", TRAIN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


requires_weights = pytest.mark.skipif(
    not WEIGHTS_AVAILABLE,
    reason=f"YOLO weights not found at {WEIGHTS_PATH}",
)


# ---------------------------------------------------------------------------
# CLASS_MAP ↔ cards.yaml alignment
# ---------------------------------------------------------------------------


class TestClassMapAlignment:
    def test_cards_yaml_has_52_classes(self) -> None:
        names = parse_cards_yaml_names(CARDS_YAML)
        assert len(names) == 52

    def test_cards_yaml_matches_yolo_engine_class_map(self) -> None:
        names = validate_class_alignment(CARDS_YAML, YOLOEngine.CLASS_MAP)
        assert names[0] == "2c"
        assert names[-1] == "As"

    def test_cards_yaml_matches_build_class_names(self) -> None:
        yaml_names = parse_cards_yaml_names(CARDS_YAML)
        assert yaml_names == build_class_names()

    def test_class_map_to_names_order(self) -> None:
        names = class_map_to_names(YOLOEngine.CLASS_MAP)
        assert names == build_class_names()

    def test_mismatch_raises(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(
            "nc: 52\nnames:\n  - As\n  - 2c\n",
            encoding="utf-8",
        )
        with pytest.raises(ClassAlignmentError):
            validate_class_alignment(bad_yaml, YOLOEngine.CLASS_MAP)


# ---------------------------------------------------------------------------
# Training scaffold
# ---------------------------------------------------------------------------


class TestTrainingScaffold:
    def test_default_weights_path(self) -> None:
        assert WEIGHTS_PATH == PROJECT_ROOT / "models" / "yolov8n_cards" / "best.pt"

    def test_models_readme_documents_export_and_map_target(self) -> None:
        text = MODELS_README.read_text(encoding="utf-8")
        assert "best.pt" in text
        assert "85%" in text or "0.85" in text
        assert "CLASS_MAP" in text
        assert "cards.yaml" in text

    def test_map50_target_constant(self) -> None:
        assert MAP50_TARGET == pytest.approx(0.85)

    def test_colab_instructions_reference_cards_yaml_and_export(self) -> None:
        text = colab_training_instructions()
        assert "cards.yaml" in text
        assert "models/yolov8n_cards/best.pt" in text
        assert "Colab" in text or "colab" in text.lower()

    def test_resolve_training_config_defaults(self) -> None:
        config = resolve_training_config()
        assert config.data_yaml == CARDS_YAML.resolve()
        assert config.export_path == WEIGHTS_PATH.resolve()
        assert config.epochs == 100
        assert config.imgsz == 640

    def test_validate_training_data_requires_images(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "cards.yaml"
        data_yaml.write_text(
            "path: .\ntrain: empty/images\nval: empty/valid\nnc: 52\nnames:\n  - 2c\n",
            encoding="utf-8",
        )
        (tmp_path / "empty" / "images").mkdir(parents=True)
        with pytest.raises(TrainingDataError, match="No training images"):
            validate_training_data(data_yaml)

    def test_validate_training_data_counts_images(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "cards.yaml"
        data_yaml.write_text(
            "path: .\ntrain: train/images\nval: train/images\nnc: 52\nnames:\n  - 2c\n",
            encoding="utf-8",
        )
        images = tmp_path / "train" / "images"
        images.mkdir(parents=True)
        (images / "card_001.jpg").write_bytes(b"\xff\xd8\xff")
        summary = validate_training_data(data_yaml)
        assert summary["train_images"] == 1
        assert summary["ready"] is True

    def test_train_script_dry_run(self) -> None:
        train_yolo = _load_train_script()
        rc = train_yolo.main(["--dry-run", "--data", str(CARDS_YAML)])
        assert rc == 0

    def test_train_script_colab_instructions(self, capsys: pytest.CaptureFixture[str]) -> None:
        train_yolo = _load_train_script()
        rc = train_yolo.main(["--colab-instructions"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "Colab" in captured.out or "colab" in captured.out.lower()


# ---------------------------------------------------------------------------
# YOLOEngine — stub mode (no weights)
# ---------------------------------------------------------------------------


class TestYOLOEngineStub:
    def test_unavailable_without_weights_path(self) -> None:
        engine = YOLOEngine("/nonexistent/best.pt")
        assert engine.is_available is False

    def test_detect_returns_empty_in_stub_mode(self) -> None:
        engine = YOLOEngine(None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert engine.detect_cards(frame) == []

    def test_none_model_path_is_stub(self) -> None:
        engine = YOLOEngine()
        assert engine.is_available is False


# ---------------------------------------------------------------------------
# YOLOEngine — mocked ultralytics
# ---------------------------------------------------------------------------


def _mock_yolo_inference(as_class_id: int = 0, confidence: float = 0.92) -> MagicMock:
    """Build a fake ultralytics result with one detection."""
    box = MagicMock()
    box.cls = [MagicMock(item=lambda: float(as_class_id))]
    box.conf = [MagicMock(item=lambda: confidence)]
    box.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 60.0, 90.0])]

    result = MagicMock()
    result.boxes = [box]

    model = MagicMock()
    model.return_value = [result]
    return model


class TestYOLOEngineMocked:
    @patch("os.path.isfile", return_value=True)
    def test_loads_model_when_file_exists(self, _isfile: MagicMock) -> None:
        fake_yolo_cls = MagicMock()
        fake_model = MagicMock()
        fake_yolo_cls.return_value = fake_model

        with patch.dict(
            "sys.modules",
            {"ultralytics": MagicMock(YOLO=fake_yolo_cls)},
        ):
            engine = YOLOEngine("/fake/best.pt")

        assert engine.is_available is True
        fake_yolo_cls.assert_called_once_with("/fake/best.pt")

    @patch("os.path.isfile", return_value=True)
    def test_detect_cards_maps_class_id(self, _isfile: MagicMock) -> None:
        ace_spades_id = 51
        fake_yolo_cls = MagicMock(return_value=_mock_yolo_inference(ace_spades_id))

        with patch.dict(
            "sys.modules",
            {"ultralytics": MagicMock(YOLO=fake_yolo_cls)},
        ):
            engine = YOLOEngine("/fake/best.pt")
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            detections = engine.detect_cards(frame, confidence=0.5)

        assert len(detections) == 1
        card, bbox, score = detections[0]
        assert card == Card(rank=Rank.ACE, suit=Suit.SPADES)
        assert bbox == (10, 20, 50, 70)
        assert score == pytest.approx(0.92)

    @patch("os.path.isfile", return_value=True)
    def test_detect_skips_unknown_class_id(self, _isfile: MagicMock) -> None:
        fake_yolo_cls = MagicMock(return_value=_mock_yolo_inference(99))

        with patch.dict(
            "sys.modules",
            {"ultralytics": MagicMock(YOLO=fake_yolo_cls)},
        ):
            engine = YOLOEngine("/fake/best.pt")
            detections = engine.detect_cards(np.zeros((50, 50, 3), dtype=np.uint8))

        assert detections == []

    @patch("os.path.isfile", return_value=True)
    def test_detect_survives_inference_error(self, _isfile: MagicMock) -> None:
        broken_model = MagicMock(side_effect=RuntimeError("inference failed"))
        fake_yolo_cls = MagicMock(return_value=broken_model)

        with patch.dict(
            "sys.modules",
            {"ultralytics": MagicMock(YOLO=fake_yolo_cls)},
        ):
            engine = YOLOEngine("/fake/best.pt")
            detections = engine.detect_cards(np.zeros((50, 50, 3), dtype=np.uint8))

        assert detections == []


class TestRunTrainingMocked:
    def test_run_training_exports_weights(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        images = data_root / "train" / "images"
        images.mkdir(parents=True)
        (images / "sample.png").write_bytes(b"\x89PNG")

        data_yaml = data_root / "cards.yaml"
        data_yaml.write_text(
            "path: .\ntrain: train/images\nval: train/images\nnc: 52\n"
            "names:\n  - 2c\n",
            encoding="utf-8",
        )

        project_dir = tmp_path / "runs" / "detect"
        weights_dir = project_dir / "train" / "weights"
        weights_dir.mkdir(parents=True)
        best_src = weights_dir / "best.pt"
        best_src.write_bytes(b"fake-weights")

        export_path = tmp_path / "models" / "yolov8n_cards" / "best.pt"
        config = resolve_training_config(
            data_yaml=data_yaml,
            export_path=export_path,
            epochs=1,
            project=str(project_dir),
            name="train",
        )

        fake_results = SimpleNamespace(metrics={"metrics/mAP50(B)": 0.9})
        fake_model = MagicMock()
        fake_model.train.return_value = fake_results
        fake_yolo_cls = MagicMock(return_value=fake_model)

        with (
            patch("riverrater.training.yolo_train.copy2") as mock_copy,
            patch.dict(
                "sys.modules",
                {"ultralytics": MagicMock(YOLO=fake_yolo_cls)},
            ),
        ):
            results = run_training(config)

        fake_model.train.assert_called_once()
        assert results is fake_results
        mock_copy.assert_called_once_with(best_src, export_path)


# ---------------------------------------------------------------------------
# Optional integration — real weights (skipped when absent)
# ---------------------------------------------------------------------------


class TestYOLOEngineWithWeights:
    @requires_weights
    def test_weights_file_loads_when_ultralytics_present(self) -> None:
        pytest.importorskip("ultralytics")
        engine = YOLOEngine(str(WEIGHTS_PATH))
        assert engine.is_available is True