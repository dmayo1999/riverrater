"""
tests/test_settings_dialog.py — Tests for the SettingsDialog.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from riverrater.main import AppConfig
from riverrater.hud.settings_dialog import SettingsDialog


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(
        capture_region=(10, 20, 800, 600),
        hud_position=(150, 200),
        hud_opacity=0.75,
        num_decks=8,
        min_bet=25.0,
        max_bet=1000.0,
        bankroll=10_000.0,
        vision_profile="test_profile",
        detection_confidence=0.65,
        pot_ocr_enabled=False,
        pot_ocr_confidence=0.55,
        pot_roi=(100, 110, 120, 30),
        bet_roi=(200, 210, 80, 25),
        yolo_model_path="/tmp/custom.pt",
        yolo_confidence=0.4,
    )


@pytest.fixture
def dialog(qtbot, config, tmp_path):
    """Create a SettingsDialog backed by a temp config path."""
    config_path = tmp_path / "config.json"
    dlg = SettingsDialog(config, config_path)
    qtbot.addWidget(dlg)
    return dlg, config_path


class TestSettingsDialogInit:
    def test_window_title(self, dialog):
        dlg, _ = dialog
        assert dlg.windowTitle() == "Settings"

    def test_loads_capture_region(self, dialog):
        dlg, _ = dialog
        assert dlg._capture_enabled.isChecked()
        assert dlg._capture_left.value() == 10
        assert dlg._capture_top.value() == 20
        assert dlg._capture_width.value() == 800
        assert dlg._capture_height.value() == 600

    def test_loads_hud_fields(self, dialog):
        dlg, _ = dialog
        assert dlg.hud_opacity == 0.75
        assert dlg._hud_x.value() == 150
        assert dlg._hud_y.value() == 200

    def test_loads_blackjack_fields(self, dialog):
        dlg, _ = dialog
        assert dlg.num_decks == 8
        assert dlg._min_bet.value() == 25.0
        assert dlg._max_bet.value() == 1000.0
        assert dlg._bankroll.value() == 10_000.0

    def test_loads_vision_fields(self, dialog):
        dlg, _ = dialog
        assert dlg.vision_profile == "test_profile"
        assert dlg.detection_confidence == 0.65

    def test_loads_pot_ocr_fields(self, dialog):
        dlg, _ = dialog
        assert dlg.pot_ocr_enabled is False
        assert dlg._pot_ocr_confidence.value() == 0.55
        assert dlg._pot_roi_enabled.isChecked()
        assert dlg._bet_roi_enabled.isChecked()

    def test_loads_yolo_fields(self, dialog):
        dlg, _ = dialog
        assert dlg.yolo_model_path == "/tmp/custom.pt"
        assert dlg._yolo_confidence.value() == 0.4


class TestSettingsDialogDefaults:
    def test_optional_capture_region_disabled(self, qtbot, tmp_path):
        cfg = AppConfig()
        dlg = SettingsDialog(cfg, tmp_path / "config.json")
        qtbot.addWidget(dlg)
        assert not dlg._capture_enabled.isChecked()

    def test_optional_rois_disabled(self, qtbot, tmp_path):
        cfg = AppConfig()
        dlg = SettingsDialog(cfg, tmp_path / "config.json")
        qtbot.addWidget(dlg)
        assert not dlg._pot_roi_enabled.isChecked()
        assert not dlg._bet_roi_enabled.isChecked()


class TestSettingsDialogValidation:
    def test_hud_opacity_range(self, dialog):
        dlg, _ = dialog
        dlg.hud_opacity = 0.1
        assert dlg.hud_opacity == 0.1
        dlg.hud_opacity = 2.0
        assert dlg.hud_opacity <= 1.0

    def test_num_decks_range(self, dialog):
        dlg, _ = dialog
        dlg.num_decks = 0
        assert dlg.num_decks >= 1
        dlg.num_decks = 20
        assert dlg.num_decks <= 8


class TestSettingsDialogCollect:
    def test_collect_config_updates_fields(self, dialog, config):
        dlg, _ = dialog
        dlg.hud_opacity = 0.9
        dlg.num_decks = 2
        dlg.vision_profile = "live_dealer"
        dlg.detection_confidence = 0.7
        dlg.pot_ocr_enabled = True
        dlg.yolo_model_path = ""

        updated = dlg.collect_config(config)
        assert updated.hud_opacity == 0.9
        assert updated.num_decks == 2
        assert updated.vision_profile == "live_dealer"
        assert updated.detection_confidence == 0.7
        assert updated.pot_ocr_enabled is True
        assert updated.yolo_model_path is None

    def test_collect_clears_optional_regions(self, dialog, config):
        dlg, _ = dialog
        dlg._capture_enabled.setChecked(False)
        dlg._pot_roi_enabled.setChecked(False)
        dlg._bet_roi_enabled.setChecked(False)

        updated = dlg.collect_config(config)
        assert updated.capture_region is None
        assert updated.pot_roi is None
        assert updated.bet_roi is None


class TestSettingsDialogSave:
    def test_save_persists_config(self, dialog, qtbot):
        dlg, config_path = dialog
        dlg.hud_opacity = 0.55
        dlg.num_decks = 4
        dlg.vision_profile = "saved_profile"

        with qtbot.waitSignal(dlg.settings_saved, timeout=1000):
            dlg._on_save()

        assert config_path.exists()
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["hud_opacity"] == 0.55
        assert data["num_decks"] == 4
        assert data["vision_profile"] == "saved_profile"

    def test_settings_saved_signal_payload(self, dialog, qtbot):
        dlg, _ = dialog
        dlg.hud_opacity = 0.6

        with qtbot.waitSignal(dlg.settings_saved, timeout=1000) as blocker:
            dlg._on_save()

        saved = blocker.args[0]
        assert isinstance(saved, AppConfig)
        assert saved.hud_opacity == 0.6