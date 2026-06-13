"""Tests for opponent count stepper UX (step-13)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import Qt

from riverrater.game.state import PokerResult
from riverrater.hud.poker_input import OpponentStepper, PokerInputDialog
from riverrater.hud.poker_view import PokerView
from riverrater.main import AppConfig, GameController


@pytest.fixture
def stepper(qtbot):
    widget = OpponentStepper(value=3)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def poker_view(qtbot):
    view = PokerView()
    qtbot.addWidget(view)
    return view


class TestOpponentStepper:
    def test_default_value_is_one(self, qtbot):
        widget = OpponentStepper()
        qtbot.addWidget(widget)
        assert widget.value == 1

    def test_clamps_below_minimum(self, stepper):
        stepper.set_value(0)
        assert stepper.value == 1

    def test_clamps_above_maximum(self, stepper):
        stepper.set_value(20)
        assert stepper.value == 9

    def test_increment_emits_signal(self, stepper, qtbot):
        received: list[int] = []
        stepper.value_changed.connect(received.append)
        qtbot.mouseClick(stepper._plus_btn, Qt.MouseButton.LeftButton)
        assert received == [4]
        assert stepper.value == 4

    def test_decrement_emits_signal(self, stepper, qtbot):
        received: list[int] = []
        stepper.value_changed.connect(received.append)
        qtbot.mouseClick(stepper._minus_btn, Qt.MouseButton.LeftButton)
        assert received == [2]
        assert stepper.value == 2

    def test_plus_disabled_at_max(self, qtbot):
        widget = OpponentStepper(value=9)
        qtbot.addWidget(widget)
        assert not widget._plus_btn.isEnabled()

    def test_minus_disabled_at_min(self, qtbot):
        widget = OpponentStepper(value=1)
        qtbot.addWidget(widget)
        assert not widget._minus_btn.isEnabled()


class TestPokerInputOpponentStepper:
    def test_dialog_uses_stepper_not_spinbox(self, qtbot):
        dialog = PokerInputDialog()
        qtbot.addWidget(dialog)
        assert isinstance(dialog._opp_stepper, OpponentStepper)
        assert dialog.num_opponents == 1

    def test_dialog_accepts_initial_opponents(self, qtbot):
        dialog = PokerInputDialog(num_opponents=5)
        qtbot.addWidget(dialog)
        assert dialog.num_opponents == 5

    def test_num_opponents_setters_clamp(self, qtbot):
        dialog = PokerInputDialog()
        qtbot.addWidget(dialog)
        dialog.num_opponents = 0
        assert dialog.num_opponents == 1
        dialog.num_opponents = 20
        assert dialog.num_opponents == 9


class TestPokerViewOpponentDisplay:
    def test_title_shows_opponent_count(self, poker_view):
        poker_view.set_num_opponents(4)
        assert "4" in poker_view._opp_count_label.text()

    def test_stepper_change_emits_signal(self, poker_view, qtbot):
        received: list[int] = []
        poker_view.num_opponents_changed.connect(received.append)
        poker_view._opp_stepper.set_value(2, emit=True)
        assert received == [2]

    def test_set_num_opponents_syncs_stepper(self, poker_view):
        poker_view.set_num_opponents(7)
        assert poker_view.num_opponents == 7
        assert poker_view._opp_stepper.value == 7


class TestAppConfigNumOpponents:
    def test_default_num_opponents(self):
        assert AppConfig().num_opponents == 1

    def test_roundtrip_save_load(self, tmp_path: Path):
        path = tmp_path / "config.json"
        cfg = AppConfig(num_opponents=6)
        cfg.save(path)
        loaded = AppConfig.load(path)
        assert loaded.num_opponents == 6

    def test_missing_key_defaults_to_one(self, tmp_path: Path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"hud_opacity": 0.9}), encoding="utf-8")
        assert AppConfig.load(path).num_opponents == 1


class TestGameControllerNumOpponents:
    @pytest.fixture
    def controller(self) -> GameController:
        config = AppConfig(num_opponents=4)
        return GameController(config, MagicMock(), MagicMock(), MagicMock())

    def test_init_uses_config_num_opponents(self, controller: GameController):
        assert controller.poker_state.num_opponents == 4

    def test_set_num_opponents_updates_state(self, controller: GameController):
        controller.set_num_opponents(3)
        assert controller.poker_state.num_opponents == 3
        assert controller.config.num_opponents == 3

    def test_set_num_opponents_invalidates_cache(self, controller: GameController):
        with patch("riverrater.main.analyze_poker", return_value=PokerResult()) as mock_analyze:
            controller._tick_poker()
            controller.set_num_opponents(2)
            controller._tick_poker()
            assert mock_analyze.call_count == 2

    def test_reset_hand_preserves_config_opponents(self, controller: GameController):
        controller.set_num_opponents(5)
        controller.reset_hand()
        assert controller.poker_state.num_opponents == 5
