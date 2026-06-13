"""Tests for confidence display elements in PokerView."""

import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from riverrater.game.state import DetectionMeta, PokerAction, PokerResult
from riverrater.hud.poker_view import PokerView, _CLR_GREEN, _CLR_YELLOW, _CLR_RED


@pytest.fixture
def qapp():
    """Ensure a QApplication exists (re-use the one from conftest env setup)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def poker_view(qapp, qtbot):
    """Create a fresh PokerView for each test."""
    view = PokerView()
    qtbot.addWidget(view)
    return view


def _make_result(detection_meta=None, win_pct=0.5, required_equity=0.3):
    """Helper to create a PokerResult with some data so it passes the has_data check."""
    return PokerResult(
        win_pct=win_pct,
        tie_pct=0.05,
        required_equity=required_equity,
        actual_equity=win_pct + 0.025,
        ev_call=10.0,
        ev_fold=0.0,
        recommended_action=PokerAction.CALL,
        detection_meta=detection_meta,
    )


def test_confidence_dot_green(poker_view):
    """Overall confidence >= 0.85 → green dot."""
    meta = DetectionMeta(
        card_confidences={"Ah": 0.90, "Kd": 0.92},
        overall_confidence=0.91,
    )
    poker_view.update(_make_result(detection_meta=meta))
    style = poker_view._confidence_dot.styleSheet()
    assert _CLR_GREEN in style
    assert not poker_view._confidence_dot.isHidden()


def test_confidence_dot_yellow(poker_view):
    """Overall confidence between 0.7 and 0.85 → yellow dot."""
    meta = DetectionMeta(
        card_confidences={"Ah": 0.75},
        overall_confidence=0.75,
    )
    poker_view.update(_make_result(detection_meta=meta))
    style = poker_view._confidence_dot.styleSheet()
    assert _CLR_YELLOW in style


def test_confidence_dot_red(poker_view):
    """Overall confidence < 0.7 → red dot."""
    meta = DetectionMeta(
        card_confidences={"Ah": 0.60},
        overall_confidence=0.60,
    )
    poker_view.update(_make_result(detection_meta=meta))
    style = poker_view._confidence_dot.styleSheet()
    assert _CLR_RED in style


def test_card_labels_displayed(poker_view):
    """Card detection row shows both card strings."""
    meta = DetectionMeta(
        card_confidences={"Ah": 0.94, "Kd": 0.88},
        overall_confidence=0.91,
    )
    poker_view.update(_make_result(detection_meta=meta))
    text = poker_view._card_detection_label.text()
    assert "Ah" in text
    assert "Kd" in text
    assert not poker_view._card_detection_label.isHidden()


def test_manual_mode_display(poker_view):
    """When is_manual=True, the card row shows 'MANUAL'."""
    meta = DetectionMeta.manual()
    poker_view.update(_make_result(detection_meta=meta))
    text = poker_view._card_detection_label.text()
    assert "MANUAL" in text
    assert not poker_view._card_detection_label.isHidden()


def test_no_detection_meta_hides_row(poker_view):
    """When detection_meta is None, confidence elements are hidden."""
    poker_view.update(_make_result(detection_meta=None))
    assert poker_view._card_detection_label.isHidden()
    assert poker_view._confidence_dot.isHidden()
    assert poker_view._warning_label.isHidden()


def test_low_confidence_warning_shown(poker_view):
    """Warning banner visible when any card < 0.7."""
    meta = DetectionMeta(
        card_confidences={"Ah": 0.90, "Kd": 0.65},
        overall_confidence=0.775,
    )
    poker_view.update(_make_result(detection_meta=meta))
    assert not poker_view._warning_label.isHidden()


def test_low_confidence_warning_hidden(poker_view):
    """Warning banner hidden when all cards >= 0.7."""
    meta = DetectionMeta(
        card_confidences={"Ah": 0.90, "Kd": 0.85},
        overall_confidence=0.875,
    )
    poker_view.update(_make_result(detection_meta=meta))
    assert poker_view._warning_label.isHidden()


def test_degradation_banner_shown_when_severe(poker_view):
    """Degradation banner visible when overall confidence < 0.5."""
    meta = DetectionMeta(
        card_confidences={"Ah": 0.40, "Kd": 0.45},
        overall_confidence=0.425,
    )
    poker_view.update(_make_result(detection_meta=meta))
    assert not poker_view._degradation_widget.isHidden()


def test_degradation_banner_hidden_when_ok(poker_view):
    """Degradation banner hidden when confidence is acceptable."""
    meta = DetectionMeta(
        card_confidences={"Ah": 0.90, "Kd": 0.88},
        overall_confidence=0.89,
    )
    poker_view.update(_make_result(detection_meta=meta))
    assert poker_view._degradation_widget.isHidden()


def test_degradation_fix_emits_signal(poker_view, qtbot):
    """Fix button emits degradation_fix_requested with action."""
    received: list[str] = []
    poker_view.degradation_fix_requested.connect(received.append)

    meta = DetectionMeta(
        card_confidences={"Ah": 0.40},
        overall_confidence=0.40,
    )
    poker_view.update(_make_result(detection_meta=meta))
    qtbot.mouseClick(poker_view._degradation_fix_btn, Qt.MouseButton.LeftButton)

    assert received == ["calibrate"]
