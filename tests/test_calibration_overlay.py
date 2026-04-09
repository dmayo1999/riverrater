"""Tests for CalibrationOverlay widget."""

import pytest
import numpy as np

from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from riverrater.hud.calibration_overlay import CalibrationOverlay


@pytest.fixture
def qapp():
    """Ensure a QApplication exists."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeEngine:
    """Minimal stand-in for TemplateEngine used during tests."""

    def __init__(self):
        self.templates = []

    def add_template(self, card, roi):
        self.templates.append((card, roi))

    def save_profile(self, path):
        self.saved_path = path


@pytest.fixture
def dummy_frame():
    """100×100 BGR numpy frame."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def overlay(qapp, dummy_frame):
    """Create a CalibrationOverlay with a dummy frame."""
    engine = _FakeEngine()
    ov = CalibrationOverlay(
        frame=dummy_frame,
        template_engine=engine,
        profile_path="/tmp/test_profile",
    )
    return ov


def test_overlay_creation(overlay):
    """Widget is created successfully."""
    assert overlay is not None


def test_initial_state(overlay):
    """No pending bbox, no entries, Finish disabled, Cancel enabled."""
    assert overlay._pending_bbox is None
    assert len(overlay._entries) == 0
    assert not overlay._finish_btn.isEnabled()
    assert overlay._cancel_btn.isEnabled()


def test_rank_selection(overlay):
    """Clicking a rank button stores the selection."""
    overlay._on_rank_clicked("A")
    assert overlay._selected_rank == "A"


def test_suit_selection(overlay):
    """Clicking a suit button stores the selection."""
    overlay._on_rank_clicked("K")
    overlay._on_suit_clicked("h")
    assert overlay._selected_suit == "h"


def test_confirm_adds_entry(overlay):
    """Setting a pending bbox + rank + suit and confirming adds an entry."""
    overlay._pending_bbox = (10, 10, 20, 20)
    overlay._on_rank_clicked("T")
    overlay._on_suit_clicked("d")
    overlay._on_confirm()
    assert len(overlay._entries) == 1
    assert overlay._entries[0][0] == "Td"


def test_undo_removes_entry(overlay):
    """Adding 2 entries then undoing leaves 1."""
    # Add first entry
    overlay._pending_bbox = (10, 10, 20, 20)
    overlay._on_rank_clicked("A")
    overlay._on_suit_clicked("h")
    overlay._on_confirm()

    # Add second entry
    overlay._pending_bbox = (30, 30, 20, 20)
    overlay._on_rank_clicked("K")
    overlay._on_suit_clicked("d")
    overlay._on_confirm()

    assert len(overlay._entries) == 2

    overlay._on_undo()
    assert len(overlay._entries) == 1
    assert overlay._entries[0][0] == "Ah"


def test_cancel_emits_signal(overlay, qtbot):
    """Cancel button emits calibration_cancelled signal."""
    with qtbot.waitSignal(overlay.calibration_cancelled, timeout=1000):
        overlay._on_cancel()


def test_finish_emits_signal(overlay, qtbot):
    """Finish emits calibration_finished when entries exist."""
    overlay._pending_bbox = (5, 5, 15, 15)
    overlay._on_rank_clicked("2")
    overlay._on_suit_clicked("c")
    overlay._on_confirm()

    with qtbot.waitSignal(overlay.calibration_finished, timeout=1000):
        overlay._on_finish()


def test_escape_cancels(overlay, qtbot):
    """Pressing Escape emits calibration_cancelled."""
    overlay.show()
    with qtbot.waitSignal(overlay.calibration_cancelled, timeout=1000):
        QTest.keyPress(overlay, Qt.Key.Key_Escape)


def test_coordinate_scaling(qapp):
    """Verify display-to-frame coordinate mapping with a known scale factor."""
    # Create a 200×100 frame
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    engine = _FakeEngine()
    ov = CalibrationOverlay(
        frame=frame,
        template_engine=engine,
        profile_path="/tmp/test_profile",
    )

    # Manually set frame_rect to simulate a display area at 2x scale
    ov._frame_rect = QRect(0, 0, 400, 200)

    # A display rect at (100, 50, 200, 100) should map to (50, 25, 100, 50) in frame
    display_rect = QRect(100, 50, 200, 100)
    fx, fy, fw, fh = ov._display_to_frame_bbox(display_rect)

    assert fx == 50
    assert fy == 25
    assert fw == 100
    assert fh == 50
