"""Tests for CalibrationOverlay widget."""

import pytest
import numpy as np

from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from riverrater.game.state import Card, Rank, Suit
from riverrater.hud.calibration_overlay import (
    ALL_CARD_STRINGS,
    CalibrationOverlay,
    format_guided_prompt,
    get_captured_cards,
    next_missing_card,
)


@pytest.fixture
def qapp():
    """Ensure a QApplication exists."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeEngine:
    """Minimal stand-in for TemplateEngine used during tests."""

    def __init__(self, existing_templates=None):
        self.templates = []
        self._templates = existing_templates or {}

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


# ---------------------------------------------------------------------------
# P1 helpers
# ---------------------------------------------------------------------------


def test_all_card_strings_has_52_cards():
    """Canonical deck ordering covers every rank/suit pair."""
    assert len(ALL_CARD_STRINGS) == 52
    assert len(set(ALL_CARD_STRINGS)) == 52


def test_next_missing_card_empty_profile():
    """First missing card is the first in canonical order."""
    assert next_missing_card(set()) == ALL_CARD_STRINGS[0]


def test_next_missing_card_skips_captured():
    """Guided helper skips cards that already have templates."""
    captured = {"2h", "3h", "4h"}
    assert next_missing_card(captured) == "5h"


def test_format_guided_prompt():
    """Guided prompt includes rank, suit symbol, and hint text."""
    prompt = format_guided_prompt("3c")
    assert "3" in prompt
    assert "\u2663" in prompt
    assert "no template yet" in prompt


def test_get_captured_cards_merges_engine_and_entries():
    """Captured set includes both engine templates and session entries."""
    ah = Card(rank=Rank.ACE, suit=Suit.HEARTS)
    engine = _FakeEngine(existing_templates={ah: [np.zeros((5, 5), dtype=np.uint8)]})
    entries = [("Kd", (0, 0, 10, 10))]
    captured = get_captured_cards(engine, entries)
    assert captured == {"Ah", "Kd"}


# ---------------------------------------------------------------------------
# P1 widget features
# ---------------------------------------------------------------------------


def test_roi_preview_2x_zoom(overlay):
    """Pending bbox preview pixmap is scaled to 2× the crop dimensions."""
    overlay._pending_bbox = (10, 10, 20, 20)
    overlay._update_roi_preview()
    pm = overlay._roi_preview.pixmap()
    assert pm is not None
    assert pm.width() == 40
    assert pm.height() == 40


def test_progress_counter_includes_engine_templates(qapp, dummy_frame):
    """Counter reflects templates already loaded in the engine."""
    ah = Card(rank=Rank.ACE, suit=Suit.HEARTS)
    engine = _FakeEngine(existing_templates={ah: [np.zeros((5, 5), dtype=np.uint8)]})
    ov = CalibrationOverlay(
        frame=dummy_frame,
        template_engine=engine,
        profile_path="/tmp/test_profile",
    )
    assert ov._counter_label.text() == "1/52 cards with templates"


def test_progress_grid_highlights_captured_card(overlay):
    """Grid cell turns green after a card is confirmed."""
    overlay._pending_bbox = (10, 10, 20, 20)
    overlay._on_rank_clicked("A")
    overlay._on_suit_clicked("h")
    overlay._on_confirm()

    ah_cell = overlay._grid_cells["Ah"]
    assert "150,83" in ah_cell.styleSheet()


def test_guided_prompt_on_init(overlay):
    """Idle overlay suggests the first missing card."""
    assert format_guided_prompt(ALL_CARD_STRINGS[0]) in overlay._instruction_label.text()


def test_guided_prompt_advances_after_confirm(overlay):
    """After capturing first card, prompt moves to the next missing card."""
    overlay._pending_bbox = (10, 10, 20, 20)
    overlay._on_rank_clicked("2")
    overlay._on_suit_clicked("h")
    overlay._on_confirm()

    expected = format_guided_prompt("3h")
    assert expected in overlay._instruction_label.text()


def test_undo_updates_progress_counter(overlay):
    """Undo decrements the progress counter."""
    overlay._pending_bbox = (10, 10, 20, 20)
    overlay._on_rank_clicked("A")
    overlay._on_suit_clicked("h")
    overlay._on_confirm()
    assert overlay._counter_label.text() == "1/52 cards with templates"

    overlay._on_undo()
    assert overlay._counter_label.text() == "0/52 cards with templates"


def test_drag_repositions_entry(qapp):
    """Dragging a confirmed bbox moves it within the frame."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    engine = _FakeEngine()
    ov = CalibrationOverlay(
        frame=frame,
        template_engine=engine,
        profile_path="/tmp/test_profile",
    )
    ov.show()
    ov._frame_rect = QRect(0, 0, 100, 100)
    ov._entries = [("Ah", (10, 10, 20, 20))]

    QTest.mousePress(ov, Qt.MouseButton.LeftButton, pos=QPoint(15, 15))
    QTest.mouseMove(ov, QPoint(35, 35))
    QTest.mouseRelease(ov, Qt.MouseButton.LeftButton, pos=QPoint(35, 35))

    x, y, w, h = ov._entries[0][1]
    assert x == 30
    assert y == 30
    assert w == 20
    assert h == 20