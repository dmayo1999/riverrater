"""
Tests for calibration.py (CalibrationCapture and CalibrationSession).

Covers:
- parse_card_string for valid inputs
- parse_card_string for invalid inputs
- CalibrationSession add / finish flow
- CalibrationCapture.get_roi extracts correct region
"""

from __future__ import annotations

import numpy as np
import pytest

from riverrater.vision.calibration import CalibrationCapture, CalibrationSession
from riverrater.vision.template_engine import TemplateEngine

# ---------------------------------------------------------------------------
# Local Card / Rank / Suit fallbacks (mirrors game/state.py)
# ---------------------------------------------------------------------------
try:
    from riverrater.game.state import Card, Rank, Suit
except ImportError:
    from enum import Enum
    from dataclasses import dataclass

    class Suit(Enum):
        HEARTS = "h"
        DIAMONDS = "d"
        CLUBS = "c"
        SPADES = "s"

    class Rank(Enum):
        TWO = "2"
        THREE = "3"
        FOUR = "4"
        FIVE = "5"
        SIX = "6"
        SEVEN = "7"
        EIGHT = "8"
        NINE = "9"
        TEN = "T"
        JACK = "J"
        QUEEN = "Q"
        KING = "K"
        ACE = "A"

    @dataclass
    class Card:
        rank: Rank
        suit: Suit

        def __str__(self) -> str:
            return f"{self.rank.value}{self.suit.value}"

        def __hash__(self) -> int:
            return hash((self.rank, self.suit))

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, Card):
                return False
            return self.rank == other.rank and self.suit == other.suit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A 200×300 BGR frame with a recognisable colour pattern."""
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    # Red rectangle at (50, 80) of size 40×60.
    frame[50:110, 80:140] = (0, 0, 255)
    return frame


# ---------------------------------------------------------------------------
# Tests: parse_card_string — valid inputs
# ---------------------------------------------------------------------------


class TestParseCardStringValid:
    @pytest.mark.parametrize(
        "card_str, expected_rank, expected_suit",
        [
            ("Ah", Rank.ACE, Suit.HEARTS),
            ("ah", Rank.ACE, Suit.HEARTS),   # lowercase rank should also work
            ("Td", Rank.TEN, Suit.DIAMONDS),
            ("2c", Rank.TWO, Suit.CLUBS),
            ("Ks", Rank.KING, Suit.SPADES),
            ("Jh", Rank.JACK, Suit.HEARTS),
            ("Qd", Rank.QUEEN, Suit.DIAMONDS),
            ("9s", Rank.NINE, Suit.SPADES),
            ("3c", Rank.THREE, Suit.CLUBS),
        ],
    )
    def test_valid_card_strings(
        self, card_str: str, expected_rank: Rank, expected_suit: Suit
    ) -> None:
        card = CalibrationSession.parse_card_string(card_str)
        assert card.rank == expected_rank, f"Rank mismatch for '{card_str}'"
        assert card.suit == expected_suit, f"Suit mismatch for '{card_str}'"

    def test_returns_card_instance(self) -> None:
        card = CalibrationSession.parse_card_string("Ah")
        assert isinstance(card, Card)

    def test_all_ranks_parseable(self) -> None:
        """Every rank character should parse without error for each suit."""
        rank_chars = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        for rank_char in rank_chars:
            card = CalibrationSession.parse_card_string(f"{rank_char}h")
            assert str(card.rank.value).upper() == rank_char.upper()

    def test_all_suits_parseable(self) -> None:
        """Every suit character should parse without error."""
        for suit_char in ["h", "d", "c", "s"]:
            card = CalibrationSession.parse_card_string(f"A{suit_char}")
            assert card.suit.value == suit_char


# ---------------------------------------------------------------------------
# Tests: parse_card_string — invalid inputs
# ---------------------------------------------------------------------------


class TestParseCardStringInvalid:
    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="2-character"):
            CalibrationSession.parse_card_string("")

    def test_single_char_raises(self) -> None:
        with pytest.raises(ValueError):
            CalibrationSession.parse_card_string("A")

    def test_three_char_string_raises(self) -> None:
        with pytest.raises(ValueError):
            CalibrationSession.parse_card_string("Ahx")

    def test_invalid_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            CalibrationSession.parse_card_string("Xh")

    def test_invalid_suit_raises(self) -> None:
        with pytest.raises(ValueError, match="suit"):
            CalibrationSession.parse_card_string("Ax")

    def test_numeric_string_raises(self) -> None:
        with pytest.raises(ValueError):
            CalibrationSession.parse_card_string("12")

    def test_non_string_raises(self) -> None:
        with pytest.raises((ValueError, AttributeError)):
            CalibrationSession.parse_card_string(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: CalibrationCapture.get_roi
# ---------------------------------------------------------------------------


class TestGetRoi:
    def test_extracts_correct_region(self, sample_frame: np.ndarray) -> None:
        cap = CalibrationCapture()
        # The red rectangle in sample_frame is at y=50, x=80, w=60, h=60.
        roi = cap.get_roi(sample_frame, x=80, y=50, w=60, h=60)
        assert roi.shape == (60, 60, 3)
        # All pixels in the ROI should be (0, 0, 255).
        assert np.all(roi == np.array([0, 0, 255], dtype=np.uint8))

    def test_roi_is_copy(self, sample_frame: np.ndarray) -> None:
        cap = CalibrationCapture()
        roi = cap.get_roi(sample_frame, x=80, y=50, w=60, h=60)
        roi[:] = 0  # Modify the ROI.
        # The original frame should not be affected.
        assert not np.all(sample_frame[50:110, 80:140] == 0)

    def test_roi_at_origin(self) -> None:
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 42
        cap = CalibrationCapture()
        roi = cap.get_roi(frame, x=0, y=0, w=10, h=10)
        assert roi.shape == (10, 10, 3)

    def test_zero_width_raises(self, sample_frame: np.ndarray) -> None:
        cap = CalibrationCapture()
        with pytest.raises(ValueError):
            cap.get_roi(sample_frame, x=10, y=10, w=0, h=20)

    def test_zero_height_raises(self, sample_frame: np.ndarray) -> None:
        cap = CalibrationCapture()
        with pytest.raises(ValueError):
            cap.get_roi(sample_frame, x=10, y=10, w=20, h=0)

    def test_out_of_bounds_region_clamped(self, sample_frame: np.ndarray) -> None:
        """An ROI that extends beyond frame boundaries should be clamped and not error."""
        cap = CalibrationCapture()
        # Request a region that starts inside but extends past the right edge.
        roi = cap.get_roi(sample_frame, x=280, y=0, w=100, h=50)
        assert roi.shape[1] == 20  # 300 - 280 = 20 px width
        assert roi.shape[0] == 50

    def test_fully_outside_frame_raises(self, sample_frame: np.ndarray) -> None:
        cap = CalibrationCapture()
        with pytest.raises(ValueError):
            cap.get_roi(sample_frame, x=400, y=400, w=50, h=50)

    def test_start_capture_returns_frame_copy(self, sample_frame: np.ndarray) -> None:
        cap = CalibrationCapture()
        result = cap.start_capture(sample_frame)
        assert result.shape == sample_frame.shape
        # Modifying the result should not alter the original.
        result[:] = 0
        assert not np.all(sample_frame == 0)

    def test_start_capture_empty_frame_raises(self) -> None:
        cap = CalibrationCapture()
        empty = np.array([], dtype=np.uint8)
        with pytest.raises(ValueError):
            cap.start_capture(empty)


# ---------------------------------------------------------------------------
# Tests: CalibrationSession — add / finish flow
# ---------------------------------------------------------------------------


class TestCalibrationSessionFlow:
    def _make_frame_with_card(
        self, color: tuple[int, int, int], pos: tuple[int, int], size: tuple[int, int]
    ) -> np.ndarray:
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        y, x = pos
        h, w = size
        frame[y : y + h, x : x + w] = color
        return frame

    def test_add_single_calibration(self) -> None:
        session = CalibrationSession()
        frame = self._make_frame_with_card((0, 255, 0), (50, 60), (40, 60))
        session.add_calibration("Ah", (60, 50, 60, 40), frame)
        assert len(session._pending) == 1

    def test_add_multiple_calibrations(self) -> None:
        session = CalibrationSession()
        frame = self._make_frame_with_card((100, 150, 200), (10, 20), (30, 40))
        session.add_calibration("Ah", (20, 10, 40, 30), frame)
        session.add_calibration("Ks", (20, 10, 40, 30), frame)
        assert len(session._pending) == 2

    def test_finish_adds_templates_to_engine(self) -> None:
        session = CalibrationSession()
        frame = self._make_frame_with_card((200, 100, 50), (20, 30), (50, 70))
        session.add_calibration("Ah", (30, 20, 70, 50), frame)
        session.add_calibration("Ks", (30, 20, 70, 50), frame)

        engine = TemplateEngine()
        session.finish(engine)

        ah = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        ks = Card(rank=Rank.KING, suit=Suit.SPADES)
        assert ah in engine._templates
        assert ks in engine._templates

    def test_finish_sets_roi_regions_from_unique_bboxes(self) -> None:
        session = CalibrationSession()
        frame = self._make_frame_with_card((200, 100, 50), (20, 30), (50, 70))
        session.add_calibration("Ah", (30, 20, 70, 50), frame)
        session.add_calibration("Ks", (120, 20, 70, 50), frame)

        engine = TemplateEngine()
        session.finish(engine)

        assert engine.roi_regions == [(30, 20, 70, 50), (120, 20, 70, 50)]
        assert engine.uses_roi_scoped_search is True

    def test_finish_clears_pending(self) -> None:
        session = CalibrationSession()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        session.add_calibration("Ah", (0, 0, 50, 50), frame)

        engine = TemplateEngine()
        session.finish(engine)

        assert len(session._pending) == 0

    def test_finish_marks_session_inactive(self) -> None:
        session = CalibrationSession()
        engine = TemplateEngine()
        session.finish(engine)
        assert not session._active

    def test_add_after_finish_raises(self) -> None:
        session = CalibrationSession()
        engine = TemplateEngine()
        session.finish(engine)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            session.add_calibration("Ah", (0, 0, 10, 10), frame)

    def test_finish_after_finish_raises(self) -> None:
        session = CalibrationSession()
        engine = TemplateEngine()
        session.finish(engine)
        with pytest.raises(RuntimeError):
            session.finish(engine)

    def test_cancel_discards_pending(self) -> None:
        session = CalibrationSession()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        session.add_calibration("Ah", (0, 0, 50, 50), frame)
        session.cancel()

        assert len(session._pending) == 0
        assert not session._active

    def test_add_after_cancel_raises(self) -> None:
        session = CalibrationSession()
        session.cancel()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            session.add_calibration("Ah", (0, 0, 10, 10), frame)

    def test_finish_invalid_engine_raises(self) -> None:
        session = CalibrationSession()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        session.add_calibration("Ah", (0, 0, 50, 50), frame)
        with pytest.raises(AttributeError):
            session.finish("not_an_engine")  # type: ignore[arg-type]

    def test_parse_card_string_delegates_correctly(self) -> None:
        session = CalibrationSession()
        card = session.parse_card_string("Td")
        assert card == Card(rank=Rank.TEN, suit=Suit.DIAMONDS)

    def test_empty_session_finish_no_error(self) -> None:
        """Finishing an empty session should succeed without error."""
        session = CalibrationSession()
        engine = TemplateEngine()
        session.finish(engine)  # Should not raise.
        assert len(engine._templates) == 0
