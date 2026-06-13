"""Tests for PotOCR digit detection and GameController integration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from riverrater.main import AppConfig, GameController
from riverrater.vision.pot_ocr import (
    PotOCR,
    load_profile_rois,
    render_amount_image,
    resolve_pot_rois,
)


def _embed_roi(
    amount: str,
    roi: tuple[int, int, int, int],
    frame_size: tuple[int, int] = (320, 160),
) -> np.ndarray:
    """Place a synthetic amount label at *roi* inside a blank frame."""
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    x, y, w, h = roi
    label = render_amount_image(amount, size=(w, h))
    frame[y : y + h, x : x + w] = label
    return frame


@pytest.fixture
def pot_roi() -> tuple[int, int, int, int]:
    return (20, 30, 180, 48)


@pytest.fixture
def bet_roi() -> tuple[int, int, int, int]:
    return (20, 100, 120, 48)


@pytest.fixture
def pot_ocr(pot_roi: tuple[int, int, int, int], bet_roi: tuple[int, int, int, int]) -> PotOCR:
    return PotOCR(pot_roi=pot_roi, bet_roi=bet_roi, confidence_threshold=0.55)


class TestPotOCRParsing:
    def test_read_pot_from_synthetic_image(self, pot_ocr: PotOCR, pot_roi: tuple[int, int, int, int]) -> None:
        frame = _embed_roi("150.50", pot_roi)
        result = pot_ocr.read_pot(frame)

        assert result is not None
        assert abs(result.value - 150.50) < 0.01
        assert result.confidence >= 0.55

    def test_read_bet_from_synthetic_image(self, pot_ocr: PotOCR, bet_roi: tuple[int, int, int, int]) -> None:
        frame = _embed_roi("25", bet_roi)
        result = pot_ocr.read_bet(frame)

        assert result is not None
        assert abs(result.value - 25.0) < 0.01
        assert result.confidence >= 0.55

    def test_read_values_parses_both_regions(
        self,
        pot_ocr: PotOCR,
        pot_roi: tuple[int, int, int, int],
        bet_roi: tuple[int, int, int, int],
    ) -> None:
        frame = np.zeros((160, 320, 3), dtype=np.uint8)
        frame[pot_roi[1] : pot_roi[1] + pot_roi[3], pot_roi[0] : pot_roi[0] + pot_roi[2]] = render_amount_image(
            "200.00",
            size=(pot_roi[2], pot_roi[3]),
        )
        frame[bet_roi[1] : bet_roi[1] + bet_roi[3], bet_roi[0] : bet_roi[0] + bet_roi[2]] = render_amount_image(
            "50",
            size=(bet_roi[2], bet_roi[3]),
        )

        pot_result, bet_result = pot_ocr.read_values(frame)

        assert pot_result is not None
        assert bet_result is not None
        assert abs(pot_result.value - 200.0) < 0.01
        assert abs(bet_result.value - 50.0) < 0.01

    def test_missing_roi_returns_none(self) -> None:
        ocr = PotOCR(pot_roi=None, bet_roi=None)
        frame = _embed_roi("100", (0, 0, 120, 48))

        assert ocr.read_pot(frame) is None
        assert ocr.read_bet(frame) is None

    def test_low_confidence_threshold_rejects_parse(self, pot_roi: tuple[int, int, int, int]) -> None:
        ocr = PotOCR(pot_roi=pot_roi, confidence_threshold=0.99)
        frame = _embed_roi("75.25", pot_roi)

        assert ocr.read_pot(frame) is None

    def test_unconfigured_single_roi(self, pot_roi: tuple[int, int, int, int]) -> None:
        ocr = PotOCR(pot_roi=pot_roi, bet_roi=None)
        frame = _embed_roi("42", pot_roi)

        assert ocr.read_pot(frame) is not None
        assert ocr.read_bet(frame) is None


class TestPotOCRConfig:
    def test_load_profile_rois(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            regions = {
                "card_slots": [[10, 20, 40, 60]],
                "pot_roi": [20, 30, 180, 48],
                "bet_roi": [20, 100, 120, 48],
            }
            regions_path = Path(tmpdir) / "regions.json"
            regions_path.write_text(json.dumps(regions), encoding="utf-8")

            pot_roi, bet_roi = load_profile_rois(tmpdir)

        assert pot_roi == (20, 30, 180, 48)
        assert bet_roi == (20, 100, 120, 48)

    def test_resolve_pot_rois_prefers_app_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            regions = {
                "pot_roi": [1, 2, 3, 4],
                "bet_roi": [5, 6, 7, 8],
            }
            (Path(tmpdir) / "regions.json").write_text(json.dumps(regions), encoding="utf-8")

            pot_roi, bet_roi = resolve_pot_rois(
                config_pot_roi=(10, 20, 30, 40),
                config_bet_roi=None,
                profile_path=tmpdir,
            )

        assert pot_roi == (10, 20, 30, 40)
        assert bet_roi == (5, 6, 7, 8)

    def test_app_config_loads_roi_fields(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(
                {
                    "pot_roi": [20, 30, 180, 48],
                    "bet_roi": [20, 100, 120, 48],
                    "pot_ocr_confidence": 0.7,
                },
                fh,
            )
            config_path = fh.name

        config = AppConfig.load(config_path)
        assert config.pot_roi == (20, 30, 180, 48)
        assert config.bet_roi == (20, 100, 120, 48)
        assert config.pot_ocr_confidence == 0.7


@pytest.fixture
def controller_with_ocr(
    pot_ocr: PotOCR,
    pot_roi: tuple[int, int, int, int],
    bet_roi: tuple[int, int, int, int],
) -> GameController:
    config = AppConfig(pot_ocr_enabled=True)
    capture = MagicMock()
    frame = np.zeros((160, 320, 3), dtype=np.uint8)
    frame[pot_roi[1] : pot_roi[1] + pot_roi[3], pot_roi[0] : pot_roi[0] + pot_roi[2]] = render_amount_image(
        "300.00",
        size=(pot_roi[2], pot_roi[3]),
    )
    frame[bet_roi[1] : bet_roi[1] + bet_roi[3], bet_roi[0] : bet_roi[0] + bet_roi[2]] = render_amount_image(
        "75",
        size=(bet_roi[2], bet_roi[3]),
    )
    capture.get_latest_frame.return_value = frame

    template_engine = MagicMock()
    from riverrater.game.state import Card, Rank, Suit

    card_ah = Card(rank=Rank.ACE, suit=Suit.HEARTS)
    card_kd = Card(rank=Rank.KING, suit=Suit.DIAMONDS)
    template_engine.detect_cards.return_value = [
        (card_ah, (0, 0, 10, 10), 0.95),
        (card_kd, (20, 0, 10, 10), 0.92),
    ]

    return GameController(
        config,
        capture,
        template_engine,
        MagicMock(),
        pot_ocr=pot_ocr,
    )


class TestGameControllerPotOCR:
    def test_tick_poker_updates_state_from_ocr(self, controller_with_ocr: GameController) -> None:
        controller_with_ocr._frame_count = 0
        controller_with_ocr._tick_poker()

        assert abs(controller_with_ocr.poker_state.pot_size - 300.0) < 0.01
        assert abs(controller_with_ocr.poker_state.bet_to_call - 75.0) < 0.01

    def test_manual_set_poker_values_overrides_ocr(self, controller_with_ocr: GameController) -> None:
        controller_with_ocr._frame_count = 0
        controller_with_ocr._tick_poker()
        controller_with_ocr.set_poker_values(pot_size=10.0, bet_to_call=5.0, num_opponents=1)

        controller_with_ocr._frame_count = 3
        controller_with_ocr._tick_poker()

        assert controller_with_ocr.poker_state.pot_size == 10.0
        assert controller_with_ocr.poker_state.bet_to_call == 5.0

    def test_reset_hand_re_enables_ocr(self, controller_with_ocr: GameController) -> None:
        controller_with_ocr.set_poker_values(pot_size=10.0, bet_to_call=5.0, num_opponents=1)
        controller_with_ocr.reset_hand()

        controller_with_ocr._frame_count = 0
        controller_with_ocr._tick_poker()

        assert abs(controller_with_ocr.poker_state.pot_size - 300.0) < 0.01
        assert abs(controller_with_ocr.poker_state.bet_to_call - 75.0) < 0.01

    def test_ocr_disabled_by_config(self, pot_ocr: PotOCR, pot_roi: tuple[int, int, int, int], bet_roi: tuple[int, int, int, int]) -> None:
        config = AppConfig(pot_ocr_enabled=False)
        capture = MagicMock()
        frame = np.zeros((160, 320, 3), dtype=np.uint8)
        frame[pot_roi[1] : pot_roi[1] + pot_roi[3], pot_roi[0] : pot_roi[0] + pot_roi[2]] = render_amount_image(
            "300.00",
            size=(pot_roi[2], pot_roi[3]),
        )
        capture.get_latest_frame.return_value = frame

        controller = GameController(config, capture, MagicMock(), MagicMock(), pot_ocr=pot_ocr)
        controller._frame_count = 0
        controller._tick_poker()

        assert controller.poker_state.pot_size == 0.0

    def test_ocr_skipped_without_hole_cards(
        self,
        pot_ocr: PotOCR,
        pot_roi: tuple[int, int, int, int],
        bet_roi: tuple[int, int, int, int],
    ) -> None:
        """Pot OCR is gated until at least two hole cards are detected."""
        config = AppConfig(pot_ocr_enabled=True)
        capture = MagicMock()
        frame = np.zeros((160, 320, 3), dtype=np.uint8)
        frame[pot_roi[1] : pot_roi[1] + pot_roi[3], pot_roi[0] : pot_roi[0] + pot_roi[2]] = render_amount_image(
            "300.00",
            size=(pot_roi[2], pot_roi[3]),
        )
        capture.get_latest_frame.return_value = frame

        template_engine = MagicMock()
        template_engine.detect_cards.return_value = []

        controller = GameController(
            config,
            capture,
            template_engine,
            MagicMock(),
            pot_ocr=pot_ocr,
        )
        controller._frame_count = 0
        controller._tick_poker()

        assert controller.poker_state.pot_size == 0.0

    def test_set_poker_values_invalidates_cache(self, controller_with_ocr: GameController) -> None:
        with patch("riverrater.main.analyze_poker") as mock_analyze:
            from riverrater.game.state import PokerResult

            mock_analyze.return_value = PokerResult()
            controller_with_ocr._frame_count = 0
            controller_with_ocr._tick_poker()
            controller_with_ocr.set_poker_values(pot_size=50.0, bet_to_call=10.0, num_opponents=2)
            controller_with_ocr._frame_count = 3
            controller_with_ocr._tick_poker()

            assert mock_analyze.call_count == 2