"""Tests for GameController poker tick caching and blackjack YOLO wiring."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from riverrater.game.state import (
    BlackjackAction,
    BlackjackResult,
    Card,
    DetectionMeta,
    GameMode,
    PokerAction,
    PokerResult,
    PokerState,
    Rank,
    Suit,
)
from riverrater.game.poker_math import _cached_equity, clear_equity_cache
from riverrater.main import AppConfig, GameController, _poker_equity_hash, _poker_full_hash
from riverrater.utils.session_log import SessionLogger


def _make_card(rank_val: str, suit_val: str) -> Card:
    return Card(rank=Rank(rank_val), suit=Suit(suit_val))


@pytest.fixture
def controller() -> GameController:
    """GameController with mocked capture, vision, and overlay."""
    config = AppConfig()
    capture = MagicMock()
    capture.get_latest_frame.return_value = None
    template_engine = MagicMock()
    overlay = MagicMock()
    return GameController(config, capture, template_engine, overlay)


def test_tick_poker_skips_analyze_when_state_unchanged(controller: GameController) -> None:
    """analyze_poker runs once across repeated ticks with identical PokerState."""
    with patch("riverrater.main.analyze_poker", return_value=PokerResult()) as mock_analyze:
        controller._tick_poker()
        controller._tick_poker()
        assert mock_analyze.call_count == 1


def test_tick_poker_skips_redundant_hud_push_when_unchanged(controller: GameController) -> None:
    """HUD is updated on first tick only when state and detection meta are stable."""
    with patch("riverrater.main.analyze_poker", return_value=PokerResult()):
        controller._tick_poker()
        controller._tick_poker()
    assert controller.overlay.update_poker.call_count == 1


def test_tick_poker_analyzes_after_set_poker_values(controller: GameController) -> None:
    """Manual pot/bet/opponent input invalidates cache and re-runs analysis."""
    with patch("riverrater.main.analyze_poker", return_value=PokerResult()) as mock_analyze:
        controller._tick_poker()
        controller.set_poker_values(pot_size=100.0, bet_to_call=25.0, num_opponents=2)
        controller._tick_poker()
        assert mock_analyze.call_count == 2
        assert controller.overlay.update_poker.call_count == 2


def test_tick_poker_analyzes_when_cards_change(controller: GameController) -> None:
    """Changing hole cards between ticks triggers a fresh analyze_poker call."""
    card_ah = _make_card("A", "h")
    card_kd = _make_card("K", "d")

    with patch("riverrater.main.analyze_poker", return_value=PokerResult()) as mock_analyze:
        controller.poker_state.hole_cards = [card_ah, card_kd]
        controller._tick_poker()

        controller.poker_state.community_cards = [_make_card("2", "c")]
        controller._tick_poker()

        assert mock_analyze.call_count == 2


def test_reset_hand_forces_hud_update_on_next_tick(controller: GameController) -> None:
    """reset_hand clears cache so the HUD refreshes even from an empty state."""
    with patch("riverrater.main.analyze_poker", return_value=PokerResult()):
        controller._tick_poker()
        controller._tick_poker()
        controller.reset_hand()
        controller._tick_poker()

    assert controller.overlay.update_poker.call_count == 2


def test_tick_poker_recomputes_ev_on_pot_only_change(controller: GameController) -> None:
    """Pot/bet changes reuse cached equity via recompute_poker_ev."""
    card_ah = _make_card("A", "h")
    card_kd = _make_card("K", "d")
    controller.poker_state.hole_cards = [card_ah, card_kd]
    poker_result = PokerResult(win_pct=0.55, tie_pct=0.05)

    with patch("riverrater.main.analyze_poker", return_value=poker_result) as mock_analyze:
        with patch("riverrater.main.recompute_poker_ev", return_value=PokerResult()) as mock_recompute:
            controller._tick_poker()
            controller.poker_state.pot_size = 200.0
            controller._tick_poker()

            assert mock_analyze.call_count == 1
            assert mock_recompute.call_count == 1
            mock_recompute.assert_called_with(
                0.55,
                0.05,
                200.0,
                controller.poker_state.bet_to_call,
            )


def test_reset_hand_clears_equity_cache(controller: GameController) -> None:
    from riverrater.game.poker_math import calculate_equity

    clear_equity_cache()
    controller.poker_state.hole_cards = [_make_card("A", "h"), _make_card("K", "d")]
    calculate_equity(controller.poker_state.hole_cards, [], num_opponents=1, simulations=100)
    assert _cached_equity.cache_info().currsize >= 1

    controller.reset_hand()
    assert _cached_equity.cache_info().currsize == 0


def test_poker_equity_and_full_hash_split() -> None:
    state = PokerState(
        hole_cards=[_make_card("A", "h"), _make_card("K", "d")],
        community_cards=[_make_card("2", "c")],
        pot_size=100.0,
        bet_to_call=25.0,
        num_opponents=2,
    )
    equity_hash = _poker_equity_hash(state)
    full_hash = _poker_full_hash(state)

    assert equity_hash == (
        tuple(state.hole_cards),
        tuple(state.community_cards),
        2,
    )
    assert full_hash[:3] == equity_hash
    assert full_hash[3:] == (100.0, 25.0)


def test_detection_meta_change_updates_hud_without_reanalyze(controller: GameController) -> None:
    """Confidence-only detection changes refresh HUD but skip analyze_poker."""
    card_ah = _make_card("A", "h")
    card_kd = _make_card("K", "d")
    controller.poker_state.hole_cards = [card_ah, card_kd]
    controller._detection_meta = DetectionMeta.from_detections([
        (card_ah, (0, 0, 10, 10), 0.80),
        (card_kd, (20, 0, 10, 10), 0.70),
    ])

    with patch("riverrater.main.analyze_poker", return_value=PokerResult()) as mock_analyze:
        controller._tick_poker()

        controller._detection_meta = DetectionMeta.from_detections([
            (card_ah, (0, 0, 10, 10), 0.99),
            (card_kd, (20, 0, 10, 10), 0.99),
        ])
        controller._tick_poker()

        assert mock_analyze.call_count == 1
        assert controller.overlay.update_poker.call_count == 2


# ---------------------------------------------------------------------------
# Blackjack YOLO integration
# ---------------------------------------------------------------------------


def _make_blackjack_controller(
    *,
    yolo_engine: MagicMock | None = None,
    frame: np.ndarray | None = None,
) -> GameController:
    config = AppConfig(game_mode=GameMode.BLACKJACK.value, yolo_confidence=0.5)
    capture = MagicMock()
    capture.get_latest_frame.return_value = frame
    return GameController(
        config,
        capture,
        MagicMock(),
        MagicMock(),
        yolo_engine=yolo_engine,
    )


def test_tick_blackjack_skips_yolo_when_unavailable() -> None:
    yolo_engine = MagicMock()
    yolo_engine.is_available = False
    controller = _make_blackjack_controller(
        yolo_engine=yolo_engine,
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
    )

    controller._frame_count = 0
    controller._tick_blackjack()

    yolo_engine.detect_cards.assert_not_called()


def test_tick_blackjack_runs_yolo_when_available() -> None:
    yolo_engine = MagicMock()
    yolo_engine.is_available = True
    yolo_engine.detect_cards.return_value = []
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    controller = _make_blackjack_controller(yolo_engine=yolo_engine, frame=frame)

    controller._frame_count = 0
    controller._tick_blackjack()

    yolo_engine.detect_cards.assert_called_once_with(frame, confidence=0.5)


def test_apply_blackjack_detections_assigns_dealer_and_player() -> None:
    dealer = _make_card("K", "s")
    player_one = _make_card("A", "h")
    player_two = _make_card("9", "d")
    detections = [
        (player_one, (10, 80, 40, 60), 0.91),
        (dealer, (10, 10, 40, 60), 0.88),
        (player_two, (60, 90, 40, 60), 0.86),
    ]
    controller = _make_blackjack_controller()

    controller._apply_blackjack_detections(detections)

    assert controller.blackjack_state.dealer_upcard == dealer
    assert controller.blackjack_state.player_hand == [player_one, player_two]
    assert set(controller.blackjack_state.cards_seen) == {dealer, player_one, player_two}
    assert controller._detection_meta is not None
    assert controller._detection_meta.overall_confidence == pytest.approx(0.883333, rel=1e-3)


def test_apply_blackjack_detections_dedupes_same_card_across_frames() -> None:
    """Same physical card detected in 10 frames is counted once."""
    card = _make_card("7", "c")
    detections = [(card, (0, 0, 10, 10), 0.9)]
    controller = _make_blackjack_controller()

    for _ in range(10):
        controller._apply_blackjack_detections(detections)

    assert controller.blackjack_state.cards_seen.count(card) == 1


def test_apply_blackjack_detections_counts_duplicate_ranks_at_different_positions() -> None:
    """Multi-deck shoe: two 7c at different table positions both count."""
    card = _make_card("7", "c")
    controller = _make_blackjack_controller()

    controller._apply_blackjack_detections([(card, (0, 0, 10, 10), 0.9)])
    controller._apply_blackjack_detections([(card, (300, 0, 10, 10), 0.9)])

    assert controller.blackjack_state.cards_seen.count(card) == 2


def test_reset_shoe_clears_card_tracker() -> None:
    card = _make_card("A", "s")
    detections = [(card, (10, 10, 40, 60), 0.9)]
    controller = _make_blackjack_controller()

    controller._apply_blackjack_detections(detections)
    controller.reset_shoe()
    controller._apply_blackjack_detections(detections)

    assert controller.blackjack_state.cards_seen.count(card) == 1


def test_manual_input_works_without_yolo_engine() -> None:
    controller = _make_blackjack_controller(yolo_engine=None)

    controller.add_card_manual("Ah", "player")
    controller.add_card_manual("Kd", "dealer")

    assert controller.blackjack_state.player_hand == [_make_card("A", "h")]
    assert controller.blackjack_state.dealer_upcard == _make_card("K", "d")
    assert controller._detection_meta is not None
    assert controller._detection_meta.is_manual is True


# ---------------------------------------------------------------------------
# Session logging integration
# ---------------------------------------------------------------------------


def test_tick_poker_logs_once_for_unchanged_state() -> None:
    session_logger = MagicMock(spec=SessionLogger)
    config = AppConfig(session_logging=True)
    controller = GameController(
        config,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        session_logger=session_logger,
    )
    controller.poker_state.hole_cards = [_make_card("A", "h"), _make_card("K", "d")]
    poker_result = PokerResult(
        actual_equity=0.55,
        ev_call=10.0,
        recommended_action=PokerAction.CALL,
    )

    with patch("riverrater.main.analyze_poker", return_value=poker_result):
        controller._tick_poker()
        controller._tick_poker()

    assert session_logger.log_poker.call_count == 1


def test_tick_poker_logs_on_state_change(tmp_path) -> None:
    session_logger = MagicMock(spec=SessionLogger)
    config = AppConfig(session_logging=True)
    controller = GameController(
        config,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        session_logger=session_logger,
    )
    controller.poker_state.hole_cards = [_make_card("A", "h"), _make_card("K", "d")]
    poker_result = PokerResult(
        actual_equity=0.55,
        ev_call=10.0,
        recommended_action=PokerAction.CALL,
    )

    with patch("riverrater.main.analyze_poker", return_value=poker_result):
        controller._tick_poker()
        controller.poker_state.community_cards = [_make_card("2", "c")]
        controller._tick_poker()

    assert session_logger.log_poker.call_count == 2


def test_tick_blackjack_logs_through_session_logger() -> None:
    session_logger = MagicMock(spec=SessionLogger)
    config = AppConfig(game_mode=GameMode.BLACKJACK.value)
    controller = GameController(
        config,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        session_logger=session_logger,
    )
    bj_result = BlackjackResult(
        running_count=3,
        true_count=1.2,
        recommended_action=BlackjackAction.STAND,
        recommended_bet=40.0,
    )

    with patch("riverrater.main.analyze_blackjack", return_value=bj_result):
        controller._tick_blackjack()
        controller._tick_blackjack()

    assert session_logger.log_blackjack.call_count == 2