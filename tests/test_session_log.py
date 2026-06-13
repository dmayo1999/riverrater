"""Tests for JSONL session logging."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from riverrater.game.state import (
    BlackjackAction,
    BlackjackResult,
    BlackjackState,
    Card,
    PokerAction,
    PokerResult,
    PokerState,
    Rank,
    Suit,
)
from riverrater.utils.session_log import SessionLogger


def _make_card(rank_val: str, suit_val: str) -> Card:
    return Card(rank=Rank(rank_val), suit=Suit(suit_val))


@pytest.fixture
def sessions_dir(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


@pytest.fixture
def logger(sessions_dir: Path) -> SessionLogger:
    return SessionLogger(enabled=True, sessions_dir=sessions_dir)


def _read_lines(sessions_dir: Path) -> list[dict]:
    files = list(sessions_dir.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def test_log_poker_writes_jsonl(logger: SessionLogger, sessions_dir: Path) -> None:
    state = PokerState(
        hole_cards=[_make_card("A", "h"), _make_card("K", "d")],
        pot_size=100.0,
        bet_to_call=25.0,
        num_opponents=2,
    )
    result = PokerResult(
        actual_equity=0.62,
        required_equity=0.33,
        ev_call=12.5,
        ev_fold=0.0,
        ev_raise=18.0,
        recommended_action=PokerAction.RAISE,
    )

    logger.log_poker(state, result)

    records = _read_lines(sessions_dir)
    assert len(records) == 1
    record = records[0]
    assert record["mode"] == "poker"
    assert record["hole_cards"] == ["Ah", "Kd"]
    assert record["pot_size"] == 100.0
    assert record["actual_equity"] == pytest.approx(0.62)
    assert record["ev_call"] == pytest.approx(12.5)
    assert record["recommended_action"] == "raise"
    assert "ts" in record


def test_log_poker_dedupes_unchanged_ticks(logger: SessionLogger, sessions_dir: Path) -> None:
    state = PokerState(hole_cards=[_make_card("A", "h"), _make_card("K", "d")])
    result = PokerResult(actual_equity=0.5, recommended_action=PokerAction.CALL)

    logger.log_poker(state, result)
    logger.log_poker(state, result)

    records = _read_lines(sessions_dir)
    assert len(records) == 1


def test_log_poker_logs_on_equity_change(logger: SessionLogger, sessions_dir: Path) -> None:
    state = PokerState(hole_cards=[_make_card("A", "h"), _make_card("K", "d")])
    first = PokerResult(actual_equity=0.5, recommended_action=PokerAction.CALL)
    second = PokerResult(actual_equity=0.55, recommended_action=PokerAction.CALL)

    logger.log_poker(state, first)
    logger.log_poker(state, second)

    records = _read_lines(sessions_dir)
    assert len(records) == 2


def test_log_blackjack_writes_count_strategy_bet(
    logger: SessionLogger,
    sessions_dir: Path,
) -> None:
    state = BlackjackState(
        player_hand=[_make_card("A", "h"), _make_card("9", "d")],
        dealer_upcard=_make_card("K", "s"),
        cards_seen=[_make_card("2", "c"), _make_card("5", "h")],
    )
    result = BlackjackResult(
        running_count=2,
        true_count=1.5,
        recommended_action=BlackjackAction.STAND,
        recommended_bet=25.0,
        shoe_favorability=0.62,
    )

    logger.log_blackjack(state, result)

    records = _read_lines(sessions_dir)
    assert len(records) == 1
    record = records[0]
    assert record["mode"] == "blackjack"
    assert record["running_count"] == 2
    assert record["true_count"] == pytest.approx(1.5)
    assert record["recommended_action"] == "Stand"
    assert record["recommended_bet"] == pytest.approx(25.0)
    assert record["player_hand"] == ["Ah", "9d"]
    assert record["dealer_upcard"] == "Ks"
    assert record["cards_seen"] == 2


def test_log_blackjack_dedupes_unchanged_ticks(
    logger: SessionLogger,
    sessions_dir: Path,
) -> None:
    state = BlackjackState()
    result = BlackjackResult(running_count=0, recommended_bet=10.0)

    logger.log_blackjack(state, result)
    logger.log_blackjack(state, result)

    records = _read_lines(sessions_dir)
    assert len(records) == 1


def test_disabled_logger_writes_nothing(sessions_dir: Path) -> None:
    logger = SessionLogger(enabled=False, sessions_dir=sessions_dir)
    state = PokerState(hole_cards=[_make_card("A", "h"), _make_card("K", "d")])
    result = PokerResult(actual_equity=0.5)

    logger.log_poker(state, result)

    assert list(sessions_dir.glob("*.jsonl")) == []


def test_reset_poker_cache_forces_next_log(logger: SessionLogger, sessions_dir: Path) -> None:
    state = PokerState(hole_cards=[_make_card("A", "h"), _make_card("K", "d")])
    result = PokerResult(actual_equity=0.5)

    logger.log_poker(state, result)
    logger.reset_poker_cache()
    logger.log_poker(state, result)

    records = _read_lines(sessions_dir)
    assert len(records) == 2


@patch("riverrater.utils.session_log.date")
def test_daily_log_filename(mock_date, logger: SessionLogger, sessions_dir: Path) -> None:
    mock_date.today.return_value = date.fromisoformat("2026-06-13")

    state = PokerState()
    logger.log_poker(state, PokerResult())

    assert (sessions_dir / "2026-06-13.jsonl").exists()