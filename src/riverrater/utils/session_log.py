"""
JSONL session logging for RiverRater.

Appends one JSON object per line to ``~/.riverrater/sessions/{date}.jsonl``.
Poker and blackjack events are recorded only when meaningful result fields
change, avoiding duplicate lines on unchanged ticks.
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from riverrater.game.state import (
    BlackjackAction,
    BlackjackResult,
    BlackjackState,
    PokerAction,
    PokerResult,
    PokerState,
)
from riverrater.utils.logging import get_logger

logger = get_logger(__name__)

_SESSIONS_DIR = Path.home() / ".riverrater" / "sessions"


def _poker_log_fingerprint(state: PokerState, result: PokerResult) -> tuple:
    """Hashable snapshot of poker fields worth persisting."""
    action = result.recommended_action.value if result.recommended_action else None
    return (
        tuple(str(card) for card in state.hole_cards),
        tuple(str(card) for card in state.community_cards),
        state.pot_size,
        state.bet_to_call,
        state.num_opponents,
        round(result.actual_equity, 6),
        round(result.required_equity, 6),
        round(result.ev_call, 4),
        round(result.ev_fold, 4),
        round(result.ev_raise, 4),
        action,
    )


def _blackjack_log_fingerprint(state: BlackjackState, result: BlackjackResult) -> tuple:
    """Hashable snapshot of blackjack fields worth persisting."""
    action = (
        result.recommended_action.value if result.recommended_action else None
    )
    return (
        result.running_count,
        round(result.true_count, 4),
        action,
        round(result.recommended_bet, 2),
        round(result.shoe_favorability, 4),
        tuple(str(card) for card in state.player_hand),
        str(state.dealer_upcard) if state.dealer_upcard else None,
        len(state.cards_seen),
    )


def _serialize_poker_event(state: PokerState, result: PokerResult) -> dict[str, Any]:
    """Build a JSON-serializable poker log record."""
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": "poker",
        "hole_cards": [str(card) for card in state.hole_cards],
        "community_cards": [str(card) for card in state.community_cards],
        "pot_size": state.pot_size,
        "bet_to_call": state.bet_to_call,
        "num_opponents": state.num_opponents,
        "actual_equity": result.actual_equity,
        "required_equity": result.required_equity,
        "ev_call": result.ev_call,
        "ev_fold": result.ev_fold,
        "ev_raise": result.ev_raise,
        "recommended_action": (
            result.recommended_action.value if result.recommended_action else None
        ),
    }


def _serialize_blackjack_event(
    state: BlackjackState,
    result: BlackjackResult,
) -> dict[str, Any]:
    """Build a JSON-serializable blackjack log record."""
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": "blackjack",
        "running_count": result.running_count,
        "true_count": result.true_count,
        "recommended_action": (
            result.recommended_action.value if result.recommended_action else None
        ),
        "recommended_bet": result.recommended_bet,
        "shoe_favorability": result.shoe_favorability,
        "player_hand": [str(card) for card in state.player_hand],
        "dealer_upcard": str(state.dealer_upcard) if state.dealer_upcard else None,
        "cards_seen": len(state.cards_seen),
    }


class SessionLogger:
    """Append-only JSONL logger for poker and blackjack session events."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        sessions_dir: Path | str | None = None,
    ) -> None:
        self.enabled = enabled
        self._sessions_dir = Path(sessions_dir) if sessions_dir is not None else _SESSIONS_DIR
        self._lock = threading.Lock()
        self._last_poker_fingerprint: Optional[tuple] = None
        self._last_blackjack_fingerprint: Optional[tuple] = None

    def log_poker(self, state: PokerState, result: PokerResult) -> None:
        """Log poker equity/EV/action when the meaningful snapshot changes."""
        if not self.enabled:
            return

        fingerprint = _poker_log_fingerprint(state, result)
        if fingerprint == self._last_poker_fingerprint:
            return

        self._last_poker_fingerprint = fingerprint
        self._append(_serialize_poker_event(state, result))

    def log_blackjack(self, state: BlackjackState, result: BlackjackResult) -> None:
        """Log blackjack count/strategy/bet when the meaningful snapshot changes."""
        if not self.enabled:
            return

        fingerprint = _blackjack_log_fingerprint(state, result)
        if fingerprint == self._last_blackjack_fingerprint:
            return

        self._last_blackjack_fingerprint = fingerprint
        self._append(_serialize_blackjack_event(state, result))

    def reset_poker_cache(self) -> None:
        """Clear poker deduplication so the next event is always logged."""
        self._last_poker_fingerprint = None

    def reset_blackjack_cache(self) -> None:
        """Clear blackjack deduplication so the next event is always logged."""
        self._last_blackjack_fingerprint = None

    def _append(self, record: dict[str, Any]) -> None:
        """Write one JSON line to the daily session file."""
        with self._lock:
            try:
                self._sessions_dir.mkdir(parents=True, exist_ok=True)
                log_path = self._sessions_dir / f"{date.today().isoformat()}.jsonl"
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, separators=(",", ":")))
                    fh.write("\n")
            except OSError as exc:
                logger.warning("Session log write failed: %s", exc)