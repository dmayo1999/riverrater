"""
game package — exports all shared types and analysis functions.
"""

from riverrater.game.state import (
    Card,
    Rank,
    Suit,
    PokerState,
    BlackjackState,
    PokerResult,
    BlackjackResult,
    DetectionMeta,
    GameMode,
    PokerAction,
    BlackjackAction,
    FULL_DECK,
)

from riverrater.game.poker_math import analyze_poker
from riverrater.game.blackjack_math import analyze_blackjack

__all__ = [
    "Card",
    "Rank",
    "Suit",
    "PokerState",
    "BlackjackState",
    "PokerResult",
    "BlackjackResult",
    "DetectionMeta",
    "GameMode",
    "PokerAction",
    "BlackjackAction",
    "FULL_DECK",
    "analyze_poker",
    "analyze_blackjack",
]
