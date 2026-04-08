"""
state.py — Canonical shared data types for RiverRater.

All other modules import from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

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


class PokerAction(Enum):
    FOLD = "fold"
    CALL = "call"
    RAISE = "raise"


class BlackjackAction(Enum):
    HIT = "Hit"
    STAND = "Stand"
    DOUBLE = "Double"
    SPLIT = "Split"
    SURRENDER = "Surrender"


class GameMode(Enum):
    POKER = "poker"
    BLACKJACK = "blackjack"


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------

@dataclass
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.value}"

    def __repr__(self) -> str:
        return f"Card({self.rank.value}{self.suit.value})"

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    @classmethod
    def from_string(cls, s: str) -> "Card":
        """Parse a card string like 'Ah', 'Td', '2c'.

        The first character(s) are the rank, the last character is the suit.
        """
        if len(s) < 2:
            raise ValueError(f"Invalid card string: {s!r}")
        rank_str = s[:-1].upper()
        suit_str = s[-1].lower()
        # Handle '10' as 'T'
        if rank_str == "10":
            rank_str = "T"
        rank = Rank(rank_str)
        suit = Suit(suit_str)
        return cls(rank=rank, suit=suit)

    @property
    def blackjack_value(self) -> int:
        """Blackjack pip value: ace=11, face cards=10, pip cards=face value."""
        if self.rank == Rank.ACE:
            return 11
        if self.rank in (Rank.JACK, Rank.QUEEN, Rank.KING, Rank.TEN):
            return 10
        # TWO through NINE — rank value is a digit string
        return int(self.rank.value)


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PokerState:
    hole_cards: list[Card] = field(default_factory=list)       # Player's 2 cards
    community_cards: list[Card] = field(default_factory=list)  # 0, 3, 4, or 5 cards
    pot_size: float = 0.0
    bet_to_call: float = 0.0
    num_opponents: int = 1


@dataclass
class BlackjackState:
    player_hand: list[Card] = field(default_factory=list)
    dealer_upcard: Optional[Card] = None
    cards_seen: list[Card] = field(default_factory=list)   # All cards dealt from shoe
    num_decks: int = 6                                     # Standard shoe size


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PokerResult:
    win_pct: float = 0.0
    tie_pct: float = 0.0
    required_equity: float = 0.0    # pot_odds as equity threshold
    actual_equity: float = 0.0      # win_pct + tie_pct / 2
    ev_call: float = 0.0
    ev_fold: float = 0.0
    ev_raise: float = 0.0
    recommended_action: Optional[PokerAction] = None


@dataclass
class BlackjackResult:
    running_count: int = 0
    true_count: float = 0.0
    recommended_action: Optional[BlackjackAction] = None
    recommended_bet: float = 0.0    # Kelly-based bet size
    shoe_favorability: float = 0.0  # 0.0 to 1.0 for heat meter
    hand_total: int = 0
    is_soft: bool = False           # Contains usable ace


# ---------------------------------------------------------------------------
# Full deck — computed at module level
# ---------------------------------------------------------------------------

FULL_DECK: list[Card] = [
    Card(rank=rank, suit=suit)
    for suit in Suit
    for rank in Rank
]
