"""
blackjack_math.py — Complete blackjack math engine for RiverRater.

Implements Hi-Lo counting, Kelly Criterion bet sizing, full basic strategy,
and shoe favorability scoring.
"""

from __future__ import annotations

import math
from typing import Optional

from riverrater.game.state import (
    BlackjackAction,
    BlackjackResult,
    BlackjackState,
    Card,
    Rank,
)


# ---------------------------------------------------------------------------
# Hand value
# ---------------------------------------------------------------------------

def hand_value(cards: list[Card]) -> tuple[int, bool]:
    """Compute blackjack hand total and softness.

    Aces start at 11; each excess 11→1 conversion reduces the total by 10.
    Returns (total, is_soft) where is_soft=True means at least one ace is
    counted as 11.
    """
    total = 0
    aces = 0
    for card in cards:
        v = card.blackjack_value
        total += v
        if card.rank == Rank.ACE:
            aces += 1

    # Convert aces from 11 to 1 as needed
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    is_soft = aces > 0  # At least one ace still counted as 11
    return (total, is_soft)


# ---------------------------------------------------------------------------
# Hi-Lo card counting
# ---------------------------------------------------------------------------

def running_count(cards_seen: list[Card]) -> int:
    """Hi-Lo running count.

    2-6  → +1
    7-9  →  0
    T/J/Q/K/A → -1
    """
    count = 0
    for card in cards_seen:
        rank = card.rank
        if rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX):
            count += 1
        elif rank in (Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE):
            count -= 1
        # 7, 8, 9 → 0, no change
    return count


def true_count(running: int, decks_remaining: float) -> float:
    """Normalise running count by decks remaining.

    decks_remaining is clamped to a minimum of 0.5 to avoid division by near-zero.
    """
    dr = max(0.5, decks_remaining)
    return running / dr


# ---------------------------------------------------------------------------
# Basic strategy
# ---------------------------------------------------------------------------

# Dealer upcard to column index (2-9, T/J/Q/K→10, A→11)
def _dealer_col(dealer: Card) -> int:
    rank = dealer.rank
    if rank == Rank.ACE:
        return 11
    if rank in (Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING):
        return 10
    return int(rank.value)


# Abbreviations used in the tables:
#   H  = Hit
#   S  = Stand
#   D  = Double (or Hit if not allowed — we always allow double)
#   P  = Split
#   Su = Surrender (or Hit if not allowed — we always allow surrender)

_H  = BlackjackAction.HIT
_S  = BlackjackAction.STAND
_D  = BlackjackAction.DOUBLE
_P  = BlackjackAction.SPLIT
_Su = BlackjackAction.SURRENDER

# ---------------------------------------------------------------------------
# Pair splitting table: key = rank value (2-11 for aces), dealer col (2-11)
# Indexed as [pair_rank_value][dealer_upcard_value]
# ---------------------------------------------------------------------------
# pair_rank_value: 2=2, 3=3, 4=4, 5=5, 6=6, 7=7, 8=8, 9=9, 10=T/J/Q/K, 11=A
#
# Standard table (dealer stands soft 17, DAS allowed, late surrender allowed):
#
#              2    3    4    5    6    7    8    9   10    A
_PAIR_TABLE: dict[int, dict[int, BlackjackAction]] = {
    2:  {2:_P, 3:_P, 4:_P, 5:_P, 6:_P, 7:_P, 8:_H, 9:_H, 10:_H, 11:_H},
    3:  {2:_P, 3:_P, 4:_P, 5:_P, 6:_P, 7:_P, 8:_H, 9:_H, 10:_H, 11:_H},
    4:  {2:_H, 3:_H, 4:_H, 5:_P, 6:_P, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},
    5:  {2:_D, 3:_D, 4:_D, 5:_D, 6:_D, 7:_D, 8:_D, 9:_D, 10:_H, 11:_H},
    6:  {2:_P, 3:_P, 4:_P, 5:_P, 6:_P, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},
    7:  {2:_P, 3:_P, 4:_P, 5:_P, 6:_P, 7:_P, 8:_H, 9:_H, 10:_H, 11:_H},
    8:  {2:_P, 3:_P, 4:_P, 5:_P, 6:_P, 7:_P, 8:_P, 9:_P, 10:_P, 11:_P},
    9:  {2:_P, 3:_P, 4:_P, 5:_P, 6:_P, 7:_S, 8:_P, 9:_P, 10:_S, 11:_S},
    10: {2:_S, 3:_S, 4:_S, 5:_S, 6:_S, 7:_S, 8:_S, 9:_S, 10:_S, 11:_S},
    11: {2:_P, 3:_P, 4:_P, 5:_P, 6:_P, 7:_P, 8:_P, 9:_P, 10:_P, 11:_P},
}

# ---------------------------------------------------------------------------
# Soft totals table: key = soft total (13=A+2 … 21=A+10/BJ)
# dealer col (2-11)
# ---------------------------------------------------------------------------
#              2    3    4    5    6    7    8    9   10    A
_SOFT_TABLE: dict[int, dict[int, BlackjackAction]] = {
    13: {2:_H, 3:_H, 4:_H, 5:_D, 6:_D, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},  # A+2
    14: {2:_H, 3:_H, 4:_H, 5:_D, 6:_D, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},  # A+3
    15: {2:_H, 3:_H, 4:_D, 5:_D, 6:_D, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},  # A+4
    16: {2:_H, 3:_H, 4:_D, 5:_D, 6:_D, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},  # A+5
    17: {2:_D, 3:_D, 4:_D, 5:_D, 6:_D, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},  # A+6
    18: {2:_S, 3:_D, 4:_D, 5:_D, 6:_D, 7:_S, 8:_S, 9:_H, 10:_H, 11:_H},  # A+7
    19: {2:_S, 3:_S, 4:_S, 5:_S, 6:_S, 7:_S, 8:_S, 9:_S, 10:_S, 11:_S},  # A+8
    20: {2:_S, 3:_S, 4:_S, 5:_S, 6:_S, 7:_S, 8:_S, 9:_S, 10:_S, 11:_S},  # A+9
    21: {2:_S, 3:_S, 4:_S, 5:_S, 6:_S, 7:_S, 8:_S, 9:_S, 10:_S, 11:_S},  # A+10/BJ
}

# ---------------------------------------------------------------------------
# Hard totals table: key = hard total (5-20), dealer col (2-11)
# Totals ≤4 → always hit; totals ≥21 → always stand
# ---------------------------------------------------------------------------
#               2    3    4    5    6    7    8    9   10    A
_HARD_TABLE: dict[int, dict[int, BlackjackAction]] = {
    5:  {2:_H, 3:_H, 4:_H, 5:_H, 6:_H, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},
    6:  {2:_H, 3:_H, 4:_H, 5:_H, 6:_H, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},
    7:  {2:_H, 3:_H, 4:_H, 5:_H, 6:_H, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},
    8:  {2:_H, 3:_H, 4:_H, 5:_H, 6:_H, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},
    9:  {2:_H, 3:_D, 4:_D, 5:_D, 6:_D, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},
    10: {2:_D, 3:_D, 4:_D, 5:_D, 6:_D, 7:_D, 8:_D, 9:_D, 10:_H, 11:_H},
    11: {2:_D, 3:_D, 4:_D, 5:_D, 6:_D, 7:_D, 8:_D, 9:_D, 10:_D, 11:_H},
    12: {2:_H, 3:_H, 4:_S, 5:_S, 6:_S, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},
    13: {2:_S, 3:_S, 4:_S, 5:_S, 6:_S, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},
    14: {2:_S, 3:_S, 4:_S, 5:_S, 6:_S, 7:_H, 8:_H, 9:_H, 10:_H, 11:_H},
    15: {2:_S, 3:_S, 4:_S, 5:_S, 6:_S, 7:_H, 8:_H, 9:_H, 10:_Su, 11:_H},
    16: {2:_S, 3:_S, 4:_S, 5:_S, 6:_S, 7:_H, 8:_H, 9:_Su, 10:_Su, 11:_Su},
    17: {2:_S, 3:_S, 4:_S, 5:_S, 6:_S, 7:_S, 8:_S, 9:_S, 10:_S, 11:_S},
    18: {2:_S, 3:_S, 4:_S, 5:_S, 6:_S, 7:_S, 8:_S, 9:_S, 10:_S, 11:_S},
    19: {2:_S, 3:_S, 4:_S, 5:_S, 6:_S, 7:_S, 8:_S, 9:_S, 10:_S, 11:_S},
    20: {2:_S, 3:_S, 4:_S, 5:_S, 6:_S, 7:_S, 8:_S, 9:_S, 10:_S, 11:_S},
}


def _pair_rank_value(card: Card) -> int:
    """Return the pair grouping value for a card (Ace=11, face=10, pip=pip)."""
    if card.rank == Rank.ACE:
        return 11
    if card.rank in (Rank.JACK, Rank.QUEEN, Rank.KING, Rank.TEN):
        return 10
    return int(card.rank.value)


def _is_pair(player_cards: list[Card]) -> bool:
    """Return True if the player holds exactly 2 cards of the same blackjack pair value."""
    if len(player_cards) != 2:
        return False
    return _pair_rank_value(player_cards[0]) == _pair_rank_value(player_cards[1])


def basic_strategy(player_cards: list[Card], dealer_upcard: Card) -> BlackjackAction:
    """Return the optimal BlackjackAction for the given hand and dealer upcard.

    Rules assumed: dealer stands on soft 17, DAS allowed, late surrender allowed.
    Pair splitting, soft totals, and hard totals are all handled.
    """
    dc = _dealer_col(dealer_upcard)
    total, is_soft = hand_value(player_cards)

    # --- Pairs ---
    if _is_pair(player_cards):
        pv = _pair_rank_value(player_cards[0])
        action = _PAIR_TABLE.get(pv, {}).get(dc)
        if action is not None and action == _P:
            return _P
        # If not splitting, fall through to soft/hard logic

    # --- Soft totals ---
    if is_soft and total in _SOFT_TABLE:
        action = _SOFT_TABLE[total].get(dc)
        if action is not None:
            return action

    # --- Hard totals ---
    if total <= 4:
        return _H
    if total >= 21:
        return _S
    action = _HARD_TABLE.get(total, {}).get(dc, _H)
    return action


# ---------------------------------------------------------------------------
# Kelly Criterion bet sizing
# ---------------------------------------------------------------------------

def kelly_bet(
    tc: float,
    min_bet: float,
    max_bet: float,
    bankroll: float,
) -> float:
    """Compute Kelly Criterion bet size.

    Player edge ≈ (true_count - 1) × 0.005  (Hi-Lo approximation).
    If edge ≤ 0, return min_bet.
    Recommended bet = bankroll × edge (even money: kelly fraction = edge).
    Clamped to [min_bet, max_bet] and rounded to nearest min_bet increment.
    """
    edge = (tc - 1.0) * 0.005
    if edge <= 0.0:
        return min_bet

    kelly_fraction = edge  # Even money: odds = 1.0, so fraction = edge/1
    raw_bet = bankroll * kelly_fraction

    # Round to nearest min_bet increment
    if min_bet > 0:
        raw_bet = round(raw_bet / min_bet) * min_bet

    # Clamp
    return max(min_bet, min(max_bet, raw_bet))


# ---------------------------------------------------------------------------
# Shoe favourability (heat meter)
# ---------------------------------------------------------------------------

def shoe_favorability(tc: float) -> float:
    """Map true count to a 0.0–1.0 favourability score using linear interpolation.

    Breakpoints:
        TC ≤ -2  → 0.0
        TC   0   → 0.3
        TC  +2   → 0.6
        TC  +5   → 0.9
        TC ≥ +7  → 1.0
    """
    # Define piecewise linear breakpoints (tc_value, favorability)
    breakpoints = [
        (-2.0, 0.0),
        (0.0,  0.3),
        (2.0,  0.6),
        (5.0,  0.9),
        (7.0,  1.0),
    ]

    if tc <= breakpoints[0][0]:
        return breakpoints[0][1]
    if tc >= breakpoints[-1][0]:
        return breakpoints[-1][1]

    # Find the surrounding segment
    for i in range(len(breakpoints) - 1):
        tc_lo, fav_lo = breakpoints[i]
        tc_hi, fav_hi = breakpoints[i + 1]
        if tc_lo <= tc <= tc_hi:
            t = (tc - tc_lo) / (tc_hi - tc_lo)
            return fav_lo + t * (fav_hi - fav_lo)

    return 0.0  # Fallback (unreachable)


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_blackjack(
    state: BlackjackState,
    min_bet: float = 10.0,
    max_bet: float = 500.0,
    bankroll: float = 5000.0,
) -> BlackjackResult:
    """Run full blackjack analysis and return a populated BlackjackResult.

    Parameters
    ----------
    state:
        Current blackjack hand and shoe state.
    min_bet:
        Table minimum bet (from config).
    max_bet:
        Table maximum bet (from config).
    bankroll:
        Player's current bankroll (from config).
    """
    # Hand value
    hand_total, is_soft = hand_value(state.player_hand)

    # Counting
    rc = running_count(state.cards_seen)
    total_cards_remaining = state.num_decks * 52 - len(state.cards_seen)
    decks_remaining = max(0.5, total_cards_remaining / 52.0)
    tc = true_count(rc, decks_remaining)

    # Recommended action (requires dealer upcard)
    recommended_action: Optional[BlackjackAction] = None
    if state.dealer_upcard is not None and len(state.player_hand) >= 2:
        recommended_action = basic_strategy(state.player_hand, state.dealer_upcard)

    # Kelly bet — now uses caller-provided config values
    recommended_bet = kelly_bet(
        tc=tc,
        min_bet=min_bet,
        max_bet=max_bet,
        bankroll=bankroll,
    )

    # Shoe favourability
    fav = shoe_favorability(tc)

    return BlackjackResult(
        running_count=rc,
        true_count=tc,
        recommended_action=recommended_action,
        recommended_bet=recommended_bet,
        shoe_favorability=fav,
        hand_total=hand_total,
        is_soft=is_soft,
    )
