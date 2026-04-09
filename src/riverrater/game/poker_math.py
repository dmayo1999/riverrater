"""
poker_math.py — Monte Carlo equity calculator and EV analysis for RiverRater.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Optional

from riverrater.game.state import (
    Card,
    FULL_DECK,
    PokerAction,
    PokerResult,
    PokerState,
    Rank,
    Suit,
)


# ---------------------------------------------------------------------------
# Rank ordering for poker (TWO=2 … ACE=14)
# ---------------------------------------------------------------------------

_RANK_ORDER: dict[Rank, int] = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 11,
    Rank.QUEEN: 12,
    Rank.KING: 13,
    Rank.ACE: 14,
}

# Hand rank constants
HAND_HIGH_CARD = 0
HAND_PAIR = 1
HAND_TWO_PAIR = 2
HAND_THREE_OF_A_KIND = 3
HAND_STRAIGHT = 4
HAND_FLUSH = 5
HAND_FULL_HOUSE = 6
HAND_FOUR_OF_A_KIND = 7
HAND_STRAIGHT_FLUSH = 8
HAND_ROYAL_FLUSH = 9


# ---------------------------------------------------------------------------
# Hand evaluation
# ---------------------------------------------------------------------------

def _rank_values(cards: list[Card]) -> list[int]:
    """Return sorted (descending) rank values for a set of cards."""
    return sorted([_RANK_ORDER[c.rank] for c in cards], reverse=True)


def _best_straight(values: list[int]) -> Optional[int]:
    """Return the highest straight top card from a sorted list of unique values, or None.

    Handles the wheel (A-2-3-4-5) by treating Ace as 1.
    `values` must be sorted descending and de-duplicated.
    """
    # Add ace-low representation
    if 14 in values:
        values = values + [1]
    for top in range(len(values) - 4):
        window = values[top:top + 5]
        if window[0] - window[4] == 4 and len(set(window)) == 5:
            return window[0]
    return None


def evaluate_hand(cards: list[Card]) -> tuple[int, list[int]]:
    """Evaluate the best 5-card poker hand from up to 7 cards.

    Returns (hand_rank, kickers) where hand_rank is 0 (high card) – 9 (royal flush)
    and kickers is a list of integers used to break ties, highest first.
    """
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")

    best: Optional[tuple[int, list[int]]] = None

    for combo in combinations(cards, 5):
        result = _evaluate_five(list(combo))
        if best is None or result > best:
            best = result

    assert best is not None
    return best


def _evaluate_five(cards: list[Card]) -> tuple[int, list[int]]:
    """Evaluate exactly 5 cards and return (hand_rank, kickers)."""
    values = sorted([_RANK_ORDER[c.rank] for c in cards], reverse=True)
    suits = [c.suit for c in cards]

    is_flush = len(set(suits)) == 1
    unique_values = sorted(set(values), reverse=True)
    straight_top = _best_straight(unique_values)
    is_straight = straight_top is not None

    # Count occurrences
    from collections import Counter
    counts = Counter(values)
    # Sort by (count desc, rank desc) for kicker ordering
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    group_ranks = [rank for rank, _ in groups]
    group_counts = [cnt for _, cnt in groups]

    if is_straight and is_flush:
        if straight_top == 14:
            return (HAND_ROYAL_FLUSH, [straight_top])
        return (HAND_STRAIGHT_FLUSH, [straight_top])

    if group_counts[0] == 4:
        quad_rank = group_ranks[0]
        kicker = group_ranks[1]
        return (HAND_FOUR_OF_A_KIND, [quad_rank, kicker])

    if group_counts[0] == 3 and group_counts[1] == 2:
        return (HAND_FULL_HOUSE, [group_ranks[0], group_ranks[1]])

    if is_flush:
        return (HAND_FLUSH, values)

    if is_straight:
        # For wheel (A-2-3-4-5), top is 5
        return (HAND_STRAIGHT, [straight_top])

    if group_counts[0] == 3:
        trips_rank = group_ranks[0]
        kickers = sorted([r for r in values if r != trips_rank], reverse=True)
        return (HAND_THREE_OF_A_KIND, [trips_rank] + kickers)

    if group_counts[0] == 2 and group_counts[1] == 2:
        pair1 = group_ranks[0]
        pair2 = group_ranks[1]
        kicker = [r for r in values if r not in (pair1, pair2)]
        return (HAND_TWO_PAIR, [pair1, pair2] + kicker)

    if group_counts[0] == 2:
        pair_rank = group_ranks[0]
        kickers = sorted([r for r in values if r != pair_rank], reverse=True)
        return (HAND_PAIR, [pair_rank] + kickers)

    return (HAND_HIGH_CARD, values)


# ---------------------------------------------------------------------------
# Monte Carlo equity
# ---------------------------------------------------------------------------

def calculate_equity(
    hole_cards: list[Card],
    community_cards: list[Card],
    num_opponents: int = 1,
    simulations: int = 5000,
) -> tuple[float, float]:
    """Monte Carlo simulation of poker equity.

    Returns (win_pct, tie_pct) as floats in [0, 1].
    """
    known_cards = set(hole_cards + community_cards)
    remaining_deck = [c for c in FULL_DECK if c not in known_cards]

    wins = 0
    ties = 0
    total = 0

    for _ in range(simulations):
        deck_copy = remaining_deck.copy()
        random.shuffle(deck_copy)

        # Deal remaining community cards
        needed_community = 5 - len(community_cards)
        if needed_community > len(deck_copy):
            continue
        sim_community = community_cards + deck_copy[:needed_community]
        deck_copy = deck_copy[needed_community:]

        # Deal opponent hands
        if len(deck_copy) < num_opponents * 2:
            continue
        opponent_hands: list[list[Card]] = []
        for i in range(num_opponents):
            opp_hand = deck_copy[i * 2: i * 2 + 2]
            opponent_hands.append(opp_hand)

        # Evaluate player hand
        player_best = evaluate_hand(hole_cards + sim_community)

        # Evaluate opponent hands and find best opponent
        best_opp = max(evaluate_hand(opp + sim_community) for opp in opponent_hands)

        total += 1
        if player_best > best_opp:
            wins += 1
        elif player_best == best_opp:
            ties += 1

    if total == 0:
        return (0.0, 0.0)

    return (wins / total, ties / total)


# ---------------------------------------------------------------------------
# Pot odds and EV
# ---------------------------------------------------------------------------

def calculate_pot_odds(pot_size: float, bet_to_call: float) -> float:
    """Return required equity to break even on a call.

    Required equity = bet_to_call / (pot_size + bet_to_call).
    Returns 0.0 on division by zero.
    """
    denom = pot_size + bet_to_call
    if denom == 0.0:
        return 0.0
    return bet_to_call / denom


def calculate_ev(
    win_pct: float,
    pot_size: float,
    bet_to_call: float,
) -> tuple[float, float, float]:
    """Return (ev_call, ev_fold, ev_raise_estimate).

    ev_call  = win_pct * pot_size  -  (1 - win_pct) * bet_to_call
    ev_fold  = 0.0
    ev_raise = 1.5 * ev_call when ev_call > 0, else ev_call (don't
               amplify negative EV — raising a losing hand is worse
               than calling, not 1.5× better).
    """
    ev_call = (win_pct * pot_size) - ((1.0 - win_pct) * bet_to_call)
    ev_fold = 0.0
    # Only amplify positive EV; when ev_call is negative, raising is
    # strictly worse (more money into a losing spot).
    ev_raise = 1.5 * ev_call if ev_call > 0 else ev_call * 2.0
    return (ev_call, ev_fold, ev_raise)


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_poker(state: PokerState) -> PokerResult:
    """Run full poker analysis and return a populated PokerResult."""
    win_pct, tie_pct = calculate_equity(
        hole_cards=state.hole_cards,
        community_cards=state.community_cards,
        num_opponents=state.num_opponents,
    )

    actual_equity = win_pct + tie_pct / 2.0
    required_equity = calculate_pot_odds(state.pot_size, state.bet_to_call)
    ev_call, ev_fold, ev_raise = calculate_ev(win_pct, state.pot_size, state.bet_to_call)

    # Recommend highest EV action
    evs = {
        PokerAction.CALL: ev_call,
        PokerAction.FOLD: ev_fold,
        PokerAction.RAISE: ev_raise,
    }
    recommended_action = max(evs, key=lambda a: evs[a])

    return PokerResult(
        win_pct=win_pct,
        tie_pct=tie_pct,
        required_equity=required_equity,
        actual_equity=actual_equity,
        ev_call=ev_call,
        ev_fold=ev_fold,
        ev_raise=ev_raise,
        recommended_action=recommended_action,
    )
