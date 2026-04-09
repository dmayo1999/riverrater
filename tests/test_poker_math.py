"""
tests/test_poker_math.py — Comprehensive tests for poker_math module.
"""

import random
import pytest

from riverrater.game.state import (
    Card, Rank, Suit, PokerState, PokerResult, PokerAction,
)
from riverrater.game.poker_math import (
    evaluate_hand,
    calculate_equity,
    calculate_pot_odds,
    calculate_ev,
    analyze_poker,
    HAND_HIGH_CARD,
    HAND_PAIR,
    HAND_TWO_PAIR,
    HAND_THREE_OF_A_KIND,
    HAND_STRAIGHT,
    HAND_FLUSH,
    HAND_FULL_HOUSE,
    HAND_FOUR_OF_A_KIND,
    HAND_STRAIGHT_FLUSH,
    HAND_ROYAL_FLUSH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def c(s: str) -> Card:
    """Shorthand card constructor."""
    return Card.from_string(s)


def cards(*strings: str) -> list[Card]:
    return [c(s) for s in strings]


# ---------------------------------------------------------------------------
# test_hand_evaluation
# ---------------------------------------------------------------------------

class TestHandEvaluation:
    def test_high_card(self):
        hand = cards("2h", "5d", "7c", "9s", "Jh")
        rank, kickers = evaluate_hand(hand)
        assert rank == HAND_HIGH_CARD
        assert kickers[0] == 11  # Jack high

    def test_pair(self):
        hand = cards("Ah", "Ad", "2c", "5s", "9h")
        rank, kickers = evaluate_hand(hand)
        assert rank == HAND_PAIR
        assert kickers[0] == 14  # Pair of aces

    def test_two_pair(self):
        hand = cards("Ah", "Ad", "Kc", "Ks", "9h")
        rank, kickers = evaluate_hand(hand)
        assert rank == HAND_TWO_PAIR
        assert kickers[0] == 14  # Aces over Kings
        assert kickers[1] == 13

    def test_three_of_a_kind(self):
        hand = cards("7h", "7d", "7c", "2s", "9h")
        rank, kickers = evaluate_hand(hand)
        assert rank == HAND_THREE_OF_A_KIND
        assert kickers[0] == 7

    def test_straight(self):
        hand = cards("5h", "6d", "7c", "8s", "9h")
        rank, kickers = evaluate_hand(hand)
        assert rank == HAND_STRAIGHT
        assert kickers[0] == 9  # Nine-high straight

    def test_flush(self):
        hand = cards("2h", "5h", "7h", "9h", "Jh")
        rank, kickers = evaluate_hand(hand)
        assert rank == HAND_FLUSH
        assert kickers[0] == 11  # Jack-high flush

    def test_full_house(self):
        hand = cards("Ah", "Ad", "Ac", "Ks", "Kh")
        rank, kickers = evaluate_hand(hand)
        assert rank == HAND_FULL_HOUSE
        assert kickers[0] == 14  # Aces full of Kings
        assert kickers[1] == 13

    def test_four_of_a_kind(self):
        hand = cards("Kh", "Kd", "Kc", "Ks", "2h")
        rank, kickers = evaluate_hand(hand)
        assert rank == HAND_FOUR_OF_A_KIND
        assert kickers[0] == 13  # Quad kings

    def test_straight_flush(self):
        hand = cards("5h", "6h", "7h", "8h", "9h")
        rank, kickers = evaluate_hand(hand)
        assert rank == HAND_STRAIGHT_FLUSH
        assert kickers[0] == 9

    def test_royal_flush(self):
        hand = cards("Th", "Jh", "Qh", "Kh", "Ah")
        rank, kickers = evaluate_hand(hand)
        assert rank == HAND_ROYAL_FLUSH

    def test_best_five_from_seven(self):
        # Player has a royal flush buried in 7 cards
        seven = cards("Th", "Jh", "Qh", "Kh", "Ah", "2c", "3d")
        rank, _ = evaluate_hand(seven)
        assert rank == HAND_ROYAL_FLUSH

    def test_comparison_pair_beats_high_card(self):
        pair_hand = evaluate_hand(cards("Ah", "Ad", "2c", "5s", "9h"))
        high_hand = evaluate_hand(cards("Kh", "Qd", "Jc", "9s", "7h"))
        assert pair_hand > high_hand

    def test_comparison_kicker_tiebreak(self):
        # Two hands both have a pair of aces; one has a K kicker, other a Q
        ace_k = evaluate_hand(cards("Ah", "Ad", "Kc", "5s", "2h"))
        ace_q = evaluate_hand(cards("Ac", "As", "Qc", "5d", "2d"))
        assert ace_k > ace_q


# ---------------------------------------------------------------------------
# test_straight_edge_cases
# ---------------------------------------------------------------------------

class TestStraightEdgeCases:
    def test_wheel(self):
        """A-2-3-4-5 is a straight (wheel), top = 5."""
        hand = cards("Ah", "2d", "3c", "4s", "5h")
        rank, kickers = evaluate_hand(hand)
        assert rank == HAND_STRAIGHT
        assert kickers[0] == 5

    def test_broadway(self):
        """T-J-Q-K-A (broadway) is the highest straight, top = 14."""
        hand = cards("Th", "Jd", "Qc", "Ks", "Ah")
        rank, kickers = evaluate_hand(hand)
        # Broadway is a straight (not a flush here)
        assert rank == HAND_STRAIGHT
        assert kickers[0] == 14

    def test_wheel_vs_six_high(self):
        """A 6-high straight should beat a wheel."""
        wheel = evaluate_hand(cards("Ah", "2d", "3c", "4s", "5h"))
        six_hi = evaluate_hand(cards("2h", "3d", "4c", "5s", "6h"))
        assert six_hi > wheel

    def test_no_fake_straight_wraparound(self):
        """Q-K-A-2-3 is NOT a straight — no wraparound in poker."""
        hand = cards("Qh", "Kd", "Ac", "2s", "3h")
        rank, _ = evaluate_hand(hand)
        assert rank != HAND_STRAIGHT

    def test_straight_flush_wheel(self):
        """A-2-3-4-5 all same suit is a straight flush."""
        hand = cards("Ah", "2h", "3h", "4h", "5h")
        rank, kickers = evaluate_hand(hand)
        assert rank == HAND_STRAIGHT_FLUSH
        assert kickers[0] == 5


# ---------------------------------------------------------------------------
# test_equity_known_scenarios
# ---------------------------------------------------------------------------

class TestEquityKnownScenarios:
    """These tests use large simulations and allow ±5% tolerance."""

    def test_pocket_aces_preflop(self):
        """AA vs 1 random opponent should win ~85% of the time."""
        random.seed(42)
        hole = cards("Ah", "Ad")
        community: list[Card] = []
        win_pct, tie_pct = calculate_equity(
            hole_cards=hole,
            community_cards=community,
            num_opponents=1,
            simulations=10_000,
        )
        equity = win_pct + tie_pct / 2
        assert abs(equity - 0.85) < 0.05, f"AA equity={equity:.3f}, expected ~0.85"

    def test_ak_suited_preflop(self):
        """AKs vs 1 random opponent should win ~67% of the time."""
        random.seed(42)
        hole = cards("Ah", "Kh")
        community: list[Card] = []
        win_pct, tie_pct = calculate_equity(
            hole_cards=hole,
            community_cards=community,
            num_opponents=1,
            simulations=10_000,
        )
        equity = win_pct + tie_pct / 2
        assert abs(equity - 0.67) < 0.05, f"AKs equity={equity:.3f}, expected ~0.67"

    def test_low_pocket_pair_preflop(self):
        """22 vs 1 random opponent should win in the range 50-55%."""
        random.seed(0)
        hole = cards("2h", "2d")
        win_pct, tie_pct = calculate_equity(
            hole_cards=hole,
            community_cards=[],
            num_opponents=1,
            simulations=5_000,
        )
        equity = win_pct + tie_pct / 2
        assert 0.45 < equity < 0.60, f"22 equity={equity:.3f}, expected 45-60%"

    def test_made_flush_vs_opponent(self):
        """Player holds a made flush on the river; should have high equity."""
        random.seed(7)
        # Community is 4 cards; player has Ah,Kh (both hearts); board has 3 hearts
        hole = cards("Ah", "Kh")
        community = cards("Qh", "Jh", "2h", "5d")  # 4-card board with flush made
        win_pct, tie_pct = calculate_equity(
            hole_cards=hole,
            community_cards=community,
            num_opponents=1,
            simulations=5_000,
        )
        equity = win_pct + tie_pct / 2
        assert equity > 0.70, f"Nut flush equity={equity:.3f}, expected >0.70"


# ---------------------------------------------------------------------------
# test_pot_odds
# ---------------------------------------------------------------------------

class TestPotOdds:
    def test_standard_case(self):
        """pot=100, call=50 → required equity = 50/150 ≈ 0.333."""
        po = calculate_pot_odds(100.0, 50.0)
        assert abs(po - (50 / 150)) < 1e-9

    def test_zero_bet(self):
        """Free card / check → pot odds = 0.0."""
        po = calculate_pot_odds(200.0, 0.0)
        assert po == 0.0

    def test_zero_pot_and_bet(self):
        """Both zero → 0.0 (no division by zero)."""
        po = calculate_pot_odds(0.0, 0.0)
        assert po == 0.0

    def test_big_pot(self):
        po = calculate_pot_odds(1000.0, 100.0)
        expected = 100 / 1100
        assert abs(po - expected) < 1e-9


# ---------------------------------------------------------------------------
# test_ev_positive
# ---------------------------------------------------------------------------

class TestEV:
    def test_winning_hand_positive_ev_call(self):
        """A hand with 80% equity facing a small bet should have positive EV to call."""
        ev_call, ev_fold, ev_raise = calculate_ev(0.80, 200.0, 30.0)
        assert ev_call > 0, f"ev_call={ev_call:.2f} should be positive"
        assert ev_fold == 0.0

    def test_losing_hand_negative_ev_call(self):
        """A hand with 10% equity facing a large bet should have negative EV to call."""
        ev_call, ev_fold, ev_raise = calculate_ev(0.10, 100.0, 300.0)
        assert ev_call < 0, f"ev_call={ev_call:.2f} should be negative"

    def test_ev_raise_is_1_5x_call_when_positive(self):
        ev_call, ev_fold, ev_raise = calculate_ev(0.60, 100.0, 50.0)
        assert ev_call > 0, "Precondition: ev_call should be positive"
        assert abs(ev_raise - 1.5 * ev_call) < 1e-9

    def test_ev_raise_is_2x_call_when_negative(self):
        ev_call, ev_fold, ev_raise = calculate_ev(0.10, 100.0, 300.0)
        assert ev_call < 0, "Precondition: ev_call should be negative"
        assert abs(ev_raise - 2.0 * ev_call) < 1e-9

    def test_ev_fold_always_zero(self):
        for win in [0.0, 0.5, 1.0]:
            _, ev_fold, _ = calculate_ev(win, 100.0, 50.0)
            assert ev_fold == 0.0


# ---------------------------------------------------------------------------
# test_analyze_poker
# ---------------------------------------------------------------------------

class TestAnalyzePoker:
    def test_returns_poker_result(self):
        state = PokerState(
            hole_cards=cards("Ah", "Kd"),
            community_cards=cards("Qh", "Jd", "Tc"),
            pot_size=200.0,
            bet_to_call=50.0,
            num_opponents=1,
        )
        result = analyze_poker(state)
        assert isinstance(result, PokerResult)

    def test_result_fields_populated(self):
        state = PokerState(
            hole_cards=cards("Ah", "Ad"),
            community_cards=[],
            pot_size=100.0,
            bet_to_call=20.0,
            num_opponents=1,
        )
        result = analyze_poker(state)
        assert 0.0 <= result.win_pct <= 1.0
        assert 0.0 <= result.tie_pct <= 1.0
        assert 0.0 <= result.actual_equity <= 1.0
        assert 0.0 <= result.required_equity <= 1.0
        assert result.recommended_action in list(PokerAction)

    def test_strong_hand_recommends_raise(self):
        """Royal flush on the board — player holds the nuts."""
        random.seed(1)
        state = PokerState(
            hole_cards=cards("Ah", "Kh"),
            community_cards=cards("Qh", "Jh", "Th"),  # Royal flush on board
            pot_size=500.0,
            bet_to_call=10.0,
            num_opponents=1,
        )
        result = analyze_poker(state)
        # With an unbeatable hand, EV(raise) > EV(call) > EV(fold)
        assert result.recommended_action == PokerAction.RAISE

    def test_weak_hand_recommends_fold(self):
        """72o facing a massive bet should fold."""
        random.seed(2)
        state = PokerState(
            hole_cards=cards("7h", "2d"),
            community_cards=cards("Ac", "Ks", "Qd"),
            pot_size=50.0,
            bet_to_call=500.0,
            num_opponents=1,
        )
        result = analyze_poker(state)
        assert result.recommended_action == PokerAction.FOLD

    def test_actual_equity_formula(self):
        """actual_equity == win_pct + tie_pct / 2."""
        state = PokerState(
            hole_cards=cards("Th", "Jh"),
            community_cards=cards("Qh", "Kh", "2c"),
            pot_size=100.0,
            bet_to_call=25.0,
            num_opponents=1,
        )
        result = analyze_poker(state)
        expected = result.win_pct + result.tie_pct / 2.0
        assert abs(result.actual_equity - expected) < 1e-9
