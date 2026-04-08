"""
tests/test_blackjack_math.py — Comprehensive tests for blackjack_math module.
"""

import pytest

from riverrater.game.state import (
    Card, Rank, Suit, BlackjackState, BlackjackResult, BlackjackAction,
)
from riverrater.game.blackjack_math import (
    hand_value,
    running_count,
    true_count,
    basic_strategy,
    kelly_bet,
    shoe_favorability,
    analyze_blackjack,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def c(s: str) -> Card:
    return Card.from_string(s)


def cards(*strings: str) -> list[Card]:
    return [c(s) for s in strings]


# ---------------------------------------------------------------------------
# test_hand_value
# ---------------------------------------------------------------------------

class TestHandValue:
    def test_hard_20_ten_ten(self):
        total, is_soft = hand_value(cards("Th", "Td"))
        assert total == 20
        assert is_soft is False

    def test_soft_17_ace_six(self):
        total, is_soft = hand_value(cards("Ah", "6d"))
        assert total == 17
        assert is_soft is True

    def test_bust_three_cards(self):
        total, is_soft = hand_value(cards("Th", "Td", "5c"))
        assert total == 25
        assert is_soft is False

    def test_ace_adjustment_21(self):
        """A+A+9 → both aces can't be 11 (that's 31), so one stays 11 → 21."""
        total, is_soft = hand_value(cards("Ah", "Ac", "9d"))
        assert total == 21
        assert is_soft is True  # One ace still counts as 11

    def test_ace_as_one(self):
        """A+T+5 → ace must be 1 to avoid bust (total=16)."""
        total, is_soft = hand_value(cards("Ah", "Td", "5c"))
        assert total == 16
        assert is_soft is False  # Ace forced to be 1

    def test_blackjack(self):
        total, is_soft = hand_value(cards("Ah", "Th"))
        assert total == 21
        assert is_soft is True

    def test_hard_14(self):
        total, is_soft = hand_value(cards("9h", "5d"))
        assert total == 14
        assert is_soft is False

    def test_all_aces(self):
        """Four aces: first=11, rest forced to 1 → 11+1+1+1=14."""
        total, is_soft = hand_value(cards("Ah", "Ad", "Ac", "As"))
        assert total == 14
        assert is_soft is True  # One ace still 11

    def test_soft_18_ace_seven(self):
        total, is_soft = hand_value(cards("Ah", "7d"))
        assert total == 18
        assert is_soft is True


# ---------------------------------------------------------------------------
# test_running_count
# ---------------------------------------------------------------------------

class TestRunningCount:
    def test_low_cards_positive(self):
        """2,3,4,5,6 → +5."""
        count = running_count(cards("2h", "3d", "4c", "5s", "6h"))
        assert count == 5

    def test_high_cards_negative(self):
        """T,J,Q,K,A → -5."""
        count = running_count(cards("Th", "Jd", "Qc", "Ks", "Ah"))
        assert count == -5

    def test_neutral_cards_zero(self):
        """7,8,9 → 0."""
        count = running_count(cards("7h", "8d", "9c"))
        assert count == 0

    def test_mixed_deck(self):
        """2+A = +1-1 = 0."""
        count = running_count(cards("2h", "Ah"))
        assert count == 0

    def test_empty(self):
        count = running_count([])
        assert count == 0

    def test_full_shoe_neutral(self):
        """A complete 52-card deck should sum to 0."""
        from riverrater.game.state import FULL_DECK
        count = running_count(FULL_DECK)
        assert count == 0


# ---------------------------------------------------------------------------
# test_true_count
# ---------------------------------------------------------------------------

class TestTrueCount:
    def test_standard(self):
        """running=4, 4 decks remaining → TC = 1.0."""
        tc = true_count(4, 4.0)
        assert abs(tc - 1.0) < 1e-9

    def test_clamp_minimum(self):
        """decks_remaining < 0.5 is clamped to 0.5."""
        tc_clamped = true_count(2, 0.25)
        tc_half = true_count(2, 0.5)
        assert abs(tc_clamped - tc_half) < 1e-9
        assert abs(tc_clamped - 4.0) < 1e-9

    def test_negative_running(self):
        tc = true_count(-6, 3.0)
        assert abs(tc - (-2.0)) < 1e-9

    def test_zero_running(self):
        tc = true_count(0, 2.0)
        assert tc == 0.0


# ---------------------------------------------------------------------------
# test_basic_strategy
# ---------------------------------------------------------------------------

class TestBasicStrategy:
    def test_hard_16_vs_10_surrender(self):
        """Hard 16 vs dealer 10 → Surrender."""
        action = basic_strategy(cards("9h", "7d"), c("Th"))
        assert action == BlackjackAction.SURRENDER

    def test_hard_16_vs_7_hit(self):
        """Hard 16 vs dealer 7 → Hit."""
        action = basic_strategy(cards("9h", "7d"), c("7c"))
        assert action == BlackjackAction.HIT

    def test_hard_16_vs_6_stand(self):
        """Hard 16 vs dealer 6 → Stand."""
        action = basic_strategy(cards("9h", "7d"), c("6c"))
        assert action == BlackjackAction.STAND

    def test_hard_11_vs_6_double(self):
        """Hard 11 vs dealer 6 → Double."""
        action = basic_strategy(cards("5h", "6d"), c("6c"))
        assert action == BlackjackAction.DOUBLE

    def test_hard_11_vs_ace_hit(self):
        """Hard 11 vs dealer A → Hit (not double)."""
        action = basic_strategy(cards("5h", "6d"), c("Ah"))
        assert action == BlackjackAction.HIT

    def test_pair_eights_vs_anything_split(self):
        """Pair of 8s vs any dealer upcard → Split."""
        for dealer_rank in ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]:
            action = basic_strategy(
                cards("8h", "8d"),
                Card.from_string(f"{dealer_rank}c"),
            )
            assert action == BlackjackAction.SPLIT, (
                f"8,8 vs {dealer_rank}: expected SPLIT, got {action}"
            )

    def test_pair_aces_vs_anything_split(self):
        """Pair of aces vs any dealer upcard → Split."""
        for dealer_rank in ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]:
            action = basic_strategy(
                cards("Ah", "Ad"),
                Card.from_string(f"{dealer_rank}c"),
            )
            assert action == BlackjackAction.SPLIT, (
                f"A,A vs {dealer_rank}: expected SPLIT, got {action}"
            )

    def test_pair_tens_stand(self):
        """Pair of tens → Stand (never split)."""
        action = basic_strategy(cards("Th", "Td"), c("6c"))
        assert action == BlackjackAction.STAND

    def test_soft_18_vs_9_hit(self):
        """Soft 18 (A+7) vs dealer 9 → Hit."""
        action = basic_strategy(cards("Ah", "7d"), c("9c"))
        assert action == BlackjackAction.HIT

    def test_soft_18_vs_2_stand(self):
        """Soft 18 (A+7) vs dealer 2 → Stand."""
        action = basic_strategy(cards("Ah", "7d"), c("2c"))
        assert action == BlackjackAction.STAND

    def test_soft_18_vs_6_double(self):
        """Soft 18 (A+7) vs dealer 6 → Double."""
        action = basic_strategy(cards("Ah", "7d"), c("6c"))
        assert action == BlackjackAction.DOUBLE

    def test_hard_20_stand(self):
        """Hard 20 vs any dealer upcard → Stand."""
        for dealer_rank in ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"]:
            action = basic_strategy(
                cards("Th", "Td"),
                Card.from_string(f"{dealer_rank}c"),
            )
            assert action == BlackjackAction.STAND

    def test_hard_12_vs_4_stand(self):
        """Hard 12 vs dealer 4 → Stand."""
        action = basic_strategy(cards("7h", "5d"), c("4c"))
        assert action == BlackjackAction.STAND

    def test_hard_12_vs_2_hit(self):
        """Hard 12 vs dealer 2 → Hit."""
        action = basic_strategy(cards("7h", "5d"), c("2c"))
        assert action == BlackjackAction.HIT

    def test_hard_10_vs_9_double(self):
        """Hard 10 vs dealer 9 → Double."""
        action = basic_strategy(cards("6h", "4d"), c("9c"))
        assert action == BlackjackAction.DOUBLE

    def test_hard_10_vs_ace_hit(self):
        """Hard 10 vs dealer A → Hit."""
        action = basic_strategy(cards("6h", "4d"), c("Ah"))
        assert action == BlackjackAction.HIT

    def test_soft_17_vs_2_double(self):
        """Soft 17 (A+6) vs dealer 2 → Double."""
        action = basic_strategy(cards("Ah", "6d"), c("2c"))
        assert action == BlackjackAction.DOUBLE

    def test_pair_2s_vs_7_split(self):
        """Pair of 2s vs dealer 7 → Split."""
        action = basic_strategy(cards("2h", "2d"), c("7c"))
        assert action == BlackjackAction.SPLIT

    def test_pair_2s_vs_8_hit(self):
        """Pair of 2s vs dealer 8 → Hit (not split)."""
        action = basic_strategy(cards("2h", "2d"), c("8c"))
        assert action == BlackjackAction.HIT

    def test_hard_15_vs_10_surrender(self):
        """Hard 15 vs dealer 10 → Surrender."""
        action = basic_strategy(cards("Th", "5d"), c("Tc"))
        assert action == BlackjackAction.SURRENDER


# ---------------------------------------------------------------------------
# test_kelly_bet
# ---------------------------------------------------------------------------

class TestKellyBet:
    def test_positive_count_above_min(self):
        """TC=+3, min=10, max=500, bankroll=5000 → bet should be > min_bet."""
        bet = kelly_bet(3.0, 10.0, 500.0, 5000.0)
        assert bet > 10.0
        assert bet <= 500.0

    def test_negative_count_returns_min(self):
        """TC=-2 → always return min_bet."""
        bet = kelly_bet(-2.0, 10.0, 500.0, 5000.0)
        assert bet == 10.0

    def test_zero_count_returns_min(self):
        """TC=0 → edge=(0-1)*0.005 < 0 → min_bet."""
        bet = kelly_bet(0.0, 10.0, 500.0, 5000.0)
        assert bet == 10.0

    def test_tc_exactly_one_returns_min(self):
        """TC=1 → edge=0 exactly → min_bet."""
        bet = kelly_bet(1.0, 10.0, 500.0, 5000.0)
        assert bet == 10.0

    def test_very_high_count_capped_at_max(self):
        """TC=+50, tiny bankroll would overshoot — should be clamped to max_bet."""
        bet = kelly_bet(50.0, 10.0, 500.0, 5_000_000.0)
        assert bet == 500.0

    def test_rounding_to_min_increment(self):
        """Bet should be a multiple of min_bet."""
        bet = kelly_bet(5.0, 25.0, 1000.0, 10000.0)
        assert bet % 25.0 == 0 or abs(bet % 25.0 - 25.0) < 1e-9

    def test_bet_within_bounds(self):
        for tc_val in [-5.0, 0.0, 1.5, 3.0, 8.0]:
            bet = kelly_bet(tc_val, 10.0, 500.0, 5000.0)
            assert 10.0 <= bet <= 500.0, f"TC={tc_val}: bet={bet} out of bounds"


# ---------------------------------------------------------------------------
# test_shoe_favorability
# ---------------------------------------------------------------------------

class TestShoeFavorability:
    def test_tc_zero(self):
        """TC=0 → 0.3."""
        fav = shoe_favorability(0.0)
        assert abs(fav - 0.3) < 1e-9

    def test_tc_plus_5(self):
        """TC=+5 → 0.9."""
        fav = shoe_favorability(5.0)
        assert abs(fav - 0.9) < 1e-9

    def test_tc_plus_2(self):
        """TC=+2 → 0.6."""
        fav = shoe_favorability(2.0)
        assert abs(fav - 0.6) < 1e-9

    def test_tc_minus_2(self):
        """TC=-2 → 0.0."""
        fav = shoe_favorability(-2.0)
        assert abs(fav - 0.0) < 1e-9

    def test_tc_plus_7(self):
        """TC=+7 → 1.0."""
        fav = shoe_favorability(7.0)
        assert abs(fav - 1.0) < 1e-9

    def test_clamp_low(self):
        """TC=-10 → 0.0 (below minimum breakpoint)."""
        fav = shoe_favorability(-10.0)
        assert fav == 0.0

    def test_clamp_high(self):
        """TC=+20 → 1.0 (above maximum breakpoint)."""
        fav = shoe_favorability(20.0)
        assert fav == 1.0

    def test_interpolation_between_0_and_2(self):
        """TC=+1 → midpoint between 0.3 and 0.6 = 0.45."""
        fav = shoe_favorability(1.0)
        assert abs(fav - 0.45) < 1e-9

    def test_monotone_increasing(self):
        """Favorability should be non-decreasing as TC increases."""
        tcs = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0]
        favs = [shoe_favorability(tc) for tc in tcs]
        for i in range(len(favs) - 1):
            assert favs[i] <= favs[i + 1], (
                f"Not monotone at TC={tcs[i]}: fav={favs[i]:.3f} > fav={favs[i+1]:.3f}"
            )


# ---------------------------------------------------------------------------
# test_analyze_blackjack
# ---------------------------------------------------------------------------

class TestAnalyzeBlackjack:
    def test_returns_blackjack_result(self):
        state = BlackjackState(
            player_hand=cards("Ah", "7d"),
            dealer_upcard=c("9c"),
            cards_seen=cards("Ah", "7d", "9c"),
            num_decks=6,
        )
        result = analyze_blackjack(state)
        assert isinstance(result, BlackjackResult)

    def test_fields_populated(self):
        state = BlackjackState(
            player_hand=cards("Th", "Td"),
            dealer_upcard=c("6c"),
            cards_seen=cards("Th", "Td", "6c"),
            num_decks=6,
        )
        result = analyze_blackjack(state)
        assert result.hand_total == 20
        assert result.is_soft is False
        assert result.recommended_action is not None
        assert 0.0 <= result.shoe_favorability <= 1.0

    def test_running_count_propagated(self):
        """Cards seen: 2,3 → running count = +2."""
        state = BlackjackState(
            player_hand=cards("2h", "3d"),
            dealer_upcard=c("5c"),
            cards_seen=cards("2h", "3d"),
            num_decks=6,
        )
        result = analyze_blackjack(state)
        assert result.running_count == 2

    def test_recommended_bet_within_bounds(self):
        state = BlackjackState(
            player_hand=cards("Ah", "6d"),
            dealer_upcard=c("3c"),
            cards_seen=[],
            num_decks=6,
        )
        result = analyze_blackjack(state)
        assert 10.0 <= result.recommended_bet <= 500.0

    def test_hard_20_recommends_stand(self):
        state = BlackjackState(
            player_hand=cards("Th", "Td"),
            dealer_upcard=c("7c"),
            cards_seen=cards("Th", "Td", "7c"),
            num_decks=6,
        )
        result = analyze_blackjack(state)
        assert result.recommended_action == BlackjackAction.STAND

    def test_soft_hand_flag(self):
        """A+6 hand should be flagged as soft."""
        state = BlackjackState(
            player_hand=cards("Ah", "6d"),
            dealer_upcard=c("5c"),
            cards_seen=cards("Ah", "6d", "5c"),
            num_decks=6,
        )
        result = analyze_blackjack(state)
        assert result.is_soft is True
        assert result.hand_total == 17

    def test_positive_shoe_with_many_low_cards(self):
        """Shoe loaded with low cards → high running/true count → higher favorability."""
        low_cards = [Card.from_string(f"{r}h") for r in ["2", "3", "4", "5", "6"] * 10]
        state = BlackjackState(
            player_hand=cards("8h", "9d"),
            dealer_upcard=c("7c"),
            cards_seen=low_cards,
            num_decks=6,
        )
        result = analyze_blackjack(state)
        assert result.running_count > 0
        assert result.shoe_favorability > 0.3

    def test_true_count_calculation(self):
        """With 0 cards seen from a 6-deck shoe, TC ≈ 0."""
        state = BlackjackState(
            player_hand=cards("Kh", "5d"),
            dealer_upcard=c("6c"),
            cards_seen=[],
            num_decks=6,
        )
        result = analyze_blackjack(state)
        assert result.true_count == 0.0
