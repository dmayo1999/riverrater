"""
tests/test_poker_input.py — Tests for the PokerInputDialog.
"""

import pytest

from riverrater.hud.poker_input import PokerInputDialog


@pytest.fixture
def dialog(qtbot):
    """Create a fresh PokerInputDialog for each test."""
    dlg = PokerInputDialog()
    qtbot.addWidget(dlg)
    return dlg


class TestPokerInputDialogInit:
    def test_default_pot_size(self, dialog):
        assert dialog.pot_size == 0.0

    def test_default_bet_to_call(self, dialog):
        assert dialog.bet_to_call == 0.0

    def test_default_num_opponents(self, dialog):
        assert dialog.num_opponents == 1

    def test_window_title(self, dialog):
        assert dialog.windowTitle() == "Poker Input"


class TestPokerInputDialogValidation:
    def test_pot_size_min(self, dialog):
        dialog.pot_size = -10.0
        assert dialog.pot_size >= 0.0

    def test_bet_to_call_min(self, dialog):
        dialog.bet_to_call = -5.0
        assert dialog.bet_to_call >= 0.0

    def test_num_opponents_min(self, dialog):
        dialog.num_opponents = 0
        assert dialog.num_opponents >= 1

    def test_num_opponents_max(self, dialog):
        dialog.num_opponents = 20
        assert dialog.num_opponents <= 9


class TestPokerInputDialogSetValues:
    def test_set_pot_size(self, dialog):
        dialog.pot_size = 150.50
        assert abs(dialog.pot_size - 150.50) < 0.01

    def test_set_bet_to_call(self, dialog):
        dialog.bet_to_call = 25.0
        assert abs(dialog.bet_to_call - 25.0) < 0.01

    def test_set_num_opponents(self, dialog):
        dialog.num_opponents = 5
        assert dialog.num_opponents == 5


class TestPokerInputDialogSignal:
    def test_values_submitted_signal(self, dialog, qtbot):
        received = []
        dialog.values_submitted.connect(
            lambda p, b, n: received.append((p, b, n))
        )
        dialog.pot_size = 200.0
        dialog.bet_to_call = 50.0
        dialog.num_opponents = 3
        dialog._on_apply()
        assert len(received) == 1
        pot, bet, opp = received[0]
        assert abs(pot - 200.0) < 0.01
        assert abs(bet - 50.0) < 0.01
        assert opp == 3
