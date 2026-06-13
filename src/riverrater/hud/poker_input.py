"""
Poker manual input dialog for RiverRater.

Provides a compact dialog for entering pot_size, bet_to_call, and
num_opponents values when automatic detection is not available.

Signals:
    values_submitted(float, float, int)  — emitted with (pot_size,
        bet_to_call, num_opponents) when the user clicks Apply.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design constants — match the ManualCardInput style
# ---------------------------------------------------------------------------
_BG_DIALOG = "rgb(28, 28, 40)"
_CLR_GREEN = "#00C853"
_CLR_RED = "#FF1744"
_CLR_WHITE = "#F0F0F8"
_CLR_MUTED = "#808090"
_CLR_LIGHTGRAY = "#C0C0D0"

MIN_OPPONENTS = 1
MAX_OPPONENTS = 9


class _HRule(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setFixedHeight(1)
        self.setStyleSheet("background-color: rgba(255,255,255,40); border: none;")


class OpponentStepper(QWidget):
    """Compact −/value/+ stepper for active opponent count (1–9)."""

    value_changed = pyqtSignal(int)

    def __init__(
        self,
        value: int = MIN_OPPONENTS,
        *,
        compact: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._value = MIN_OPPONENTS
        self._compact = compact
        self._setup_ui()
        self.set_value(value)

    def _setup_ui(self) -> None:
        btn_size = 22 if self._compact else 26
        font_size = 11 if self._compact else 13

        self.setStyleSheet(
            f"""
            QPushButton {{
                color: {_CLR_WHITE};
                background-color: rgba(40, 40, 58, 255);
                border: 1px solid rgba(100,100,130,180);
                border-radius: 4px;
                font-size: {font_size}px;
                font-weight: 700;
            }}
            QPushButton:hover:enabled {{
                background-color: rgba(80, 80, 110, 240);
            }}
            QPushButton:disabled {{
                color: {_CLR_MUTED};
                background-color: rgba(30, 30, 42, 255);
            }}
            QLabel {{
                color: {_CLR_WHITE};
                font-size: {font_size}px;
                font-weight: 600;
            }}
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._minus_btn = QPushButton("−")
        self._minus_btn.setFixedSize(btn_size, btn_size)
        self._minus_btn.clicked.connect(self._decrement)

        self._value_label = QLabel(str(MIN_OPPONENTS))
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value_label.setFixedWidth(18 if self._compact else 22)

        self._plus_btn = QPushButton("+")
        self._plus_btn.setFixedSize(btn_size, btn_size)
        self._plus_btn.clicked.connect(self._increment)

        layout.addWidget(self._minus_btn)
        layout.addWidget(self._value_label)
        layout.addWidget(self._plus_btn)

    @property
    def value(self) -> int:
        return self._value

    def set_value(self, value: int, *, emit: bool = False) -> None:
        """Set opponent count, clamped to [MIN_OPPONENTS, MAX_OPPONENTS]."""
        clamped = max(MIN_OPPONENTS, min(MAX_OPPONENTS, int(value)))
        changed = clamped != self._value
        self._value = clamped
        self._value_label.setText(str(clamped))
        self._minus_btn.setEnabled(clamped > MIN_OPPONENTS)
        self._plus_btn.setEnabled(clamped < MAX_OPPONENTS)
        if emit and changed:
            self.value_changed.emit(clamped)

    def _increment(self) -> None:
        self.set_value(self._value + 1, emit=True)

    def _decrement(self) -> None:
        self.set_value(self._value - 1, emit=True)


class PokerInputDialog(QDialog):
    """Compact dialog for entering poker pot/bet values manually.

    Signals:
        values_submitted: Emitted with ``(pot_size, bet_to_call, num_opponents)``
            when the user clicks Apply.
    """

    values_submitted = pyqtSignal(float, float, int)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        num_opponents: int = MIN_OPPONENTS,
    ) -> None:
        super().__init__(parent)
        self._initial_opponents = num_opponents
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle("Poker Input")
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setFixedSize(300, 230)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {_BG_DIALOG};
            }}
            QLabel {{
                color: {_CLR_WHITE};
            }}
            QPushButton {{
                color: {_CLR_WHITE};
                border: 1px solid rgba(100,100,130,180);
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: rgba(80, 80, 110, 240);
            }}
            QDoubleSpinBox {{
                color: {_CLR_WHITE};
                background-color: rgba(40, 40, 58, 255);
                border: 1px solid rgba(100,100,130,180);
                border-radius: 4px;
                padding: 4px;
                font-size: 13px;
            }}
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # -- Title -----------------------------------------------------------
        title = QLabel("Poker Input")
        title.setStyleSheet(
            f"color: {_CLR_WHITE}; font-size: 13px; font-weight: 700; "
            "letter-spacing: 1px;"
        )
        root.addWidget(title)
        root.addWidget(_HRule(self))

        # -- Pot Size --------------------------------------------------------
        pot_row = QHBoxLayout()
        pot_lbl = QLabel("Pot Size")
        pot_lbl.setStyleSheet(f"color: {_CLR_LIGHTGRAY}; font-size: 12px;")
        self._pot_spin = QDoubleSpinBox()
        self._pot_spin.setRange(0.0, 1_000_000.0)
        self._pot_spin.setDecimals(2)
        self._pot_spin.setValue(0.0)
        self._pot_spin.setPrefix("$")
        pot_row.addWidget(pot_lbl)
        pot_row.addStretch()
        pot_row.addWidget(self._pot_spin)
        root.addLayout(pot_row)

        # -- Bet to Call -----------------------------------------------------
        bet_row = QHBoxLayout()
        bet_lbl = QLabel("Bet to Call")
        bet_lbl.setStyleSheet(f"color: {_CLR_LIGHTGRAY}; font-size: 12px;")
        self._bet_spin = QDoubleSpinBox()
        self._bet_spin.setRange(0.0, 1_000_000.0)
        self._bet_spin.setDecimals(2)
        self._bet_spin.setValue(0.0)
        self._bet_spin.setPrefix("$")
        bet_row.addWidget(bet_lbl)
        bet_row.addStretch()
        bet_row.addWidget(self._bet_spin)
        root.addLayout(bet_row)

        # -- Num Opponents ---------------------------------------------------
        opp_row = QHBoxLayout()
        opp_lbl = QLabel("Opponents")
        opp_lbl.setStyleSheet(f"color: {_CLR_LIGHTGRAY}; font-size: 12px;")
        self._opp_stepper = OpponentStepper(value=self._initial_opponents)
        opp_row.addWidget(opp_lbl)
        opp_row.addStretch()
        opp_row.addWidget(self._opp_stepper)
        root.addLayout(opp_row)

        root.addWidget(_HRule(self))

        # -- Buttons ---------------------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setFixedSize(80, 28)
        self._apply_btn.setStyleSheet(
            "background-color: rgba(0,120,60,180); font-size: 12px;"
        )
        self._apply_btn.clicked.connect(self._on_apply)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedSize(80, 28)
        self._cancel_btn.setStyleSheet(
            "background-color: rgba(180,20,30,180); font-size: 12px;"
        )
        self._cancel_btn.clicked.connect(self.reject)

        btn_row.addStretch()
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._apply_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Interaction handlers
    # ------------------------------------------------------------------

    def _on_apply(self) -> None:
        """Emit values and accept the dialog."""
        pot_size = self._pot_spin.value()
        bet_to_call = self._bet_spin.value()
        num_opponents = self._opp_stepper.value

        logger.debug(
            "Poker input: pot=%.2f, bet=%.2f, opp=%d",
            pot_size, bet_to_call, num_opponents,
        )
        self.values_submitted.emit(pot_size, bet_to_call, num_opponents)
        self.accept()

    # ------------------------------------------------------------------
    # Public accessors (for testing)
    # ------------------------------------------------------------------

    @property
    def pot_size(self) -> float:
        return self._pot_spin.value()

    @pot_size.setter
    def pot_size(self, value: float) -> None:
        self._pot_spin.setValue(value)

    @property
    def bet_to_call(self) -> float:
        return self._bet_spin.value()

    @bet_to_call.setter
    def bet_to_call(self, value: float) -> None:
        self._bet_spin.setValue(value)

    @property
    def num_opponents(self) -> int:
        return self._opp_stepper.value

    @num_opponents.setter
    def num_opponents(self, value: int) -> None:
        self._opp_stepper.set_value(value)