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
    QSpinBox,
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


class _HRule(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setFixedHeight(1)
        self.setStyleSheet("background-color: rgba(255,255,255,40); border: none;")


class PokerInputDialog(QDialog):
    """Compact dialog for entering poker pot/bet values manually.

    Signals:
        values_submitted: Emitted with ``(pot_size, bet_to_call, num_opponents)``
            when the user clicks Apply.
    """

    values_submitted = pyqtSignal(float, float, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
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
            QDoubleSpinBox, QSpinBox {{
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
        self._opp_spin = QSpinBox()
        self._opp_spin.setRange(1, 9)
        self._opp_spin.setValue(1)
        opp_row.addWidget(opp_lbl)
        opp_row.addStretch()
        opp_row.addWidget(self._opp_spin)
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
        num_opponents = self._opp_spin.value()

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
        return self._opp_spin.value()

    @num_opponents.setter
    def num_opponents(self, value: int) -> None:
        self._opp_spin.setValue(value)
