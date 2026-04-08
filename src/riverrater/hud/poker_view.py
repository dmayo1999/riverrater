"""
Poker HUD widget for RiverRater.

Displays win %, tie %, required equity, EV values, and a recommendation pill.
All styling is QSS-based — no custom paint events.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

try:
    from riverrater.game.state import PokerAction, PokerResult
except ImportError:
    from dataclasses import dataclass
    from enum import Enum

    class PokerAction(Enum):  # type: ignore[no-redef]
        FOLD = "fold"
        CALL = "call"
        RAISE = "raise"

    @dataclass
    class PokerResult:  # type: ignore[no-redef]
        win_pct: float = 0.0
        tie_pct: float = 0.0
        required_equity: float = 0.0
        actual_equity: float = 0.0
        ev_call: float = 0.0
        ev_fold: float = 0.0
        ev_raise: float = 0.0
        recommended_action: Optional[PokerAction] = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------
_BG_DARK = "rgba(20, 20, 30, 210)"
_CLR_GREEN = "#00C853"
_CLR_RED = "#FF1744"
_CLR_YELLOW = "#FFD600"
_CLR_BLUE = "#2979FF"
_CLR_MUTED = "#808090"
_CLR_WHITE = "#F0F0F8"
_CLR_LIGHTGRAY = "#C0C0D0"

_MONO_FONT = "Menlo, 'Courier New', Courier, monospace"


class _HRule(QFrame):
    """Thin horizontal divider line."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setFixedHeight(1)
        self.setStyleSheet("background-color: rgba(255,255,255,40); border: none;")


class PokerView(QWidget):
    """Compact poker analysis panel displayed inside the HUD overlay.

    Sections (top to bottom):
    - Title bar: "♠ POKER" + status dot
    - Win % (large, colour-coded vs required equity)
    - Tie %, Required Equity
    - EV Call / EV Fold
    - Recommendation pill (CALL ✓ / FOLD ✗ / RAISE)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()
        self._show_no_data()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setObjectName("PokerView")
        self.setStyleSheet(
            f"""
            QWidget#PokerView {{
                background-color: {_BG_DARK};
                border-radius: 12px;
            }}
            """
        )
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(260)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 12)
        root.setSpacing(6)

        # -- Title bar -------------------------------------------------------
        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        self._title_label = QLabel("♠ POKER")
        self._title_label.setStyleSheet(
            f"""
            color: {_CLR_WHITE};
            font-size: 11px;
            font-variant: small-caps;
            font-weight: 700;
            letter-spacing: 2px;
            """
        )

        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(
            f"color: {_CLR_MUTED}; font-size: 9px;"
        )
        self._status_dot.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        title_row.addWidget(self._title_label)
        title_row.addStretch()
        title_row.addWidget(self._status_dot)
        root.addLayout(title_row)
        root.addWidget(_HRule(self))

        # -- No-data placeholder ---------------------------------------------
        self._no_data_label = QLabel("Waiting for cards...")
        self._no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_data_label.setStyleSheet(
            f"color: {_CLR_MUTED}; font-size: 13px; padding: 20px 0px;"
        )
        root.addWidget(self._no_data_label)

        # -- Data container --------------------------------------------------
        self._data_widget = QWidget(self)
        data_layout = QVBoxLayout(self._data_widget)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(5)

        # Win %
        win_row = QHBoxLayout()
        self._win_label = QLabel("WIN")
        self._win_label.setStyleSheet(
            f"color: {_CLR_LIGHTGRAY}; font-size: 10px; font-weight: 600;"
        )
        self._win_value = QLabel("—")
        self._win_value.setStyleSheet(
            f"color: {_CLR_WHITE}; font-family: {_MONO_FONT}; font-size: 30px; font-weight: 700;"
        )
        self._win_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        win_row.addWidget(self._win_label, alignment=Qt.AlignmentFlag.AlignBottom)
        win_row.addStretch()
        win_row.addWidget(self._win_value)
        data_layout.addLayout(win_row)

        # Tie % + Required Equity (compact row)
        sub_row = QHBoxLayout()
        self._tie_label = QLabel("TIE —")
        self._tie_label.setStyleSheet(
            f"color: {_CLR_MUTED}; font-family: {_MONO_FONT}; font-size: 11px;"
        )
        self._req_label = QLabel("REQ —")
        self._req_label.setStyleSheet(
            f"color: {_CLR_MUTED}; font-family: {_MONO_FONT}; font-size: 11px;"
        )
        self._req_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        sub_row.addWidget(self._tie_label)
        sub_row.addStretch()
        sub_row.addWidget(self._req_label)
        data_layout.addLayout(sub_row)

        data_layout.addWidget(_HRule(self))

        # EV section
        ev_call_row = QHBoxLayout()
        ev_call_lbl = QLabel("EV Call")
        ev_call_lbl.setStyleSheet(f"color: {_CLR_LIGHTGRAY}; font-size: 11px;")
        self._ev_call_value = QLabel("—")
        self._ev_call_value.setStyleSheet(
            f"color: {_CLR_WHITE}; font-family: {_MONO_FONT}; font-size: 13px; font-weight: 600;"
        )
        self._ev_call_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        ev_call_row.addWidget(ev_call_lbl)
        ev_call_row.addStretch()
        ev_call_row.addWidget(self._ev_call_value)
        data_layout.addLayout(ev_call_row)

        ev_fold_row = QHBoxLayout()
        ev_fold_lbl = QLabel("EV Fold")
        ev_fold_lbl.setStyleSheet(f"color: {_CLR_LIGHTGRAY}; font-size: 11px;")
        self._ev_fold_value = QLabel("—")
        self._ev_fold_value.setStyleSheet(
            f"color: {_CLR_WHITE}; font-family: {_MONO_FONT}; font-size: 13px; font-weight: 600;"
        )
        self._ev_fold_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        ev_fold_row.addWidget(ev_fold_lbl)
        ev_fold_row.addStretch()
        ev_fold_row.addWidget(self._ev_fold_value)
        data_layout.addLayout(ev_fold_row)

        data_layout.addWidget(_HRule(self))

        # Recommendation pill
        self._rec_label = QLabel("—")
        self._rec_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rec_label.setFixedHeight(36)
        self._rec_label.setStyleSheet(
            f"""
            color: {_CLR_WHITE};
            font-family: {_MONO_FONT};
            font-size: 16px;
            font-weight: 700;
            border-radius: 8px;
            background-color: rgba(80, 80, 100, 180);
            padding: 0px 16px;
            """
        )
        data_layout.addWidget(self._rec_label)

        root.addWidget(self._data_widget)
        self._data_widget.hide()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, result: PokerResult) -> None:  # type: ignore[override]
        """Refresh all labels from a :class:`PokerResult`.

        Args:
            result: Latest analysis result from the poker math engine.
        """
        has_data = result.win_pct > 0.0 or result.required_equity > 0.0

        if not has_data:
            self._show_no_data()
            return

        self._show_data()

        # Status dot → green when live
        self._status_dot.setStyleSheet(
            f"color: {_CLR_GREEN}; font-size: 9px;"
        )

        # Win % — colour vs required equity
        win_color = (
            _CLR_GREEN
            if result.actual_equity >= result.required_equity
            else _CLR_RED
        )
        self._win_value.setText(f"{result.win_pct * 100:.1f}%")
        self._win_value.setStyleSheet(
            f"color: {win_color}; font-family: {_MONO_FONT}; font-size: 30px; font-weight: 700;"
        )

        # Tie + Required equity
        self._tie_label.setText(f"TIE {result.tie_pct * 100:.1f}%")
        self._req_label.setText(f"REQ: {result.required_equity * 100:.1f}%")

        # EV values
        self._ev_call_value.setText(self._fmt_ev(result.ev_call))
        self._ev_call_value.setStyleSheet(
            f"color: {_CLR_GREEN if result.ev_call >= 0 else _CLR_RED}; "
            f"font-family: {_MONO_FONT}; font-size: 13px; font-weight: 600;"
        )
        self._ev_fold_value.setText(self._fmt_ev(result.ev_fold))
        self._ev_fold_value.setStyleSheet(
            f"color: {_CLR_GREEN if result.ev_fold >= 0 else _CLR_RED}; "
            f"font-family: {_MONO_FONT}; font-size: 13px; font-weight: 600;"
        )

        # Recommendation
        if result.recommended_action is None:
            self._rec_label.setText("—")
            self._rec_label.setStyleSheet(
                self._rec_style("rgba(80,80,100,180)")
            )
        elif result.recommended_action == PokerAction.CALL:
            self._rec_label.setText("CALL ✓")
            self._rec_label.setStyleSheet(self._rec_style("rgba(0,150,60,200)"))
        elif result.recommended_action == PokerAction.FOLD:
            self._rec_label.setText("FOLD ✗")
            self._rec_label.setStyleSheet(self._rec_style("rgba(200,20,30,200)"))
        elif result.recommended_action == PokerAction.RAISE:
            self._rec_label.setText("RAISE ↑")
            self._rec_label.setStyleSheet(self._rec_style("rgba(0,120,210,200)"))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _show_no_data(self) -> None:
        self._status_dot.setStyleSheet(f"color: {_CLR_MUTED}; font-size: 9px;")
        self._no_data_label.show()
        self._data_widget.hide()

    def _show_data(self) -> None:
        self._no_data_label.hide()
        self._data_widget.show()

    @staticmethod
    def _fmt_ev(value: float) -> str:
        sign = "+" if value >= 0 else ""
        return f"{sign}${value:.2f}"

    @staticmethod
    def _rec_style(bg: str) -> str:
        return (
            f"color: {_CLR_WHITE}; "
            f"font-family: {_MONO_FONT}; "
            "font-size: 16px; font-weight: 700; "
            "border-radius: 8px; "
            f"background-color: {bg}; "
            "padding: 0px 16px;"
        )
