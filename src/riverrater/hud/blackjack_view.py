"""
Blackjack HUD widget for RiverRater.

Displays hand total, strategy action, running/true count, heat meter, and
recommended bet size.  All styling is QSS-based — no custom paint events.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

try:
    from riverrater.game.state import BlackjackAction, BlackjackResult
except ImportError:
    from dataclasses import dataclass
    from enum import Enum

    class BlackjackAction(Enum):  # type: ignore[no-redef]
        HIT = "Hit"
        STAND = "Stand"
        DOUBLE = "Double"
        SPLIT = "Split"
        SURRENDER = "Surrender"

    @dataclass
    class BlackjackResult:  # type: ignore[no-redef]
        running_count: int = 0
        true_count: float = 0.0
        recommended_action: Optional[BlackjackAction] = None
        recommended_bet: float = 0.0
        shoe_favorability: float = 0.0
        hand_total: int = 0
        is_soft: bool = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------
_BG_DARK = "rgba(20, 20, 30, 210)"
_CLR_GREEN = "#00C853"
_CLR_RED = "#FF1744"
_CLR_YELLOW = "#FFD600"
_CLR_BLUE = "#2979FF"
_CLR_ORANGE = "#FF6D00"
_CLR_MUTED = "#808090"
_CLR_WHITE = "#F0F0F8"
_CLR_LIGHTGRAY = "#C0C0D0"

_MONO_FONT = "Menlo, 'Courier New', Courier, monospace"

# Action colour mapping
_ACTION_STYLES: dict[str, tuple[str, str]] = {
    "Hit":       (_CLR_YELLOW,  "rgba(120,100,0,180)"),
    "Stand":     (_CLR_GREEN,   "rgba(0,140,60,180)"),
    "Double":    (_CLR_GREEN,   "rgba(0,100,160,200)"),
    "Split":     (_CLR_BLUE,    "rgba(0,80,180,180)"),
    "Surrender": (_CLR_RED,     "rgba(180,10,20,180)"),
}


class _HRule(QFrame):
    """Thin horizontal divider line."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setFixedHeight(1)
        self.setStyleSheet("background-color: rgba(255,255,255,40); border: none;")


class BlackjackView(QWidget):
    """Compact blackjack analysis panel displayed inside the HUD overlay.

    Sections (top to bottom):
    - Title bar: "♣ BLACKJACK" + status dot
    - Hand total (large) + SOFT/HARD label
    - Strategy action pill (HIT / STAND / DOUBLE / SPLIT / SURRENDER)
    - Running count + True count
    - Shoe heat meter (QProgressBar)
    - Recommended bet
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()
        self._show_no_data()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setObjectName("BlackjackView")
        self.setStyleSheet(
            f"""
            QWidget#BlackjackView {{
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

        self._title_label = QLabel("♣ BLACKJACK")
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
        self._status_dot.setStyleSheet(f"color: {_CLR_MUTED}; font-size: 9px;")
        self._status_dot.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        title_row.addWidget(self._title_label)
        title_row.addStretch()
        title_row.addWidget(self._status_dot)
        root.addLayout(title_row)
        root.addWidget(_HRule(self))

        # -- No-data placeholder ---------------------------------------------
        self._no_data_label = QLabel("Deal cards to begin...")
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

        # Hand total row
        hand_row = QHBoxLayout()
        self._total_value = QLabel("—")
        self._total_value.setStyleSheet(
            f"color: {_CLR_WHITE}; font-family: {_MONO_FONT}; "
            "font-size: 42px; font-weight: 700;"
        )
        self._soft_label = QLabel("")
        self._soft_label.setStyleSheet(
            f"color: {_CLR_MUTED}; font-size: 11px; font-weight: 600;"
        )
        self._soft_label.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        hand_row.addWidget(self._total_value)
        hand_row.addWidget(self._soft_label, alignment=Qt.AlignmentFlag.AlignBottom)
        hand_row.addStretch()
        data_layout.addLayout(hand_row)

        # Action pill
        self._action_label = QLabel("—")
        self._action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._action_label.setFixedHeight(40)
        self._action_label.setStyleSheet(self._action_style_default())
        data_layout.addWidget(self._action_label)

        data_layout.addWidget(_HRule(self))

        # Count row
        count_row = QHBoxLayout()
        self._rc_label = QLabel("RC: —")
        self._rc_label.setStyleSheet(
            f"color: {_CLR_LIGHTGRAY}; font-family: {_MONO_FONT}; font-size: 12px;"
        )
        self._tc_label = QLabel("TC: —")
        self._tc_label.setStyleSheet(
            f"color: {_CLR_LIGHTGRAY}; font-family: {_MONO_FONT}; font-size: 12px;"
        )
        self._tc_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        count_row.addWidget(self._rc_label)
        count_row.addStretch()
        count_row.addWidget(self._tc_label)
        data_layout.addLayout(count_row)

        # Heat meter label
        heat_lbl = QLabel("SHOE HEAT")
        heat_lbl.setStyleSheet(
            f"color: {_CLR_MUTED}; font-size: 9px; font-weight: 600; "
            "letter-spacing: 1px; margin-top: 2px;"
        )
        data_layout.addWidget(heat_lbl)

        # Heat meter bar
        self._heat_bar = QProgressBar()
        self._heat_bar.setRange(0, 100)
        self._heat_bar.setValue(0)
        self._heat_bar.setTextVisible(False)
        self._heat_bar.setFixedHeight(8)
        self._heat_bar.setStyleSheet(
            """
            QProgressBar {
                background-color: rgba(40, 40, 60, 180);
                border-radius: 4px;
                border: none;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0.0  #2979FF,
                    stop:0.45 #00C853,
                    stop:0.75 #FFD600,
                    stop:1.0  #FF1744
                );
            }
            """
        )
        data_layout.addWidget(self._heat_bar)

        data_layout.addWidget(_HRule(self))

        # Recommended bet
        bet_row = QHBoxLayout()
        bet_lbl = QLabel("BET")
        bet_lbl.setStyleSheet(
            f"color: {_CLR_LIGHTGRAY}; font-size: 10px; font-weight: 600;"
        )
        self._bet_value = QLabel("$—")
        self._bet_value.setStyleSheet(
            f"color: {_CLR_GREEN}; font-family: {_MONO_FONT}; "
            "font-size: 22px; font-weight: 700;"
        )
        self._bet_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bet_row.addWidget(bet_lbl, alignment=Qt.AlignmentFlag.AlignBottom)
        bet_row.addStretch()
        bet_row.addWidget(self._bet_value)
        data_layout.addLayout(bet_row)

        root.addWidget(self._data_widget)
        self._data_widget.hide()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, result: BlackjackResult) -> None:  # type: ignore[override]
        """Refresh all labels from a :class:`BlackjackResult`.

        Args:
            result: Latest analysis result from the blackjack math engine.
        """
        has_data = result.hand_total > 0 or result.running_count != 0

        if not has_data:
            self._show_no_data()
            return

        self._show_data()

        # Status dot → green when active
        self._status_dot.setStyleSheet(f"color: {_CLR_GREEN}; font-size: 9px;")

        # Hand total
        self._total_value.setText(str(result.hand_total) if result.hand_total > 0 else "—")
        self._soft_label.setText("SOFT" if result.is_soft else "HARD")

        # Strategy action
        if result.recommended_action is not None:
            action_name = result.recommended_action.value
            fg, bg = _ACTION_STYLES.get(action_name, (_CLR_WHITE, "rgba(80,80,100,180)"))
            self._action_label.setText(action_name.upper())
            self._action_label.setStyleSheet(
                f"color: {fg}; "
                f"font-family: {_MONO_FONT}; "
                "font-size: 18px; font-weight: 700; "
                "border-radius: 8px; "
                f"background-color: {bg}; "
                "padding: 0px 16px;"
            )
        else:
            self._action_label.setText("—")
            self._action_label.setStyleSheet(self._action_style_default())

        # Running count
        rc = result.running_count
        rc_sign = "+" if rc >= 0 else ""
        self._rc_label.setText(f"RC: {rc_sign}{rc}")

        # True count — colour by value
        tc = result.true_count
        tc_sign = "+" if tc >= 0 else ""
        tc_color = self._true_count_color(tc)
        self._tc_label.setText(f"TC: {tc_sign}{tc:.1f}")
        self._tc_label.setStyleSheet(
            f"color: {tc_color}; font-family: {_MONO_FONT}; font-size: 12px;"
        )

        # Heat meter
        heat_pct = int(max(0.0, min(1.0, result.shoe_favorability)) * 100)
        self._heat_bar.setValue(heat_pct)

        # Recommended bet
        if result.recommended_bet > 0:
            self._bet_value.setText(f"${result.recommended_bet:.0f}")
        else:
            self._bet_value.setText("$—")

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
    def _true_count_color(tc: float) -> str:
        if tc > 4:
            return _CLR_RED
        if tc > 1:
            return _CLR_GREEN
        if tc > -1:
            return _CLR_LIGHTGRAY
        return _CLR_BLUE

    @staticmethod
    def _action_style_default() -> str:
        return (
            f"color: {_CLR_WHITE}; "
            f"font-family: {_MONO_FONT}; "
            "font-size: 18px; font-weight: 700; "
            "border-radius: 8px; "
            "background-color: rgba(80,80,100,180); "
            "padding: 0px 16px;"
        )
