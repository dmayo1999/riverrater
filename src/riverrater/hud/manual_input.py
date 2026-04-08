"""
Manual card input dialog for RiverRater.

Provides a compact, keyboard-friendly UI for entering cards into the
blackjack state when automatic vision detection is not available.

Signals:
    card_added(str, str)  — emitted with (card_string, target) whenever the
        user completes a card selection.  ``card_string`` is e.g. ``"Ah"``,
        ``target`` is one of ``"player"``, ``"dealer"``, or ``"seen"``.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design constants — intentionally slightly lighter than the transparent HUD
# so the dialog looks opaque and readable.
# ---------------------------------------------------------------------------
_BG_DIALOG = "rgb(28, 28, 40)"
_BG_SECTION = "rgba(40, 40, 58, 255)"
_BG_BTN_RANK = "rgba(50, 50, 70, 240)"
_BG_BTN_SUIT = "rgba(45, 45, 65, 240)"
_BG_BTN_ACTIVE = "rgba(0, 150, 83, 220)"
_CLR_GREEN = "#00C853"
_CLR_RED = "#FF1744"
_CLR_YELLOW = "#FFD600"
_CLR_BLUE = "#2979FF"
_CLR_WHITE = "#F0F0F8"
_CLR_MUTED = "#808090"
_CLR_LIGHTGRAY = "#C0C0D0"
_MONO_FONT = "Menlo, 'Courier New', Courier, monospace"

_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
_SUITS = [("♥", "h", "#FF1744"), ("♦", "d", "#FF6D00"), ("♣", "c", "#C0C0D0"), ("♠", "s", "#C0C0D0")]

_TARGETS = [
    ("Player Card", "player"),
    ("Dealer Upcard", "dealer"),
    ("Dealt Card", "seen"),
]


class _HRule(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setFixedHeight(1)
        self.setStyleSheet("background-color: rgba(255,255,255,40); border: none;")


class ManualCardInput(QDialog):
    """Compact dialog for entering cards manually.

    Workflow:
    1. User clicks a rank button (highlights selection).
    2. User clicks a suit button → dialog emits ``card_added`` and resets
       rank selection.
    3. Target mode cycles between Player / Dealer / Seen via the toggle button.

    Signals:
        card_added: Emitted with ``(card_str, target)`` where ``card_str`` is
            e.g. ``"Ah"`` and ``target`` is ``"player"``, ``"dealer"``, or
            ``"seen"``.
    """

    card_added = pyqtSignal(str, str)  # (card_str, target)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected_rank: Optional[str] = None
        self._target_index: int = 0  # index into _TARGETS

        self._setup_ui()
        self._update_target_display()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle("Manual Card Input")
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setFixedSize(360, 270)
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
            QPushButton:pressed {{
                background-color: rgba(0, 150, 83, 220);
            }}
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # -- Title -----------------------------------------------------------
        title = QLabel("Manual Card Input")
        title.setStyleSheet(
            f"color: {_CLR_WHITE}; font-size: 13px; font-weight: 700; "
            "letter-spacing: 1px;"
        )
        root.addWidget(title)
        root.addWidget(_HRule(self))

        # -- Target mode row -------------------------------------------------
        target_row = QHBoxLayout()
        target_row.setSpacing(8)

        self._target_lbl = QLabel("Adding: Player Card")
        self._target_lbl.setStyleSheet(
            f"color: {_CLR_LIGHTGRAY}; font-size: 11px;"
        )

        self._toggle_btn = QPushButton("Change →")
        self._toggle_btn.setFixedSize(80, 26)
        self._toggle_btn.setStyleSheet(
            f"background-color: {_BG_BTN_RANK}; font-size: 11px;"
        )
        self._toggle_btn.clicked.connect(self._cycle_target)

        target_row.addWidget(self._target_lbl)
        target_row.addStretch()
        target_row.addWidget(self._toggle_btn)
        root.addLayout(target_row)

        # -- Rank buttons (13) -----------------------------------------------
        rank_lbl = QLabel("RANK")
        rank_lbl.setStyleSheet(
            f"color: {_CLR_MUTED}; font-size: 9px; letter-spacing: 1px; font-weight: 600;"
        )
        root.addWidget(rank_lbl)

        rank_grid = QGridLayout()
        rank_grid.setSpacing(4)
        self._rank_buttons: dict[str, QPushButton] = {}
        for i, rank in enumerate(_RANKS):
            btn = QPushButton(rank)
            btn.setFixedSize(22, 28)
            btn.setStyleSheet(f"background-color: {_BG_BTN_RANK};")
            btn.clicked.connect(lambda checked, r=rank: self._on_rank_clicked(r))
            self._rank_buttons[rank] = btn
            rank_grid.addWidget(btn, 0, i)
        root.addLayout(rank_grid)

        # -- Suit buttons (4) ------------------------------------------------
        suit_lbl = QLabel("SUIT")
        suit_lbl.setStyleSheet(
            f"color: {_CLR_MUTED}; font-size: 9px; letter-spacing: 1px; font-weight: 600;"
        )
        root.addWidget(suit_lbl)

        suit_row = QHBoxLayout()
        suit_row.setSpacing(6)
        self._suit_buttons: dict[str, QPushButton] = {}
        for symbol, code, color in _SUITS:
            btn = QPushButton(symbol)
            btn.setFixedSize(60, 30)
            btn.setStyleSheet(
                f"background-color: {_BG_BTN_SUIT}; color: {color}; font-size: 16px;"
            )
            btn.clicked.connect(lambda checked, c=code: self._on_suit_clicked(c))
            self._suit_buttons[code] = btn
            suit_row.addWidget(btn)
        root.addLayout(suit_row)

        root.addWidget(_HRule(self))

        # -- Status + action buttons -----------------------------------------
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)

        self._selection_lbl = QLabel("Select rank first")
        self._selection_lbl.setStyleSheet(f"color: {_CLR_MUTED}; font-size: 11px;")

        self._reset_btn = QPushButton("Reset Hand")
        self._reset_btn.setFixedSize(90, 28)
        self._reset_btn.setStyleSheet(
            f"background-color: rgba(180,20,30,180); font-size: 11px;"
        )
        self._reset_btn.clicked.connect(self._on_reset)

        self._done_btn = QPushButton("Done")
        self._done_btn.setFixedSize(60, 28)
        self._done_btn.setStyleSheet(
            f"background-color: rgba(0,120,60,180); font-size: 11px;"
        )
        self._done_btn.clicked.connect(self.accept)

        bottom_row.addWidget(self._selection_lbl)
        bottom_row.addStretch()
        bottom_row.addWidget(self._reset_btn)
        bottom_row.addWidget(self._done_btn)
        root.addLayout(bottom_row)

    # ------------------------------------------------------------------
    # Interaction handlers
    # ------------------------------------------------------------------

    def _on_rank_clicked(self, rank: str) -> None:
        """Highlight the selected rank button and store the selection."""
        # Clear previous highlight
        for r, btn in self._rank_buttons.items():
            btn.setStyleSheet(
                f"background-color: {_BG_BTN_RANK}; color: {_CLR_WHITE};"
            )
        # Apply highlight
        self._rank_buttons[rank].setStyleSheet(
            f"background-color: {_BG_BTN_ACTIVE}; color: {_CLR_WHITE};"
        )
        self._selected_rank = rank
        self._selection_lbl.setText(f"Selected: {rank} — pick suit")
        self._selection_lbl.setStyleSheet(f"color: {_CLR_YELLOW}; font-size: 11px;")

    def _on_suit_clicked(self, suit_code: str) -> None:
        """Complete card entry when suit is chosen (rank must already be selected)."""
        if self._selected_rank is None:
            self._selection_lbl.setText("⚠ Select rank first")
            self._selection_lbl.setStyleSheet(f"color: {_CLR_RED}; font-size: 11px;")
            return

        card_str = f"{self._selected_rank}{suit_code}"
        _, target = _TARGETS[self._target_index]

        logger.debug("Manual card added: %s → %s", card_str, target)
        self.card_added.emit(card_str, target)

        # Give brief confirmation then reset rank selection
        self._selection_lbl.setText(f"✓ Added {card_str}")
        self._selection_lbl.setStyleSheet(f"color: {_CLR_GREEN}; font-size: 11px;")
        self._clear_rank_selection()

    def _on_reset(self) -> None:
        """Emit a reset signal (card_added with special sentinel) and clear UI."""
        # Emit sentinel to signal hand reset upstream
        self.card_added.emit("__RESET__", "reset")
        self._clear_rank_selection()
        self._selection_lbl.setText("Hand reset")
        self._selection_lbl.setStyleSheet(f"color: {_CLR_MUTED}; font-size: 11px;")
        logger.debug("Manual input: hand reset requested.")

    def _cycle_target(self) -> None:
        """Cycle through input target modes."""
        self._target_index = (self._target_index + 1) % len(_TARGETS)
        self._update_target_display()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _update_target_display(self) -> None:
        label, _ = _TARGETS[self._target_index]
        self._target_lbl.setText(f"Adding: {label}")

        color_map = {"player": _CLR_GREEN, "dealer": _CLR_RED, "seen": _CLR_BLUE}
        _, target_code = _TARGETS[self._target_index]
        color = color_map.get(target_code, _CLR_WHITE)
        self._target_lbl.setStyleSheet(f"color: {color}; font-size: 11px;")

    def _clear_rank_selection(self) -> None:
        """Reset rank button highlights and internal selection."""
        self._selected_rank = None
        for btn in self._rank_buttons.values():
            btn.setStyleSheet(
                f"background-color: {_BG_BTN_RANK}; color: {_CLR_WHITE};"
            )
