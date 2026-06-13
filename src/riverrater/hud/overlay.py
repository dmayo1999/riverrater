"""
Main HUD overlay window for RiverRater.

A transparent, frameless, always-on-top window that hosts both the poker and
blackjack analysis views and can be dragged around the screen.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from riverrater.game.state import GameMode, PokerResult, BlackjackResult
from riverrater.hud.poker_view import PokerView
from riverrater.hud.blackjack_view import BlackjackView

logger = logging.getLogger(__name__)

# Stacked widget page indices
_PAGE_POKER = 0
_PAGE_BLACKJACK = 1


class HUDOverlay(QMainWindow):
    """Transparent always-on-top overlay window.

    The overlay hosts a :class:`QStackedWidget` containing one page for each
    game mode.  It can be repositioned by clicking and dragging anywhere on
    the window.

    Usage::

        overlay = HUDOverlay()
        overlay.set_mode(GameMode.POKER)
        overlay.show()
    """

    def __init__(self, *, num_opponents: int = 1) -> None:
        super().__init__()
        self._drag_position: Optional[QPoint] = None
        self._initial_num_opponents = num_opponents
        self._setup_window()
        self._setup_ui()

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """Configure window flags, transparency and initial geometry."""
        flags = (
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool  # Keeps the window off the taskbar
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowOpacity(0.85)
        self.setMinimumSize(300, 200)

    def _setup_ui(self) -> None:
        """Build the stacked widget and child views."""
        self._stack = QStackedWidget()
        self._stack.setObjectName("HUDStack")

        self._poker_view = PokerView(num_opponents=self._initial_num_opponents)
        self._blackjack_view = BlackjackView()

        self._stack.insertWidget(_PAGE_POKER, self._poker_view)
        self._stack.insertWidget(_PAGE_BLACKJACK, self._blackjack_view)
        self._stack.setCurrentIndex(_PAGE_POKER)

        self.setCentralWidget(self._stack)
        self.adjustSize()

    # ------------------------------------------------------------------
    # Public interface (matches INTERFACES.md)
    # ------------------------------------------------------------------

    def set_mode(self, mode: GameMode) -> None:
        """Switch between poker and blackjack view.

        Args:
            mode: Target :class:`GameMode`.
        """
        if mode == GameMode.POKER:
            self._stack.setCurrentIndex(_PAGE_POKER)
            logger.debug("HUD switched to poker mode.")
        elif mode == GameMode.BLACKJACK:
            self._stack.setCurrentIndex(_PAGE_BLACKJACK)
            logger.debug("HUD switched to blackjack mode.")
        else:
            logger.warning("Unknown GameMode: %s", mode)

    @property
    def poker_view(self) -> PokerView:
        """Direct access to the embedded poker HUD panel."""
        return self._poker_view

    def update_poker(self, result: PokerResult) -> None:
        """Push a :class:`PokerResult` to the poker view.

        Args:
            result: Latest analysis result from the poker math engine.
        """
        self._poker_view.update(result)

    def update_blackjack(self, result: BlackjackResult) -> None:
        """Push a :class:`BlackjackResult` to the blackjack view.

        Args:
            result: Latest analysis result from the blackjack math engine.
        """
        self._blackjack_view.update(result)

    def set_visible(self, visible: bool) -> None:
        """Show or hide the overlay.

        Args:
            visible: ``True`` to show, ``False`` to hide.
        """
        if visible:
            self.show()
        else:
            self.hide()

    def toggle_visibility(self) -> None:
        """Flip the overlay between visible and hidden."""
        if self.isVisible():
            self.hide()
            logger.debug("HUD hidden.")
        else:
            self.show()
            logger.debug("HUD shown.")

    def set_position(self, x: int, y: int) -> None:
        """Move the overlay to screen co-ordinates *(x, y)*.

        Args:
            x: Horizontal screen position (pixels from left).
            y: Vertical screen position (pixels from top).
        """
        self.move(x, y)

    def set_opacity(self, opacity: float) -> None:
        """Set window-level opacity.

        Args:
            opacity: A float in ``[0.0, 1.0]``.  Values outside this range
                are clamped automatically by Qt.
        """
        self.setWindowOpacity(max(0.0, min(1.0, opacity)))

    # ------------------------------------------------------------------
    # Mouse drag support
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Record the drag start position on left-button press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Drag the overlay to follow the mouse while the left button is held."""
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_position is not None:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Clear drag position on mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)
