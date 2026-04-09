"""
Calibration overlay for RiverRater.

Full-screen semi-transparent overlay that lets the user draw bounding boxes
around cards in a frozen screen capture, label them with rank + suit, and
commit the results as templates to the vision engine.

Signals:
    calibration_finished: Emitted when the user clicks Finish.
    calibration_cancelled: Emitted when the user cancels (Escape or Cancel).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from riverrater.vision.calibration import CalibrationSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design constants — matching the existing dark theme
# ---------------------------------------------------------------------------
_BG_DARK = "rgba(20, 20, 30, 210)"
_BG_BTN = "rgba(50, 50, 70, 240)"
_BG_BTN_ACTIVE = "rgba(0, 150, 83, 220)"
_CLR_GREEN = "#00C853"
_CLR_RED = "#FF1744"
_CLR_YELLOW = "#FFD600"
_CLR_WHITE = "#F0F0F8"
_CLR_MUTED = "#808090"
_CLR_LIGHTGRAY = "#C0C0D0"
_MONO_FONT = "Menlo, 'Courier New', Courier, monospace"

_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
_SUITS = [
    ("\u2665", "h", "#FF1744"),  # ♥
    ("\u2666", "d", "#FF6D00"),  # ♦
    ("\u2663", "c", "#C0C0D0"),  # ♣
    ("\u2660", "s", "#C0C0D0"),  # ♠
]


class _HRule(QFrame):
    """Thin horizontal divider line."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setFixedHeight(1)
        self.setStyleSheet("background-color: rgba(255,255,255,40); border: none;")


class CalibrationOverlay(QWidget):
    """Fullscreen semi-transparent overlay for interactive card template calibration.

    Signals:
        calibration_finished: Emitted when the user clicks Finish (templates committed).
        calibration_cancelled: Emitted when the user cancels (Escape or Cancel button).
    """

    calibration_finished = pyqtSignal()
    calibration_cancelled = pyqtSignal()

    def __init__(
        self,
        frame: np.ndarray,
        template_engine: object,
        profile_path: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._frame = frame
        self._template_engine = template_engine
        self._profile_path = profile_path

        # Convert BGR frame to QPixmap
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._frame_w = w
        self._frame_h = h

        # Drawing state
        self._drawing = False
        self._draw_start: Optional[QPoint] = None
        self._draw_end: Optional[QPoint] = None
        self._pending_bbox: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h) in frame coords

        # Selection state
        self._selected_rank: Optional[str] = None
        self._selected_suit: Optional[str] = None

        # Confirmed entries: list of (card_str, bbox_in_frame_coords)
        self._entries: list[tuple[str, tuple[int, int, int, int]]] = []

        # Frame display area (computed in _setup_ui / resizeEvent)
        self._frame_rect = QRect(0, 0, 0, 0)

        self._setup_ui()
        self._update_button_states()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        screen = QApplication.primaryScreen()
        if screen is not None:
            geom = screen.geometry()
            self.setGeometry(geom)
        else:
            self.setGeometry(0, 0, 1920, 1080)

        # Compute the frame display rect (left 75% of screen)
        panel_width = max(250, self.width() // 4)
        display_w = self.width() - panel_width
        display_h = self.height()

        # Scale frame to fit the display area while maintaining aspect ratio
        scale = min(display_w / self._frame_w, display_h / self._frame_h)
        scaled_w = int(self._frame_w * scale)
        scaled_h = int(self._frame_h * scale)
        offset_x = (display_w - scaled_w) // 2
        offset_y = (display_h - scaled_h) // 2
        self._frame_rect = QRect(offset_x, offset_y, scaled_w, scaled_h)

        # --- Control panel (right side) ---
        self._panel = QWidget(self)
        self._panel.setGeometry(display_w, 0, panel_width, self.height())
        self._panel.setStyleSheet(f"background-color: {_BG_DARK};")

        panel_layout = QVBoxLayout(self._panel)
        panel_layout.setContentsMargins(12, 16, 12, 12)
        panel_layout.setSpacing(8)

        # Title
        title = QLabel("CALIBRATION")
        title.setStyleSheet(
            f"color: {_CLR_WHITE}; font-size: 13px; font-weight: 700;"
            " letter-spacing: 2px; font-variant: small-caps;"
        )
        panel_layout.addWidget(title)
        panel_layout.addWidget(_HRule(self._panel))

        # Instruction
        self._instruction_label = QLabel("Draw a box around a card")
        self._instruction_label.setStyleSheet(f"color: {_CLR_LIGHTGRAY}; font-size: 11px;")
        self._instruction_label.setWordWrap(True)
        panel_layout.addWidget(self._instruction_label)

        # Rank label + buttons
        rank_lbl = QLabel("RANK")
        rank_lbl.setStyleSheet(
            f"color: {_CLR_MUTED}; font-size: 9px; letter-spacing: 1px; font-weight: 600;"
        )
        panel_layout.addWidget(rank_lbl)

        rank_row1 = QHBoxLayout()
        rank_row1.setSpacing(3)
        rank_row2 = QHBoxLayout()
        rank_row2.setSpacing(3)
        self._rank_buttons: dict[str, QPushButton] = {}
        for i, rank in enumerate(_RANKS):
            btn = QPushButton(rank)
            btn.setFixedSize(28, 28)
            btn.setStyleSheet(
                f"background-color: {_BG_BTN}; color: {_CLR_WHITE};"
                " border: 1px solid rgba(100,100,130,180); border-radius: 4px;"
                " font-size: 12px; font-weight: 600;"
            )
            btn.clicked.connect(lambda checked, r=rank: self._on_rank_clicked(r))
            self._rank_buttons[rank] = btn
            if i < 7:
                rank_row1.addWidget(btn)
            else:
                rank_row2.addWidget(btn)
        rank_row2.addStretch()
        panel_layout.addLayout(rank_row1)
        panel_layout.addLayout(rank_row2)

        # Suit label + buttons
        suit_lbl = QLabel("SUIT")
        suit_lbl.setStyleSheet(
            f"color: {_CLR_MUTED}; font-size: 9px; letter-spacing: 1px; font-weight: 600;"
        )
        panel_layout.addWidget(suit_lbl)

        suit_row = QHBoxLayout()
        suit_row.setSpacing(4)
        self._suit_buttons: dict[str, QPushButton] = {}
        for symbol, code, color in _SUITS:
            btn = QPushButton(symbol)
            btn.setFixedSize(44, 30)
            btn.setStyleSheet(
                f"background-color: {_BG_BTN}; color: {color};"
                " border: 1px solid rgba(100,100,130,180); border-radius: 4px;"
                " font-size: 16px;"
            )
            btn.clicked.connect(lambda checked, c=code: self._on_suit_clicked(c))
            self._suit_buttons[code] = btn
            suit_row.addWidget(btn)
        panel_layout.addLayout(suit_row)

        # ROI preview
        roi_lbl = QLabel("PREVIEW")
        roi_lbl.setStyleSheet(
            f"color: {_CLR_MUTED}; font-size: 9px; letter-spacing: 1px; font-weight: 600;"
        )
        panel_layout.addWidget(roi_lbl)

        self._roi_preview = QLabel()
        self._roi_preview.setFixedSize(120, 180)
        self._roi_preview.setStyleSheet(
            "background-color: rgba(40,40,58,255); border: 1px solid rgba(100,100,130,180);"
            " border-radius: 4px;"
        )
        self._roi_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._roi_preview.setScaledContents(True)
        panel_layout.addWidget(self._roi_preview)

        # Confirm button
        self._confirm_btn = QPushButton("Add Card")
        self._confirm_btn.setFixedHeight(32)
        self._confirm_btn.setStyleSheet(
            f"background-color: rgba(0,120,60,200); color: {_CLR_WHITE};"
            " border: none; border-radius: 6px; font-size: 13px; font-weight: 600;"
        )
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self._on_confirm)
        panel_layout.addWidget(self._confirm_btn)

        # Counter
        self._counter_label = QLabel("0 cards captured")
        self._counter_label.setStyleSheet(f"color: {_CLR_MUTED}; font-size: 11px;")
        panel_layout.addWidget(self._counter_label)

        panel_layout.addWidget(_HRule(self._panel))

        # Undo button
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setFixedHeight(28)
        self._undo_btn.setStyleSheet(
            f"background-color: {_BG_BTN}; color: {_CLR_WHITE};"
            " border: 1px solid rgba(100,100,130,180); border-radius: 6px;"
            " font-size: 11px;"
        )
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._on_undo)
        panel_layout.addWidget(self._undo_btn)

        # Finish button
        self._finish_btn = QPushButton("Finish")
        self._finish_btn.setFixedHeight(32)
        self._finish_btn.setStyleSheet(
            f"background-color: rgba(0,150,60,200); color: {_CLR_WHITE};"
            " border: none; border-radius: 6px; font-size: 13px; font-weight: 700;"
        )
        self._finish_btn.setEnabled(False)
        self._finish_btn.clicked.connect(self._on_finish)
        panel_layout.addWidget(self._finish_btn)

        # Cancel button
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(32)
        self._cancel_btn.setStyleSheet(
            f"background-color: rgba(200,20,30,200); color: {_CLR_WHITE};"
            " border: none; border-radius: 6px; font-size: 13px; font-weight: 700;"
        )
        self._cancel_btn.clicked.connect(self._on_cancel)
        panel_layout.addWidget(self._cancel_btn)

        panel_layout.addStretch()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Dark overlay background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 160))

        # Draw the frozen frame
        if not self._pixmap.isNull():
            painter.drawPixmap(self._frame_rect, self._pixmap)

        # Draw confirmed bboxes
        green_pen = QPen(QColor(_CLR_GREEN), 2)
        painter.setPen(green_pen)
        painter.setFont(painter.font())
        for card_str, bbox in self._entries:
            display_rect = self._frame_to_display_rect(bbox)
            painter.drawRect(display_rect)
            painter.drawText(display_rect.topLeft() + QPoint(2, -4), card_str)

        # Draw current rubber-band
        if self._drawing and self._draw_start is not None and self._draw_end is not None:
            pen = QPen(QColor(_CLR_GREEN), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            rect = QRect(self._draw_start, self._draw_end).normalized()
            painter.drawRect(rect)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse events (rubber-band drawing)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            if self._frame_rect.contains(pos):
                self._drawing = True
                self._draw_start = pos
                self._draw_end = pos
                self._pending_bbox = None
                self._update_button_states()
                self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drawing:
            self._draw_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            self._draw_end = event.pos()

            if self._draw_start is not None and self._draw_end is not None:
                rect = QRect(self._draw_start, self._draw_end).normalized()
                # Convert display coordinates to frame coordinates
                self._pending_bbox = self._display_to_frame_bbox(rect)
                self._update_roi_preview()

            self._update_button_states()
            self.update()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._on_cancel()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._confirm_btn.isEnabled():
                self._on_confirm()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Rank / Suit selection
    # ------------------------------------------------------------------

    def _on_rank_clicked(self, rank: str) -> None:
        for r, btn in self._rank_buttons.items():
            btn.setStyleSheet(
                f"background-color: {_BG_BTN}; color: {_CLR_WHITE};"
                " border: 1px solid rgba(100,100,130,180); border-radius: 4px;"
                " font-size: 12px; font-weight: 600;"
            )
        self._rank_buttons[rank].setStyleSheet(
            f"background-color: {_BG_BTN_ACTIVE}; color: {_CLR_WHITE};"
            " border: 1px solid rgba(100,100,130,180); border-radius: 4px;"
            " font-size: 12px; font-weight: 600;"
        )
        self._selected_rank = rank
        self._instruction_label.setText(f"Rank: {rank} — pick suit")
        self._update_button_states()

    def _on_suit_clicked(self, suit_code: str) -> None:
        for code, btn in self._suit_buttons.items():
            default_color = next(c for _, sc, c in _SUITS if sc == code)
            btn.setStyleSheet(
                f"background-color: {_BG_BTN}; color: {default_color};"
                " border: 1px solid rgba(100,100,130,180); border-radius: 4px;"
                " font-size: 16px;"
            )
        suit_color = next(c for _, sc, c in _SUITS if sc == suit_code)
        self._suit_buttons[suit_code].setStyleSheet(
            f"background-color: {_BG_BTN_ACTIVE}; color: {suit_color};"
            " border: 1px solid rgba(100,100,130,180); border-radius: 4px;"
            " font-size: 16px;"
        )
        self._selected_suit = suit_code
        if self._selected_rank:
            self._instruction_label.setText(
                f"Selected: {self._selected_rank}{suit_code} — draw box & confirm"
            )
        self._update_button_states()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_confirm(self) -> None:
        if self._pending_bbox is None or self._selected_rank is None or self._selected_suit is None:
            return
        card_str = f"{self._selected_rank}{self._selected_suit}"
        self._entries.append((card_str, self._pending_bbox))

        # Reset selection state
        self._pending_bbox = None
        self._draw_start = None
        self._draw_end = None
        self._selected_rank = None
        self._selected_suit = None

        # Reset button highlights
        for btn in self._rank_buttons.values():
            btn.setStyleSheet(
                f"background-color: {_BG_BTN}; color: {_CLR_WHITE};"
                " border: 1px solid rgba(100,100,130,180); border-radius: 4px;"
                " font-size: 12px; font-weight: 600;"
            )
        for code, btn in self._suit_buttons.items():
            default_color = next(c for _, sc, c in _SUITS if sc == code)
            btn.setStyleSheet(
                f"background-color: {_BG_BTN}; color: {default_color};"
                " border: 1px solid rgba(100,100,130,180); border-radius: 4px;"
                " font-size: 16px;"
            )

        self._roi_preview.clear()
        self._counter_label.setText(f"{len(self._entries)} cards captured")
        self._instruction_label.setText("Draw a box around the next card")
        self._update_button_states()
        self.update()

    def _on_undo(self) -> None:
        if self._entries:
            self._entries.pop()
            self._counter_label.setText(f"{len(self._entries)} cards captured")
            self._instruction_label.setText("Last entry removed")
            self._update_button_states()
            self.update()

    def _on_finish(self) -> None:
        if not self._entries:
            return
        session = CalibrationSession()
        for card_str, bbox in self._entries:
            session.add_calibration(card_str, bbox, self._frame)
        session.finish(self._template_engine)
        if hasattr(self._template_engine, "save_profile"):
            self._template_engine.save_profile(self._profile_path)
        self.calibration_finished.emit()
        self.close()

    def _on_cancel(self) -> None:
        self.calibration_cancelled.emit()
        self.close()

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _display_to_frame_bbox(self, display_rect: QRect) -> tuple[int, int, int, int]:
        """Convert a display-coordinate QRect to frame-coordinate (x, y, w, h)."""
        # Clamp to frame display area
        x1 = max(display_rect.x(), self._frame_rect.x())
        y1 = max(display_rect.y(), self._frame_rect.y())
        x2 = min(display_rect.x() + display_rect.width(), self._frame_rect.x() + self._frame_rect.width())
        y2 = min(display_rect.y() + display_rect.height(), self._frame_rect.y() + self._frame_rect.height())

        # Offset relative to frame display origin
        rel_x = x1 - self._frame_rect.x()
        rel_y = y1 - self._frame_rect.y()
        rel_w = x2 - x1
        rel_h = y2 - y1

        # Scale from display to frame coordinates
        if self._frame_rect.width() > 0 and self._frame_rect.height() > 0:
            scale_x = self._frame_w / self._frame_rect.width()
            scale_y = self._frame_h / self._frame_rect.height()
        else:
            scale_x = scale_y = 1.0

        fx = int(rel_x * scale_x)
        fy = int(rel_y * scale_y)
        fw = int(rel_w * scale_x)
        fh = int(rel_h * scale_y)
        return (fx, fy, fw, fh)

    def _frame_to_display_rect(self, bbox: tuple[int, int, int, int]) -> QRect:
        """Convert frame-coordinate (x, y, w, h) to a display QRect."""
        fx, fy, fw, fh = bbox
        if self._frame_w > 0 and self._frame_h > 0:
            scale_x = self._frame_rect.width() / self._frame_w
            scale_y = self._frame_rect.height() / self._frame_h
        else:
            scale_x = scale_y = 1.0

        dx = int(fx * scale_x) + self._frame_rect.left()
        dy = int(fy * scale_y) + self._frame_rect.top()
        dw = int(fw * scale_x)
        dh = int(fh * scale_y)
        return QRect(dx, dy, dw, dh)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_roi_preview(self) -> None:
        """Update the ROI preview label with a crop of the pending bbox."""
        if self._pending_bbox is None:
            self._roi_preview.clear()
            return
        x, y, w, h = self._pending_bbox
        if w <= 0 or h <= 0:
            return
        # Clamp to frame bounds
        fh, fw = self._frame.shape[:2]
        x2 = min(x + w, fw)
        y2 = min(y + h, fh)
        x = max(0, x)
        y = max(0, y)
        roi = self._frame[y:y2, x:x2]
        if roi.size == 0:
            return
        rh, rw, rch = roi.shape
        bpl = rch * rw
        qimg = QImage(roi.data, rw, rh, bpl, QImage.Format.Format_BGR888)
        pm = QPixmap.fromImage(qimg)
        self._roi_preview.setPixmap(pm)

    def _update_button_states(self) -> None:
        """Enable/disable buttons based on current state."""
        can_confirm = (
            self._pending_bbox is not None
            and self._selected_rank is not None
            and self._selected_suit is not None
        )
        self._confirm_btn.setEnabled(can_confirm)
        self._finish_btn.setEnabled(len(self._entries) >= 1)
        self._undo_btn.setEnabled(len(self._entries) >= 1)
