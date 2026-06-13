"""
Application settings dialog for RiverRater.

Provides a PyQt6 dialog for editing :class:`~riverrater.main.AppConfig`
fields.  Changes are persisted via :meth:`AppConfig.save` when the user
clicks Save.

Signals:
    settings_saved(AppConfig) — emitted with the updated config after a
        successful save.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from riverrater.main import AppConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design constants — match PokerInputDialog / ManualCardInput style
# ---------------------------------------------------------------------------
_BG_DIALOG = "rgb(28, 28, 40)"
_CLR_WHITE = "#F0F0F8"
_CLR_LIGHTGRAY = "#C0C0D0"


class _HRule(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setFixedHeight(1)
        self.setStyleSheet("background-color: rgba(255,255,255,40); border: none;")


def _region_spinbox() -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(0, 10_000)
    spin.setFixedWidth(80)
    return spin


class SettingsDialog(QDialog):
    """Dialog for editing persisted application settings.

    Signals:
        settings_saved: Emitted with the updated :class:`AppConfig` after
            the user clicks Save and values are written to disk.
    """

    settings_saved = pyqtSignal(object)

    def __init__(
        self,
        config: "AppConfig",
        config_path: str | Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._base_config = config
        self._setup_ui()
        self.load_from_config(config)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle("Settings")
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.resize(420, 560)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {_BG_DIALOG};
            }}
            QLabel {{
                color: {_CLR_WHITE};
            }}
            QGroupBox {{
                color: {_CLR_LIGHTGRAY};
                border: 1px solid rgba(100,100,130,120);
                border-radius: 6px;
                margin-top: 10px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
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
            QDoubleSpinBox, QSpinBox, QLineEdit {{
                color: {_CLR_WHITE};
                background-color: rgba(40, 40, 58, 255);
                border: 1px solid rgba(100,100,130,180);
                border-radius: 4px;
                padding: 4px;
                font-size: 13px;
            }}
            QCheckBox {{
                color: {_CLR_LIGHTGRAY};
                font-size: 12px;
            }}
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        title = QLabel("Settings")
        title.setStyleSheet(
            f"color: {_CLR_WHITE}; font-size: 13px; font-weight: 700; "
            "letter-spacing: 1px;"
        )
        outer.addWidget(title)
        outer.addWidget(_HRule(self))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 4, 0)
        content_layout.setSpacing(10)

        content_layout.addWidget(self._build_capture_group())
        content_layout.addWidget(self._build_hud_group())
        content_layout.addWidget(self._build_blackjack_group())
        content_layout.addWidget(self._build_vision_group())
        content_layout.addWidget(self._build_pot_ocr_group())
        content_layout.addWidget(self._build_yolo_group())
        content_layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

        outer.addWidget(_HRule(self))

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._save_btn = QPushButton("Save")
        self._save_btn.setFixedSize(80, 28)
        self._save_btn.setStyleSheet(
            "background-color: rgba(0,120,60,180); font-size: 12px;"
        )
        self._save_btn.clicked.connect(self._on_save)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedSize(80, 28)
        self._cancel_btn.setStyleSheet(
            "background-color: rgba(180,20,30,180); font-size: 12px;"
        )
        self._cancel_btn.clicked.connect(self.reject)

        btn_row.addStretch()
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._save_btn)
        outer.addLayout(btn_row)

    def _build_capture_group(self) -> QGroupBox:
        group = QGroupBox("Capture")
        layout = QVBoxLayout(group)

        self._capture_enabled = QCheckBox("Use custom capture region")
        layout.addWidget(self._capture_enabled)

        region_row = QHBoxLayout()
        self._capture_left = _region_spinbox()
        self._capture_top = _region_spinbox()
        self._capture_width = _region_spinbox()
        self._capture_height = _region_spinbox()
        for label, spin in (
            ("Left", self._capture_left),
            ("Top", self._capture_top),
            ("Width", self._capture_width),
            ("Height", self._capture_height),
        ):
            region_row.addWidget(QLabel(label))
            region_row.addWidget(spin)
        layout.addLayout(region_row)

        self._capture_enabled.toggled.connect(self._set_capture_fields_enabled)
        return group

    def _build_hud_group(self) -> QGroupBox:
        group = QGroupBox("HUD")
        form = QFormLayout(group)

        self._hud_x = _region_spinbox()
        self._hud_y = _region_spinbox()
        pos_row = QHBoxLayout()
        pos_row.addWidget(self._hud_x)
        pos_row.addWidget(self._hud_y)
        form.addRow("Position (x, y)", pos_row)

        self._hud_opacity = QDoubleSpinBox()
        self._hud_opacity.setRange(0.1, 1.0)
        self._hud_opacity.setSingleStep(0.05)
        self._hud_opacity.setDecimals(2)
        form.addRow("Opacity", self._hud_opacity)

        return group

    def _build_blackjack_group(self) -> QGroupBox:
        group = QGroupBox("Blackjack")
        form = QFormLayout(group)

        self._num_decks = QSpinBox()
        self._num_decks.setRange(1, 8)
        form.addRow("Decks in shoe", self._num_decks)

        self._min_bet = QDoubleSpinBox()
        self._min_bet.setRange(1.0, 1_000_000.0)
        self._min_bet.setDecimals(2)
        self._min_bet.setPrefix("$")
        form.addRow("Min bet", self._min_bet)

        self._max_bet = QDoubleSpinBox()
        self._max_bet.setRange(1.0, 1_000_000.0)
        self._max_bet.setDecimals(2)
        self._max_bet.setPrefix("$")
        form.addRow("Max bet", self._max_bet)

        self._bankroll = QDoubleSpinBox()
        self._bankroll.setRange(1.0, 100_000_000.0)
        self._bankroll.setDecimals(2)
        self._bankroll.setPrefix("$")
        form.addRow("Bankroll", self._bankroll)

        return group

    def _build_vision_group(self) -> QGroupBox:
        group = QGroupBox("Vision")
        form = QFormLayout(group)

        self._vision_profile = QLineEdit()
        form.addRow("Profile name", self._vision_profile)

        self._detection_confidence = QDoubleSpinBox()
        self._detection_confidence.setRange(0.1, 1.0)
        self._detection_confidence.setSingleStep(0.05)
        self._detection_confidence.setDecimals(2)
        form.addRow("Detection confidence", self._detection_confidence)

        return group

    def _build_pot_ocr_group(self) -> QGroupBox:
        group = QGroupBox("Pot OCR")
        layout = QVBoxLayout(group)

        self._pot_ocr_enabled = QCheckBox("Enable pot OCR")
        layout.addWidget(self._pot_ocr_enabled)

        form = QFormLayout()
        self._pot_ocr_confidence = QDoubleSpinBox()
        self._pot_ocr_confidence.setRange(0.1, 1.0)
        self._pot_ocr_confidence.setSingleStep(0.05)
        self._pot_ocr_confidence.setDecimals(2)
        form.addRow("OCR confidence", self._pot_ocr_confidence)
        layout.addLayout(form)

        self._pot_roi_enabled = QCheckBox("Custom pot ROI")
        layout.addWidget(self._pot_roi_enabled)
        pot_row = QHBoxLayout()
        self._pot_left = _region_spinbox()
        self._pot_top = _region_spinbox()
        self._pot_width = _region_spinbox()
        self._pot_height = _region_spinbox()
        for label, spin in (
            ("L", self._pot_left),
            ("T", self._pot_top),
            ("W", self._pot_width),
            ("H", self._pot_height),
        ):
            pot_row.addWidget(QLabel(label))
            pot_row.addWidget(spin)
        layout.addLayout(pot_row)

        self._bet_roi_enabled = QCheckBox("Custom bet ROI")
        layout.addWidget(self._bet_roi_enabled)
        bet_row = QHBoxLayout()
        self._bet_left = _region_spinbox()
        self._bet_top = _region_spinbox()
        self._bet_width = _region_spinbox()
        self._bet_height = _region_spinbox()
        for label, spin in (
            ("L", self._bet_left),
            ("T", self._bet_top),
            ("W", self._bet_width),
            ("H", self._bet_height),
        ):
            bet_row.addWidget(QLabel(label))
            bet_row.addWidget(spin)
        layout.addLayout(bet_row)

        self._pot_roi_enabled.toggled.connect(self._set_pot_roi_fields_enabled)
        self._bet_roi_enabled.toggled.connect(self._set_bet_roi_fields_enabled)
        return group

    def _build_yolo_group(self) -> QGroupBox:
        group = QGroupBox("YOLO")
        form = QFormLayout(group)

        self._yolo_model_path = QLineEdit()
        self._yolo_model_path.setPlaceholderText("Leave empty for default weights")
        form.addRow("Model path", self._yolo_model_path)

        self._yolo_confidence = QDoubleSpinBox()
        self._yolo_confidence.setRange(0.1, 1.0)
        self._yolo_confidence.setSingleStep(0.05)
        self._yolo_confidence.setDecimals(2)
        form.addRow("Detection confidence", self._yolo_confidence)

        return group

    # ------------------------------------------------------------------
    # Config load / collect
    # ------------------------------------------------------------------

    def load_from_config(self, config: "AppConfig") -> None:
        """Populate widgets from *config*."""
        self._base_config = config
        if config.capture_region is not None:
            left, top, width, height = config.capture_region
            self._capture_enabled.setChecked(True)
            self._capture_left.setValue(left)
            self._capture_top.setValue(top)
            self._capture_width.setValue(width)
            self._capture_height.setValue(height)
        else:
            self._capture_enabled.setChecked(False)
            self._capture_left.setValue(0)
            self._capture_top.setValue(0)
            self._capture_width.setValue(800)
            self._capture_height.setValue(600)

        self._hud_x.setValue(config.hud_position[0])
        self._hud_y.setValue(config.hud_position[1])
        self._hud_opacity.setValue(config.hud_opacity)

        self._num_decks.setValue(config.num_decks)
        self._min_bet.setValue(config.min_bet)
        self._max_bet.setValue(config.max_bet)
        self._bankroll.setValue(config.bankroll)

        self._vision_profile.setText(config.vision_profile)
        self._detection_confidence.setValue(config.detection_confidence)

        self._pot_ocr_enabled.setChecked(config.pot_ocr_enabled)
        self._pot_ocr_confidence.setValue(config.pot_ocr_confidence)

        if config.pot_roi is not None:
            left, top, width, height = config.pot_roi
            self._pot_roi_enabled.setChecked(True)
            self._pot_left.setValue(left)
            self._pot_top.setValue(top)
            self._pot_width.setValue(width)
            self._pot_height.setValue(height)
        else:
            self._pot_roi_enabled.setChecked(False)

        if config.bet_roi is not None:
            left, top, width, height = config.bet_roi
            self._bet_roi_enabled.setChecked(True)
            self._bet_left.setValue(left)
            self._bet_top.setValue(top)
            self._bet_width.setValue(width)
            self._bet_height.setValue(height)
        else:
            self._bet_roi_enabled.setChecked(False)

        self._yolo_model_path.setText(config.yolo_model_path or "")
        self._yolo_confidence.setValue(config.yolo_confidence)

        self._set_capture_fields_enabled(self._capture_enabled.isChecked())
        self._set_pot_roi_fields_enabled(self._pot_roi_enabled.isChecked())
        self._set_bet_roi_fields_enabled(self._bet_roi_enabled.isChecked())

    def collect_config(self, base: "AppConfig") -> "AppConfig":
        """Return a copy of *base* with widget values applied."""
        from dataclasses import replace

        capture_region: Optional[tuple[int, int, int, int]]
        if self._capture_enabled.isChecked():
            capture_region = (
                self._capture_left.value(),
                self._capture_top.value(),
                self._capture_width.value(),
                self._capture_height.value(),
            )
        else:
            capture_region = None

        pot_roi: Optional[tuple[int, int, int, int]]
        if self._pot_roi_enabled.isChecked():
            pot_roi = (
                self._pot_left.value(),
                self._pot_top.value(),
                self._pot_width.value(),
                self._pot_height.value(),
            )
        else:
            pot_roi = None

        bet_roi: Optional[tuple[int, int, int, int]]
        if self._bet_roi_enabled.isChecked():
            bet_roi = (
                self._bet_left.value(),
                self._bet_top.value(),
                self._bet_width.value(),
                self._bet_height.value(),
            )
        else:
            bet_roi = None

        yolo_path = self._yolo_model_path.text().strip() or None

        return replace(
            base,
            capture_region=capture_region,
            hud_position=(self._hud_x.value(), self._hud_y.value()),
            hud_opacity=self._hud_opacity.value(),
            num_decks=self._num_decks.value(),
            min_bet=self._min_bet.value(),
            max_bet=self._max_bet.value(),
            bankroll=self._bankroll.value(),
            vision_profile=self._vision_profile.text().strip() or "default",
            detection_confidence=self._detection_confidence.value(),
            pot_ocr_enabled=self._pot_ocr_enabled.isChecked(),
            pot_ocr_confidence=self._pot_ocr_confidence.value(),
            pot_roi=pot_roi,
            bet_roi=bet_roi,
            yolo_model_path=yolo_path,
            yolo_confidence=self._yolo_confidence.value(),
        )

    # ------------------------------------------------------------------
    # Interaction handlers
    # ------------------------------------------------------------------

    def _set_capture_fields_enabled(self, enabled: bool) -> None:
        for spin in (
            self._capture_left,
            self._capture_top,
            self._capture_width,
            self._capture_height,
        ):
            spin.setEnabled(enabled)

    def _set_pot_roi_fields_enabled(self, enabled: bool) -> None:
        for spin in (
            self._pot_left,
            self._pot_top,
            self._pot_width,
            self._pot_height,
        ):
            spin.setEnabled(enabled)

    def _set_bet_roi_fields_enabled(self, enabled: bool) -> None:
        for spin in (
            self._bet_left,
            self._bet_top,
            self._bet_width,
            self._bet_height,
        ):
            spin.setEnabled(enabled)

    def _on_save(self) -> None:
        updated = self.collect_config(self._base_config)
        self._base_config = updated
        updated.save(self._config_path)
        logger.debug("Settings saved to %s", self._config_path)
        self.settings_saved.emit(updated)
        self.accept()

    # ------------------------------------------------------------------
    # Public accessors (for testing)
    # ------------------------------------------------------------------

    @property
    def hud_opacity(self) -> float:
        return self._hud_opacity.value()

    @hud_opacity.setter
    def hud_opacity(self, value: float) -> None:
        self._hud_opacity.setValue(value)

    @property
    def num_decks(self) -> int:
        return self._num_decks.value()

    @num_decks.setter
    def num_decks(self, value: int) -> None:
        self._num_decks.setValue(value)

    @property
    def vision_profile(self) -> str:
        return self._vision_profile.text()

    @vision_profile.setter
    def vision_profile(self, value: str) -> None:
        self._vision_profile.setText(value)

    @property
    def detection_confidence(self) -> float:
        return self._detection_confidence.value()

    @detection_confidence.setter
    def detection_confidence(self, value: float) -> None:
        self._detection_confidence.setValue(value)

    @property
    def pot_ocr_enabled(self) -> bool:
        return self._pot_ocr_enabled.isChecked()

    @pot_ocr_enabled.setter
    def pot_ocr_enabled(self, value: bool) -> None:
        self._pot_ocr_enabled.setChecked(value)

    @property
    def yolo_model_path(self) -> str:
        return self._yolo_model_path.text()

    @yolo_model_path.setter
    def yolo_model_path(self, value: str) -> None:
        self._yolo_model_path.setText(value)