"""
Application configuration for RiverRater.

Provides the AppConfig dataclass with JSON serialization support.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# GameMode is defined in game.state; import it if available,
# otherwise define a local fallback so config.py can be imported independently.
try:
    from riverrater.game.state import GameMode
except ImportError:
    # Fallback definition — should be replaced by the actual import
    # once riverrater.game.state is built by the game-state agent.
    from enum import Enum

    class GameMode(Enum):  # type: ignore[no-redef]
        POKER = "poker"
        BLACKJACK = "blackjack"


def _get_default_config_path() -> str:
    """Return the platform-appropriate default config path."""
    return str(Path.home() / ".riverrater" / "config.json")


@dataclass
class AppConfig:
    """
    Central application configuration.

    Supports JSON round-trip via :meth:`load` / :meth:`save`.
    All fields have sensible defaults so a brand-new installation
    works out of the box.
    """

    # ------------------------------------------------------------------ #
    # Screen capture
    # ------------------------------------------------------------------ #
    capture_region: Optional[tuple[int, int, int, int]] = None
    """(left, top, width, height) of the capture region, or None for the
    full primary monitor."""
    capture_fps_target: int = 30
    """Target capture frames-per-second for the background capture loop."""

    # ------------------------------------------------------------------ #
    # Game mode
    # ------------------------------------------------------------------ #
    game_mode: GameMode = field(default_factory=lambda: GameMode.POKER)
    """Active game mode: POKER or BLACKJACK."""

    # ------------------------------------------------------------------ #
    # Vision
    # ------------------------------------------------------------------ #
    vision_profile: str = "default"
    """Name of the active vision profile (directory under ~/.riverrater/profiles/)."""
    detection_confidence: float = 0.8
    """Minimum confidence threshold (0–1) for card detections."""

    # ------------------------------------------------------------------ #
    # Blackjack-specific
    # ------------------------------------------------------------------ #
    num_decks: int = 6
    """Number of decks in the shoe."""
    min_bet: float = 10.0
    """Table minimum bet in currency units."""
    max_bet: float = 500.0
    """Table maximum bet in currency units."""
    bankroll: float = 5000.0
    """Player's current bankroll for Kelly-criterion bet sizing."""

    # ------------------------------------------------------------------ #
    # HUD
    # ------------------------------------------------------------------ #
    hud_position: tuple[int, int] = field(default_factory=lambda: (100, 100))
    """(x, y) screen position of the HUD overlay window."""
    hud_opacity: float = 0.85
    """HUD window opacity, 0.0 (fully transparent) to 1.0 (fully opaque)."""

    # ------------------------------------------------------------------ #
    # Hotkeys
    # ------------------------------------------------------------------ #
    hotkey_toggle_hud: str = "<ctrl>+<shift>+h"
    """Hotkey to show / hide the HUD overlay."""
    hotkey_calibrate: str = "<ctrl>+<shift>+c"
    """Hotkey to open the calibration wizard."""
    hotkey_manual_card: str = "<ctrl>+<shift>+m"
    """Hotkey to manually enter a card."""
    hotkey_reset_hand: str = "<ctrl>+<shift>+r"
    """Hotkey to reset / clear the current hand."""
    hotkey_switch_mode: str = "<ctrl>+<shift>+s"
    """Hotkey to switch between game modes."""

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        """
        Load an :class:`AppConfig` from a JSON file.

        Missing fields fall back to their default values, so an older
        config file is always forward-compatible.

        Parameters
        ----------
        path:
            Absolute or relative path to the JSON config file.

        Returns
        -------
        AppConfig
            Populated configuration instance.
        """
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with config_path.open("r", encoding="utf-8") as fh:
            raw: dict = json.load(fh)

        # Start with defaults and overlay whatever the file provides.
        instance = cls()

        # capture_region — stored as list[int] in JSON
        if "capture_region" in raw:
            cr = raw["capture_region"]
            instance.capture_region = tuple(cr) if cr is not None else None  # type: ignore[assignment]

        if "capture_fps_target" in raw:
            instance.capture_fps_target = int(raw["capture_fps_target"])

        if "game_mode" in raw:
            try:
                instance.game_mode = GameMode(raw["game_mode"])
            except ValueError:
                pass  # keep default if value is unrecognised

        if "vision_profile" in raw:
            instance.vision_profile = str(raw["vision_profile"])

        if "detection_confidence" in raw:
            instance.detection_confidence = float(raw["detection_confidence"])

        if "num_decks" in raw:
            instance.num_decks = int(raw["num_decks"])

        if "min_bet" in raw:
            instance.min_bet = float(raw["min_bet"])

        if "max_bet" in raw:
            instance.max_bet = float(raw["max_bet"])

        if "bankroll" in raw:
            instance.bankroll = float(raw["bankroll"])

        if "hud_position" in raw:
            hp = raw["hud_position"]
            instance.hud_position = (int(hp[0]), int(hp[1]))

        if "hud_opacity" in raw:
            instance.hud_opacity = float(raw["hud_opacity"])

        if "hotkey_toggle_hud" in raw:
            instance.hotkey_toggle_hud = str(raw["hotkey_toggle_hud"])

        if "hotkey_calibrate" in raw:
            instance.hotkey_calibrate = str(raw["hotkey_calibrate"])

        if "hotkey_manual_card" in raw:
            instance.hotkey_manual_card = str(raw["hotkey_manual_card"])

        if "hotkey_reset_hand" in raw:
            instance.hotkey_reset_hand = str(raw["hotkey_reset_hand"])

        if "hotkey_switch_mode" in raw:
            instance.hotkey_switch_mode = str(raw["hotkey_switch_mode"])

        return instance

    def save(self, path: str) -> None:
        """
        Persist the configuration to a JSON file.

        Intermediate directories are created automatically.

        Parameters
        ----------
        path:
            Destination file path.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload: dict = {
            "capture_region": list(self.capture_region) if self.capture_region is not None else None,
            "capture_fps_target": self.capture_fps_target,
            "game_mode": self.game_mode.value,
            "vision_profile": self.vision_profile,
            "detection_confidence": self.detection_confidence,
            "num_decks": self.num_decks,
            "min_bet": self.min_bet,
            "max_bet": self.max_bet,
            "bankroll": self.bankroll,
            "hud_position": list(self.hud_position),
            "hud_opacity": self.hud_opacity,
            "hotkey_toggle_hud": self.hotkey_toggle_hud,
            "hotkey_calibrate": self.hotkey_calibrate,
            "hotkey_manual_card": self.hotkey_manual_card,
            "hotkey_reset_hand": self.hotkey_reset_hand,
            "hotkey_switch_mode": self.hotkey_switch_mode,
        }

        with dest.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @staticmethod
    def get_default_config_path() -> str:
        """
        Return the default config file location.

        Returns
        -------
        str
            ``~/.riverrater/config.json`` (expanded to the user's home directory).
        """
        return _get_default_config_path()
