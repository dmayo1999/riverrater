"""
RiverRater — main entry point.

Wires together screen capture, vision engines, game state, math engines,
the HUD overlay, hotkey system, and the 30fps Qt timer that drives the
processing loop.

Usage::

    riverrater --mode poker
    riverrater --mode blackjack --debug
    riverrater --mode poker --config /path/to/config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Early PyQt6 import — must happen before any other riverrater module that
# might trigger a Qt import so that QApplication exists before any widget.
# ---------------------------------------------------------------------------
from PyQt6.QtCore import QMetaObject, QTimer, Qt, Q_ARG
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

# ---------------------------------------------------------------------------
# Internal modules — all wrapped in try/except to allow partial imports
# during development.
# ---------------------------------------------------------------------------
from riverrater.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# -- Game state --------------------------------------------------------------
try:
    from riverrater.game.state import (
        BlackjackAction,
        BlackjackResult,
        BlackjackState,
        Card,
        GameMode,
        PokerResult,
        PokerState,
        Rank,
        Suit,
    )
    _HAS_STATE = True
except ImportError as _e:
    logger.warning("game.state not available: %s", _e)
    _HAS_STATE = False
    # Minimal stubs so the rest of this file can be imported.
    from enum import Enum
    from dataclasses import dataclass as _dc, field as _f
    from typing import Optional as _Opt

    class GameMode(Enum):  # type: ignore[no-redef]
        POKER = "poker"
        BLACKJACK = "blackjack"

    class Suit(Enum):  # type: ignore[no-redef]
        HEARTS = "h"; DIAMONDS = "d"; CLUBS = "c"; SPADES = "s"

    class Rank(Enum):  # type: ignore[no-redef]
        TWO = "2"; THREE = "3"; FOUR = "4"; FIVE = "5"; SIX = "6"
        SEVEN = "7"; EIGHT = "8"; NINE = "9"; TEN = "T"
        JACK = "J"; QUEEN = "Q"; KING = "K"; ACE = "A"

    @_dc
    class Card:  # type: ignore[no-redef]
        rank: Rank; suit: Suit
        def __str__(self): return f"{self.rank.value}{self.suit.value}"
        def __hash__(self): return hash((self.rank, self.suit))
        def __eq__(self, other): return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    @_dc
    class PokerState:  # type: ignore[no-redef]
        hole_cards: list = _f(default_factory=list)
        community_cards: list = _f(default_factory=list)
        pot_size: float = 0.0; bet_to_call: float = 0.0; num_opponents: int = 1

    @_dc
    class BlackjackState:  # type: ignore[no-redef]
        player_hand: list = _f(default_factory=list)
        dealer_upcard: _Opt[Card] = None
        cards_seen: list = _f(default_factory=list)
        num_decks: int = 6

    @_dc
    class PokerResult:  # type: ignore[no-redef]
        win_pct: float = 0.0; tie_pct: float = 0.0
        required_equity: float = 0.0; actual_equity: float = 0.0
        ev_call: float = 0.0; ev_fold: float = 0.0; ev_raise: float = 0.0
        recommended_action: _Opt[object] = None

    @_dc
    class BlackjackResult:  # type: ignore[no-redef]
        running_count: int = 0; true_count: float = 0.0
        recommended_action: _Opt[object] = None
        recommended_bet: float = 0.0; shoe_favorability: float = 0.0
        hand_total: int = 0; is_soft: bool = False

# -- Math engines ------------------------------------------------------------
try:
    from riverrater.game.poker_math import analyze_poker
    _HAS_POKER_MATH = True
except ImportError as _e:
    logger.warning("poker_math not available: %s — using stub.", _e)
    _HAS_POKER_MATH = False
    def analyze_poker(state) -> PokerResult:  # type: ignore[misc]
        return PokerResult()

try:
    from riverrater.game.blackjack_math import analyze_blackjack
    _HAS_BJ_MATH = True
except ImportError as _e:
    logger.warning("blackjack_math not available: %s — using stub.", _e)
    _HAS_BJ_MATH = False
    def analyze_blackjack(state) -> BlackjackResult:  # type: ignore[misc]
        return BlackjackResult()

# -- Screen capture ----------------------------------------------------------
try:
    from riverrater.capture.screen import ScreenCapture
    _HAS_CAPTURE = True
except ImportError as _e:
    logger.warning("capture.screen not available: %s — using stub.", _e)
    _HAS_CAPTURE = False

    class ScreenCapture:  # type: ignore[no-redef]
        def __init__(self, region=None): self._region = region
        def set_region(self, region): self._region = region
        def grab_frame(self): return None
        def get_fps(self): return 0.0
        def start(self): pass
        def stop(self): pass
        def get_latest_frame(self): return None

# -- Vision engines ----------------------------------------------------------
try:
    from riverrater.vision.template_engine import TemplateEngine
    _HAS_TEMPLATE = True
except ImportError as _e:
    logger.warning("vision.template_engine not available: %s — using stub.", _e)
    _HAS_TEMPLATE = False

    class TemplateEngine:  # type: ignore[no-redef]
        def __init__(self, profile_path=None): pass
        def detect_cards(self, frame, confidence=0.8): return []
        def add_template(self, card, template_image): pass
        def save_profile(self, path): pass
        def load_profile(self, path): pass

try:
    from riverrater.vision.yolo_engine import YOLOEngine
    _HAS_YOLO = True
except ImportError as _e:
    logger.warning("vision.yolo_engine not available: %s — using stub.", _e)
    _HAS_YOLO = False

    class YOLOEngine:  # type: ignore[no-redef]
        def __init__(self, model_path=None): self._model = None
        def detect_cards(self, frame, confidence=0.5): return []
        @property
        def is_available(self): return False

# -- HUD overlay -------------------------------------------------------------
try:
    from riverrater.hud.overlay import HUDOverlay
    from riverrater.hud.manual_input import ManualCardInput
    _HAS_HUD = True
except ImportError as _e:
    logger.warning("hud not available: %s", _e)
    _HAS_HUD = False
    HUDOverlay = None  # type: ignore[assignment,misc]
    ManualCardInput = None  # type: ignore[assignment,misc]

# -- Hotkeys -----------------------------------------------------------------
from riverrater.utils.hotkeys import HotkeyManager

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path.home() / ".riverrater"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.json"


@dataclass
class AppConfig:
    """Application configuration, persisted as JSON.

    All fields have sensible defaults so a fresh config file is valid on first
    run.
    """

    # Screen capture
    capture_region: Optional[tuple[int, int, int, int]] = None
    capture_fps_target: int = 30

    # Game mode
    game_mode: str = GameMode.POKER.value

    # Vision
    vision_profile: str = "default"
    detection_confidence: float = 0.8

    # Blackjack
    num_decks: int = 6
    min_bet: float = 10.0
    max_bet: float = 500.0
    bankroll: float = 5000.0

    # HUD
    hud_position: tuple[int, int] = (100, 100)
    hud_opacity: float = 0.85

    # Hotkeys
    hotkey_toggle_hud: str = "<ctrl>+<shift>+h"
    hotkey_calibrate: str = "<ctrl>+<shift>+c"
    hotkey_manual_card: str = "<ctrl>+<shift>+m"
    hotkey_reset_hand: str = "<ctrl>+<shift>+r"
    hotkey_switch_mode: str = "<ctrl>+<shift>+s"

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        """Load config from a JSON file.  Missing keys default gracefully."""
        p = Path(path)
        if not p.exists():
            logger.info("Config not found at %s — using defaults.", p)
            return cls()
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            cfg = cls()
            for key, value in data.items():
                if hasattr(cfg, key):
                    if key in ("capture_region", "hud_position") and isinstance(value, list):
                        value = tuple(value)
                    setattr(cfg, key, value)
            return cfg
        except Exception as exc:
            logger.error("Failed to load config from %s: %s — using defaults.", p, exc)
            return cls()

    def save(self, path: str | Path) -> None:
        """Persist config as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}
        try:
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            logger.debug("Config saved to %s.", p)
        except OSError as exc:
            logger.error("Could not save config: %s", exc)


# ---------------------------------------------------------------------------
# Card string parser
# ---------------------------------------------------------------------------

def _parse_card(card_str: str) -> Optional[Card]:
    """Parse a card string like ``"Ah"`` into a :class:`Card`.

    Returns ``None`` on parse failure so callers can handle gracefully.
    """
    if len(card_str) != 2:
        return None
    rank_char, suit_char = card_str[0].upper(), card_str[1].lower()
    rank_map = {r.value: r for r in Rank}
    suit_map = {s.value: s for s in Suit}
    rank = rank_map.get(rank_char)
    suit = suit_map.get(suit_char)
    if rank is None or suit is None:
        return None
    return Card(rank=rank, suit=suit)


# ---------------------------------------------------------------------------
# GameController
# ---------------------------------------------------------------------------

class GameController:
    """Central controller that manages game state and drives the processing loop.

    Attributes:
        poker_state: Current poker hand state.
        blackjack_state: Current blackjack hand state.
        mode: Active :class:`GameMode`.
    """

    def __init__(
        self,
        config: AppConfig,
        capture: ScreenCapture,
        template_engine: TemplateEngine,
        overlay: Optional["HUDOverlay"],
    ) -> None:
        self.config = config
        self.capture = capture
        self.template_engine = template_engine
        self.overlay = overlay

        # Game state
        self.poker_state = PokerState()
        self.blackjack_state = BlackjackState(num_decks=config.num_decks)
        self.mode = GameMode(config.game_mode) if config.game_mode in [m.value for m in GameMode] else GameMode.POKER

        # Cached results for re-use between frames
        self._last_poker_result = PokerResult()
        self._last_bj_result = BlackjackResult()

        # Stats
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Per-tick processing
    # ------------------------------------------------------------------

    def process_frame(self) -> None:
        """Execute one processing tick (called at ~30fps by QTimer).

        In poker mode:
            1. Grab latest frame from capture buffer.
            2. Run template detection.
            3. Update poker state from detected cards.
            4. Run poker math.
            5. Push result to HUD.

        In blackjack mode:
            1. Run blackjack math on current state (cards come from manual input).
            2. Push result to HUD.
        """
        self._frame_count += 1

        if self.mode == GameMode.POKER:
            self._tick_poker()
        else:
            self._tick_blackjack()

    def _tick_poker(self) -> None:
        """Process one poker frame."""
        frame = self.capture.get_latest_frame()
        if frame is not None:
            detections = self.template_engine.detect_cards(
                frame,
                confidence=self.config.detection_confidence,
            )
            if detections:
                self._apply_poker_detections(detections)

        result = analyze_poker(self.poker_state)
        self._last_poker_result = result

        if self.overlay is not None:
            self.overlay.update_poker(result)

    def _tick_blackjack(self) -> None:
        """Process one blackjack tick (state is updated via manual input)."""
        result = analyze_blackjack(self.blackjack_state)
        self._last_bj_result = result

        if self.overlay is not None:
            self.overlay.update_blackjack(result)

    def _apply_poker_detections(
        self,
        detections: list[tuple[Card, tuple[int, int, int, int], float]],
    ) -> None:
        """Merge vision detections into poker state.

        Simple heuristic: first 2 detected cards → hole cards (if not already
        populated); remaining → community cards.
        """
        cards = [card for card, _bbox, _conf in detections]
        if not cards:
            return
        if not self.poker_state.hole_cards and len(cards) >= 2:
            self.poker_state.hole_cards = cards[:2]
            remaining = cards[2:]
        else:
            remaining = cards

        for card in remaining:
            if card not in self.poker_state.community_cards:
                self.poker_state.community_cards.append(card)

        logger.debug(
            "Poker state — hole: %s, community: %s",
            [str(c) for c in self.poker_state.hole_cards],
            [str(c) for c in self.poker_state.community_cards],
        )

    # ------------------------------------------------------------------
    # Manual card input (blackjack)
    # ------------------------------------------------------------------

    def add_card_manual(self, card_str: str, target: str) -> None:
        """Handle a card from the :class:`ManualCardInput` dialog.

        Args:
            card_str: Card notation like ``"Ah"``, or ``"__RESET__"`` to
                clear the hand.
            target: One of ``"player"``, ``"dealer"``, or ``"seen"``.
        """
        if card_str == "__RESET__":
            self.reset_hand()
            return

        card = _parse_card(card_str)
        if card is None:
            logger.warning("Cannot parse card string: %r", card_str)
            return

        if target == "player":
            if card not in self.blackjack_state.player_hand:
                self.blackjack_state.player_hand.append(card)
                self.blackjack_state.cards_seen.append(card)
                logger.info("Player hand: %s added.", card_str)
        elif target == "dealer":
            self.blackjack_state.dealer_upcard = card
            self.blackjack_state.cards_seen.append(card)
            logger.info("Dealer upcard: %s.", card_str)
        elif target == "seen":
            if card not in self.blackjack_state.cards_seen:
                self.blackjack_state.cards_seen.append(card)
                logger.debug("Seen card: %s.", card_str)
        else:
            logger.warning("Unknown target %r for card %s.", target, card_str)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_hand(self) -> None:
        """Clear current hand state for the active game mode."""
        if self.mode == GameMode.POKER:
            self.poker_state = PokerState()
            logger.info("Poker hand reset.")
        else:
            # Keep cards_seen (shoe memory) but clear the hand itself.
            seen = list(self.blackjack_state.cards_seen)
            self.blackjack_state = BlackjackState(num_decks=self.config.num_decks)
            self.blackjack_state.cards_seen = seen
            logger.info("Blackjack hand reset (shoe memory preserved).")

    def switch_mode(self) -> None:
        """Toggle between poker and blackjack modes."""
        if self.mode == GameMode.POKER:
            self.mode = GameMode.BLACKJACK
        else:
            self.mode = GameMode.POKER

        logger.info("Mode switched to %s.", self.mode.value)

        if self.overlay is not None:
            self.overlay.set_mode(self.mode)


# ---------------------------------------------------------------------------
# Qt dark palette
# ---------------------------------------------------------------------------

def _apply_dark_palette(app: QApplication) -> None:
    """Apply a dark system palette to the QApplication."""
    palette = QPalette()
    dark = QColor(28, 28, 40)
    mid_dark = QColor(45, 45, 60)
    accent = QColor(0, 200, 83)
    text = QColor(240, 240, 248)
    muted = QColor(128, 128, 144)

    palette.setColor(QPalette.ColorRole.Window, dark)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, mid_dark)
    palette.setColor(QPalette.ColorRole.AlternateBase, dark)
    palette.setColor(QPalette.ColorRole.ToolTipBase, dark)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, mid_dark)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, accent)
    palette.setColor(QPalette.ColorRole.Link, accent)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.PlaceholderText, muted)
    app.setPalette(palette)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="riverrater",
        description="RiverRater — real-time casino game assistant",
    )
    parser.add_argument(
        "--mode",
        choices=["poker", "blackjack"],
        default="poker",
        help="Starting game mode (default: poker)",
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help=f"Path to JSON config file (default: {_DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    1. Parse arguments.
    2. Configure logging.
    3. Load (or create) config.
    4. Create QApplication with dark palette.
    5. Create HUD overlay.
    6. Create screen capture and vision engine.
    7. Create GameController.
    8. Register hotkeys.
    9. Start 30fps QTimer processing loop.
    10. Connect ManualCardInput signals.
    11. Run Qt event loop.

    Args:
        argv: Override sys.argv for testing (``None`` → use sys.argv).

    Returns:
        Exit code (0 = success).
    """
    args = _parse_args(argv)

    # -- Logging -------------------------------------------------------------
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(level=log_level, log_file="riverrater.log")
    logger.info("RiverRater starting (mode=%s, debug=%s)", args.mode, args.debug)

    # -- Config --------------------------------------------------------------
    config_path = Path(args.config)
    config = AppConfig.load(config_path)
    # CLI mode flag overrides config
    config.game_mode = args.mode
    # Save default config on first run
    if not config_path.exists():
        config.save(config_path)
        logger.info("Default config written to %s.", config_path)

    # -- QApplication --------------------------------------------------------
    app = QApplication.instance() or QApplication(sys.argv)
    assert isinstance(app, QApplication)
    app.setApplicationName("RiverRater")
    app.setApplicationVersion("0.1.0")
    app.setStyle("Fusion")  # Good dark-mode base
    _apply_dark_palette(app)

    # -- HUD overlay ---------------------------------------------------------
    overlay: Optional[HUDOverlay] = None
    if _HAS_HUD and HUDOverlay is not None:
        overlay = HUDOverlay()
        overlay.set_position(*config.hud_position)
        overlay.set_opacity(config.hud_opacity)
        overlay.set_mode(GameMode(config.game_mode))
        overlay.show()
        logger.info("HUD overlay created.")
    else:
        logger.warning("HUD overlay unavailable — running headless.")

    # -- Manual card input dialog (for blackjack) ----------------------------
    manual_input: Optional[ManualCardInput] = None
    if _HAS_HUD and ManualCardInput is not None:
        # Parent to overlay so it shares the same Qt hierarchy
        manual_input = ManualCardInput(overlay)
        logger.debug("ManualCardInput dialog created.")

    # -- Screen capture & vision engine -------------------------------------
    capture = ScreenCapture(region=config.capture_region)
    capture.start()

    template_engine = TemplateEngine()
    profile_paths = [
        Path(f"~/.riverrater/profiles/{config.vision_profile}.json").expanduser(),
        Path(f"/etc/riverrater/profiles/{config.vision_profile}.json"),
    ]
    for profile_path in profile_paths:
        if profile_path.exists():
            try:
                template_engine.load_profile(str(profile_path))
                logger.info("Loaded vision profile: %s", profile_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load vision profile %s: %s", profile_path, exc)
            break
    else:
        logger.info("No vision profile found — template engine running empty.")

    # -- Game controller -----------------------------------------------------
    controller = GameController(
        config=config,
        capture=capture,
        template_engine=template_engine,
        overlay=overlay,
    )

    # -- Manual card input signal connection ---------------------------------
    if manual_input is not None:
        manual_input.card_added.connect(controller.add_card_manual)

    # -- Hotkeys -------------------------------------------------------------
    hotkeys = HotkeyManager()

    def _toggle_hud() -> None:
        if overlay is not None:
            # Qt widgets must be touched on the main thread
            QMetaObject.invokeMethod(overlay, "toggle_visibility", Qt.ConnectionType.QueuedConnection)

    def _calibrate() -> None:
        logger.info(
            "Calibration mode not yet wired — draw box in capture region"
        )

    def _show_manual_input() -> None:
        if manual_input is not None:
            QMetaObject.invokeMethod(manual_input, "show", Qt.ConnectionType.QueuedConnection)
            QMetaObject.invokeMethod(manual_input, "raise_", Qt.ConnectionType.QueuedConnection)
            QMetaObject.invokeMethod(manual_input, "activateWindow", Qt.ConnectionType.QueuedConnection)

    def _reset_hand() -> None:
        # Controller access is thread-safe (only reads/writes Python objects)
        controller.reset_hand()
        logger.info("Hand reset via hotkey.")

    def _switch_mode() -> None:
        controller.switch_mode()
        if overlay is not None:
            QMetaObject.invokeMethod(
                overlay,
                "set_mode",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG("PyQt_PyObject", controller.mode),
            )

    hotkeys.register(config.hotkey_toggle_hud, _toggle_hud)
    hotkeys.register(config.hotkey_calibrate, _calibrate)
    hotkeys.register(config.hotkey_manual_card, _show_manual_input)
    hotkeys.register(config.hotkey_reset_hand, _reset_hand)
    hotkeys.register(config.hotkey_switch_mode, _switch_mode)
    hotkeys.start()

    logger.info(
        "Hotkeys registered: toggle=%s, calibrate=%s, manual=%s, reset=%s, switch=%s",
        config.hotkey_toggle_hud,
        config.hotkey_calibrate,
        config.hotkey_manual_card,
        config.hotkey_reset_hand,
        config.hotkey_switch_mode,
    )

    # -- Processing timer (30fps) -------------------------------------------
    fps = max(1, config.capture_fps_target)
    interval_ms = int(1000 / fps)
    timer = QTimer()
    timer.setInterval(interval_ms)
    timer.timeout.connect(controller.process_frame)
    timer.start()
    logger.info("Processing timer started at %d fps (%d ms interval).", fps, interval_ms)

    # -- Graceful Ctrl+C handling -------------------------------------------
    def _sigint_handler(signum, frame):  # noqa: ANN001
        logger.info("SIGINT received — shutting down.")
        timer.stop()
        hotkeys.stop()
        capture.stop()
        app.quit()

    signal.signal(signal.SIGINT, _sigint_handler)

    # Allow Python to process SIGINT while Qt is running by polling every 250ms
    interrupt_timer = QTimer()
    interrupt_timer.setInterval(250)
    interrupt_timer.timeout.connect(lambda: None)  # Wakes Python event loop
    interrupt_timer.start()

    logger.info("Entering Qt event loop.")
    exit_code = app.exec()

    # -- Teardown ------------------------------------------------------------
    logger.info("Qt event loop exited (code %d).", exit_code)
    timer.stop()
    hotkeys.stop()
    capture.stop()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
