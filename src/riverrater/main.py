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
from riverrater.game.state import (
    BlackjackAction,
    BlackjackResult,
    BlackjackState,
    Card,
    DetectionMeta,
    GameMode,
    PokerResult,
    PokerState,
    Rank,
    Suit,
)

# -- Math engines ------------------------------------------------------------
from riverrater.game.poker_math import analyze_poker, clear_equity_cache, recompute_poker_ev
from riverrater.game.blackjack_math import analyze_blackjack

# -- Screen capture ----------------------------------------------------------
from riverrater.capture.screen import ScreenCapture

# -- Vision engines ----------------------------------------------------------
from riverrater.training.yolo_train import default_weights_path
from riverrater.vision.card_tracker import CardTracker
from riverrater.vision.template_engine import TemplateEngine
from riverrater.vision.yolo_engine import YOLOEngine
from riverrater.vision.pot_ocr import PotOCR, resolve_pot_rois

# -- HUD overlay -------------------------------------------------------------
try:
    from riverrater.hud.overlay import HUDOverlay
    from riverrater.hud.manual_input import ManualCardInput
    from riverrater.hud.poker_input import PokerInputDialog
    from riverrater.hud.settings_dialog import SettingsDialog
    _HAS_HUD = True
except ImportError as _e:
    logger.warning("hud not available: %s", _e)
    _HAS_HUD = False
    HUDOverlay = None  # type: ignore[assignment,misc]
    ManualCardInput = None  # type: ignore[assignment,misc]
    PokerInputDialog = None  # type: ignore[assignment,misc]
    SettingsDialog = None  # type: ignore[assignment,misc]

try:
    from riverrater.hud.calibration_overlay import CalibrationOverlay
    _HAS_CALIBRATION = True
except ImportError as _e:
    logger.warning("calibration overlay not available: %s", _e)
    _HAS_CALIBRATION = False
    CalibrationOverlay = None  # type: ignore[assignment,misc]

# -- Hotkeys -----------------------------------------------------------------
from riverrater.utils.hotkeys import HotkeyManager
from riverrater.utils.session_log import SessionLogger

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
    yolo_model_path: Optional[str] = None
    yolo_confidence: float = 0.5
    pot_roi: Optional[tuple[int, int, int, int]] = None
    bet_roi: Optional[tuple[int, int, int, int]] = None
    pot_ocr_confidence: float = 0.6
    pot_ocr_enabled: bool = True

    # Blackjack
    num_decks: int = 6
    min_bet: float = 10.0
    max_bet: float = 500.0
    bankroll: float = 5000.0

    # HUD
    hud_position: tuple[int, int] = (100, 100)
    hud_opacity: float = 0.85

    # Poker
    num_opponents: int = 1

    # Hotkeys
    hotkey_toggle_hud: str = "<ctrl>+<shift>+h"
    hotkey_calibrate: str = "<ctrl>+<shift>+c"
    hotkey_manual_card: str = "<ctrl>+<shift>+m"
    hotkey_reset_hand: str = "<ctrl>+<shift>+r"
    hotkey_switch_mode: str = "<ctrl>+<shift>+s"
    hotkey_reset_shoe: str = "<ctrl>+<shift>+n"  # "New shoe"
    hotkey_poker_input: str = "<ctrl>+<shift>+p"
    hotkey_settings: str = "<ctrl>+<shift>+o"

    # Session logging
    session_logging: bool = True

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
                    if key in ("capture_region", "hud_position", "pot_roi", "bet_roi") and isinstance(value, list):
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
# Poker state fingerprinting (skip redundant analyze/HUD on unchanged ticks)
# ---------------------------------------------------------------------------

def _poker_equity_hash(state: PokerState) -> tuple:
    """Hashable fingerprint of fields that affect Monte Carlo equity."""
    return (
        tuple(state.hole_cards),
        tuple(state.community_cards),
        state.num_opponents,
    )


def _poker_full_hash(state: PokerState) -> tuple:
    """Full fingerprint including pot/bet fields that affect EV only."""
    return (
        *_poker_equity_hash(state),
        state.pot_size,
        state.bet_to_call,
    )


def _detection_meta_fingerprint(meta: Optional[DetectionMeta]) -> Optional[tuple]:
    """Hashable fingerprint for vision confidence metadata shown on the HUD."""
    if meta is None:
        return None
    return (
        meta.overall_confidence,
        frozenset(meta.card_confidences.items()),
        meta.is_manual,
    )


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
        pot_ocr: Optional[PotOCR] = None,
        yolo_engine: Optional[YOLOEngine] = None,
        session_logger: Optional[SessionLogger] = None,
        config_path: Optional[str | Path] = None,
    ) -> None:
        self.config = config
        self._config_path = Path(config_path) if config_path is not None else None
        self.capture = capture
        self.template_engine = template_engine
        self.overlay = overlay
        self.pot_ocr = pot_ocr
        self.yolo_engine = yolo_engine
        self._session_logger = session_logger or SessionLogger(
            enabled=config.session_logging,
        )

        # Game state
        self.poker_state = PokerState(num_opponents=config.num_opponents)
        self.blackjack_state = BlackjackState(num_decks=config.num_decks)
        self.mode = GameMode(config.game_mode) if config.game_mode in [m.value for m in GameMode] else GameMode.POKER

        # Cached results for re-use between frames
        self._last_poker_result = PokerResult()
        self._last_bj_result = BlackjackResult()

        # Poker tick cache — split equity vs pot/bet for fast EV-only updates
        self._last_equity_hash: Optional[tuple] = None
        self._last_full_hash: Optional[tuple] = None
        self._last_detection_meta_fingerprint: Optional[tuple] = None

        # Frame-skip: only run expensive template matching when the frame
        # has changed significantly.  Stores a downsampled grayscale snapshot.
        self._prev_frame_gray: Optional["np.ndarray"] = None
        self._frame_change_threshold: float = 5.0  # Mean absolute pixel diff
        self._detect_every_n: int = 3  # Run detection at most every N ticks

        # Detection confidence metadata
        self._detection_meta: Optional[DetectionMeta] = None

        # Manual pot/bet input overrides OCR until the hand is reset.
        self._poker_values_manual: bool = False

        # Cross-frame YOLO deduplication for blackjack card counting
        self._card_tracker = CardTracker()

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
            1. When YOLO is available, detect cards from the latest frame.
            2. Otherwise cards come from manual input only.
            3. Run blackjack math on current state.
            4. Push result to HUD.
        """
        self._frame_count += 1

        if self.mode == GameMode.POKER:
            self._tick_poker()
        else:
            self._tick_blackjack()

    def _tick_poker(self) -> None:
        """Process one poker frame.

        Runs template detection only when the captured frame has changed
        significantly from the previous one (mean pixel delta above
        threshold) and at most every ``_detect_every_n`` ticks.  This
        avoids burning CPU on template matching when the screen is idle.
        """
        frame = self.capture.get_latest_frame()
        should_detect = False

        if frame is not None and self._frame_count % self._detect_every_n == 0:
            should_detect = self._frame_changed(frame)

        if should_detect and frame is not None:
            detections = self.template_engine.detect_cards(
                frame,
                confidence=self.config.detection_confidence,
            )
            if detections:
                self._apply_poker_detections(detections)

            # Gate pot OCR on card hash — only read pot/bet once hole cards exist.
            if (
                self.pot_ocr is not None
                and self.config.pot_ocr_enabled
                and not self._poker_values_manual
                and len(self.poker_state.hole_cards) >= 2
            ):
                self._apply_pot_ocr(frame)

        equity_hash = _poker_equity_hash(self.poker_state)
        full_hash = _poker_full_hash(self.poker_state)
        meta_fingerprint = _detection_meta_fingerprint(self._detection_meta)
        equity_changed = equity_hash != self._last_equity_hash
        pot_changed = full_hash != self._last_full_hash
        meta_changed = meta_fingerprint != self._last_detection_meta_fingerprint
        is_first_tick = self._last_equity_hash is None

        if equity_changed or is_first_tick:
            result = analyze_poker(self.poker_state)
            result.detection_meta = self._detection_meta
            self._last_poker_result = result
            self._last_equity_hash = equity_hash
            self._last_full_hash = full_hash
            self._last_detection_meta_fingerprint = meta_fingerprint
        elif pot_changed:
            result = recompute_poker_ev(
                self._last_poker_result.win_pct,
                self._last_poker_result.tie_pct,
                self.poker_state.pot_size,
                self.poker_state.bet_to_call,
            )
            result.detection_meta = self._detection_meta
            self._last_poker_result = result
            self._last_full_hash = full_hash
            self._last_detection_meta_fingerprint = meta_fingerprint
        elif meta_changed:
            self._last_poker_result.detection_meta = self._detection_meta
            self._last_detection_meta_fingerprint = meta_fingerprint
        else:
            return

        self._session_logger.log_poker(self.poker_state, self._last_poker_result)

        if self.overlay is not None:
            self.overlay.update_poker(self._last_poker_result)

    def _frame_changed(self, frame: "np.ndarray") -> bool:
        """Return True if *frame* differs enough from the previous to warrant re-detection."""
        try:
            import cv2
            small = cv2.resize(frame, (64, 48))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if self._prev_frame_gray is None:
                self._prev_frame_gray = gray
                return True

            diff = cv2.absdiff(gray, self._prev_frame_gray)
            mean_diff = float(diff.mean())
            self._prev_frame_gray = gray
            return mean_diff > self._frame_change_threshold
        except Exception:
            return True  # If anything fails, run detection anyway

    def _tick_blackjack(self) -> None:
        """Process one blackjack tick.

        Uses YOLO card detection when :attr:`yolo_engine` is loaded and
        :attr:`~riverrater.vision.yolo_engine.YOLOEngine.is_available` is
        ``True``; otherwise state is updated only via manual input.
        """
        if self.yolo_engine is not None and self.yolo_engine.is_available:
            frame = self.capture.get_latest_frame()
            should_detect = False

            if frame is not None and self._frame_count % self._detect_every_n == 0:
                should_detect = self._frame_changed(frame)

            if should_detect and frame is not None:
                detections = self.yolo_engine.detect_cards(
                    frame,
                    confidence=self.config.yolo_confidence,
                )
                if detections:
                    self._apply_blackjack_detections(detections)

        result = analyze_blackjack(
            self.blackjack_state,
            min_bet=self.config.min_bet,
            max_bet=self.config.max_bet,
            bankroll=self.config.bankroll,
        )
        self._last_bj_result = result
        self._session_logger.log_blackjack(self.blackjack_state, result)

        if self.overlay is not None:
            self.overlay.update_blackjack(result)

    def _apply_blackjack_detections(
        self,
        detections: list[tuple[Card, tuple[int, int, int, int], float]],
    ) -> None:
        """Merge YOLO detections into blackjack state.

        Cards are sorted top-to-bottom by bbox *y* (dealer upcard is typically
        above the player hand on live-dealer layouts).  Cross-frame
        :class:`~riverrater.vision.card_tracker.CardTracker` deduplication
        ensures each physical card is added to ``cards_seen`` once, while
        still counting duplicate rank/suit copies dealt to different positions
        in a multi-deck shoe.
        """
        if not detections:
            return

        self._detection_meta = DetectionMeta.from_detections(detections)

        new_detections = self._card_tracker.register_detections(detections)
        for card, _bbox, _conf in new_detections:
            self.blackjack_state.cards_seen.append(card)

        sorted_dets = sorted(detections, key=lambda item: item[1][1])

        if self.blackjack_state.dealer_upcard is None and sorted_dets:
            self.blackjack_state.dealer_upcard = sorted_dets[0][0]
            logger.debug(
                "Dealer upcard from YOLO: %s",
                self.blackjack_state.dealer_upcard,
            )

        for card, _bbox, _conf in sorted_dets[1:]:
            if card not in self.blackjack_state.player_hand:
                self.blackjack_state.player_hand.append(card)

        logger.debug(
            "Blackjack state — player: %s, dealer: %s, seen: %d",
            [str(c) for c in self.blackjack_state.player_hand],
            self.blackjack_state.dealer_upcard,
            len(self.blackjack_state.cards_seen),
        )

    def _apply_poker_detections(
        self,
        detections: list[tuple[Card, tuple[int, int, int, int], float]],
    ) -> None:
        """Merge vision detections into poker state.

        Simple heuristic: first 2 detected cards → hole cards (if not already
        populated); remaining → community cards.
        """
        self._detection_meta = DetectionMeta.from_detections(detections)
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

    def _apply_pot_ocr(self, frame: "np.ndarray") -> None:
        """Update pot/bet fields from calibrated OCR regions when confident."""
        pot_result, bet_result = self.pot_ocr.read_values(frame)
        updated = False

        if pot_result is not None:
            self.poker_state.pot_size = pot_result.value
            updated = True
            logger.debug(
                "Pot OCR: pot=%.2f (confidence=%.3f)",
                pot_result.value,
                pot_result.confidence,
            )

        if bet_result is not None:
            self.poker_state.bet_to_call = bet_result.value
            updated = True
            logger.debug(
                "Pot OCR: bet=%.2f (confidence=%.3f)",
                bet_result.value,
                bet_result.confidence,
            )

        if updated:
            self._invalidate_poker_tick_cache()

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

        self._detection_meta = DetectionMeta.manual()

        if target == "player":
            if card not in self.blackjack_state.player_hand:
                self.blackjack_state.player_hand.append(card)
                # Only add to seen if not already tracked (prevents count corruption)
                if card not in self.blackjack_state.cards_seen:
                    self.blackjack_state.cards_seen.append(card)
                logger.info("Player hand: %s added.", card_str)
            else:
                logger.debug("Player card %s already in hand — skipping.", card_str)
        elif target == "dealer":
            self.blackjack_state.dealer_upcard = card
            if card not in self.blackjack_state.cards_seen:
                self.blackjack_state.cards_seen.append(card)
            logger.info("Dealer upcard: %s.", card_str)
        elif target == "seen":
            if card not in self.blackjack_state.cards_seen:
                self.blackjack_state.cards_seen.append(card)
                logger.debug("Seen card: %s.", card_str)
            else:
                logger.debug("Card %s already seen — skipping.", card_str)
        else:
            logger.warning("Unknown target %r for card %s.", target, card_str)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _invalidate_poker_tick_cache(self) -> None:
        """Force analyze_poker and HUD refresh on the next poker tick."""
        self._last_equity_hash = None
        self._last_full_hash = None
        self._last_detection_meta_fingerprint = None

    def _persist_num_opponents(self, num_opponents: int) -> None:
        """Update config and save last opponent count when a path is configured."""
        self.config.num_opponents = num_opponents
        if self._config_path is not None:
            self.config.save(self._config_path)

    def set_num_opponents(self, num_opponents: int) -> None:
        """Update active opponent count and force equity recalculation."""
        clamped = max(1, min(9, int(num_opponents)))
        if self.poker_state.num_opponents == clamped:
            return
        self.poker_state.num_opponents = clamped
        self._persist_num_opponents(clamped)
        self._invalidate_poker_tick_cache()
        logger.info("Opponent count set to %d.", clamped)

    def reset_hand(self) -> None:
        """Clear current hand state for the active game mode."""
        if self.mode == GameMode.POKER:
            self.poker_state = PokerState(num_opponents=self.config.num_opponents)
            self._poker_values_manual = False
            clear_equity_cache()
            self._invalidate_poker_tick_cache()
            self._session_logger.reset_poker_cache()
            logger.info("Poker hand reset.")
        else:
            # Keep cards_seen (shoe memory) but clear the hand itself.
            seen = list(self.blackjack_state.cards_seen)
            self.blackjack_state = BlackjackState(num_decks=self.config.num_decks)
            self.blackjack_state.cards_seen = seen
            self._session_logger.reset_blackjack_cache()
            logger.info("Blackjack hand reset (shoe memory preserved).")

    def reset_shoe(self) -> None:
        """Clear the entire shoe — running count, cards seen, everything.

        Use when the dealer shuffles a new shoe.
        """
        self.blackjack_state = BlackjackState(num_decks=self.config.num_decks)
        self._card_tracker.reset()
        self._session_logger.reset_blackjack_cache()
        logger.info("Shoe reset — all counts cleared.")

    def set_poker_values(self, pot_size: float, bet_to_call: float, num_opponents: int) -> None:
        """Update poker state with manually entered pot/bet values.

        Args:
            pot_size: Current pot size.
            bet_to_call: Amount needed to call.
            num_opponents: Number of active opponents.
        """
        self.poker_state.pot_size = pot_size
        self.poker_state.bet_to_call = bet_to_call
        self.poker_state.num_opponents = max(1, min(9, int(num_opponents)))
        self._poker_values_manual = True
        self._persist_num_opponents(self.poker_state.num_opponents)
        self._invalidate_poker_tick_cache()
        logger.info(
            "Poker values set: pot=%.2f, bet=%.2f, opp=%d",
            pot_size, bet_to_call, num_opponents,
        )

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
        overlay = HUDOverlay(num_opponents=config.num_opponents)
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
    loaded_profile_path: Optional[Path] = None
    profile_paths = [
        Path(f"~/.riverrater/profiles/{config.vision_profile}.json").expanduser(),
        Path(f"/etc/riverrater/profiles/{config.vision_profile}.json"),
    ]
    for profile_path in profile_paths:
        if profile_path.exists():
            try:
                template_engine.load_profile(str(profile_path))
                loaded_profile_path = profile_path
                logger.info("Loaded vision profile: %s", profile_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load vision profile %s: %s", profile_path, exc)
            break
    else:
        logger.info("No vision profile found — template engine running empty.")

    pot_roi, bet_roi = resolve_pot_rois(
        config.pot_roi,
        config.bet_roi,
        loaded_profile_path,
    )
    pot_ocr: Optional[PotOCR] = None
    if config.pot_ocr_enabled and (pot_roi is not None or bet_roi is not None):
        pot_ocr = PotOCR(
            pot_roi=pot_roi,
            bet_roi=bet_roi,
            confidence_threshold=config.pot_ocr_confidence,
        )
        logger.info(
            "Pot OCR enabled (pot_roi=%s, bet_roi=%s, threshold=%.2f)",
            pot_roi,
            bet_roi,
            config.pot_ocr_confidence,
        )

    # -- YOLO engine (blackjack vision) --------------------------------------
    yolo_weights = config.yolo_model_path or str(default_weights_path())
    yolo_engine = YOLOEngine(yolo_weights)
    if yolo_engine.is_available:
        logger.info("YOLO engine ready (%s).", yolo_weights)
    else:
        logger.info(
            "YOLO unavailable at %s — blackjack uses manual input fallback.",
            yolo_weights,
        )

    # -- Game controller -----------------------------------------------------
    controller = GameController(
        config=config,
        capture=capture,
        template_engine=template_engine,
        overlay=overlay,
        pot_ocr=pot_ocr,
        yolo_engine=yolo_engine,
        config_path=config_path,
    )

    # -- Poker input dialog (for poker mode) ----------------------------------
    poker_input: Optional[PokerInputDialog] = None
    if _HAS_HUD and PokerInputDialog is not None:
        poker_input = PokerInputDialog(overlay, num_opponents=config.num_opponents)

        def _on_poker_values_submitted(
            pot_size: float,
            bet_to_call: float,
            num_opponents: int,
        ) -> None:
            controller.set_poker_values(pot_size, bet_to_call, num_opponents)
            if overlay is not None:
                overlay.poker_view.set_num_opponents(
                    controller.poker_state.num_opponents,
                )

        poker_input.values_submitted.connect(_on_poker_values_submitted)
        logger.debug("PokerInputDialog created.")

    def _sync_opponent_count(count: int) -> None:
        """Keep HUD stepper, dialog, and game state aligned."""
        controller.set_num_opponents(count)
        if overlay is not None:
            overlay.poker_view.set_num_opponents(controller.poker_state.num_opponents)
        if poker_input is not None:
            poker_input.num_opponents = controller.poker_state.num_opponents

    if overlay is not None:
        overlay.poker_view.num_opponents_changed.connect(_sync_opponent_count)

    # -- Settings dialog ------------------------------------------------------
    settings_dialog: Optional[SettingsDialog] = None

    def _apply_settings(updated: AppConfig) -> None:
        nonlocal pot_ocr, yolo_engine, loaded_profile_path

        config.__dict__.update(updated.__dict__)

        if overlay is not None:
            overlay.set_opacity(config.hud_opacity)
            overlay.set_position(*config.hud_position)

        with capture._frame_lock:
            capture._region = config.capture_region

        if controller.blackjack_state.num_decks != config.num_decks:
            controller.blackjack_state.num_decks = config.num_decks

        profile_paths = [
            Path(f"~/.riverrater/profiles/{config.vision_profile}.json").expanduser(),
            Path(f"/etc/riverrater/profiles/{config.vision_profile}.json"),
        ]
        for profile_path in profile_paths:
            if profile_path.exists():
                try:
                    template_engine.load_profile(str(profile_path))
                    loaded_profile_path = profile_path
                    logger.info("Reloaded vision profile: %s", profile_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to reload vision profile %s: %s",
                        profile_path,
                        exc,
                    )
                break

        pot_roi, bet_roi = resolve_pot_rois(
            config.pot_roi,
            config.bet_roi,
            loaded_profile_path,
        )
        if config.pot_ocr_enabled and (pot_roi is not None or bet_roi is not None):
            pot_ocr = PotOCR(
                pot_roi=pot_roi,
                bet_roi=bet_roi,
                confidence_threshold=config.pot_ocr_confidence,
            )
            controller.pot_ocr = pot_ocr
            logger.info("Pot OCR reconfigured from settings.")
        else:
            pot_ocr = None
            controller.pot_ocr = None
            logger.info("Pot OCR disabled from settings.")

        yolo_weights = config.yolo_model_path or str(default_weights_path())
        new_yolo = YOLOEngine(yolo_weights)
        if new_yolo.is_available:
            yolo_engine = new_yolo
            controller.yolo_engine = yolo_engine
            logger.info("YOLO engine reconfigured from settings (%s).", yolo_weights)
        elif yolo_engine is None:
            controller.yolo_engine = new_yolo
            logger.info(
                "YOLO unavailable at %s — blackjack uses manual input fallback.",
                yolo_weights,
            )
        else:
            logger.warning(
                "YOLO unavailable at %s — keeping previous engine.",
                yolo_weights,
            )

        logger.info("Settings applied.")

    if _HAS_HUD and SettingsDialog is not None:
        settings_dialog = SettingsDialog(config, config_path, overlay)
        settings_dialog.settings_saved.connect(_apply_settings)
        logger.debug("SettingsDialog created.")

    # -- Manual card input signal connection ---------------------------------
    if manual_input is not None:
        manual_input.card_added.connect(controller.add_card_manual)
        manual_input.shoe_reset.connect(controller.reset_shoe)

    # -- Hotkeys -------------------------------------------------------------
    hotkeys = HotkeyManager()

    def _toggle_hud() -> None:
        if overlay is not None:
            # Qt widgets must be touched on the main thread
            QMetaObject.invokeMethod(overlay, "toggle_visibility", Qt.ConnectionType.QueuedConnection)

    def _calibrate() -> None:
        if not _HAS_CALIBRATION or CalibrationOverlay is None:
            logger.warning("Calibration overlay not available.")
            return

        def _calibrate_on_main_thread() -> None:
            frame = capture.grab_frame() if hasattr(capture, "grab_frame") else capture.get_latest_frame()
            if frame is None:
                logger.warning("Cannot calibrate — no frame available.")
                return

            timer.stop()

            profile_path = str(Path("~/.riverrater/profiles").expanduser() / config.vision_profile)

            cal_overlay = CalibrationOverlay(
                frame=frame,
                template_engine=template_engine,
                profile_path=profile_path,
            )

            def _on_cal_finished() -> None:
                timer.start()
                logger.info("Calibration complete — templates saved.")

            def _on_cal_cancelled() -> None:
                timer.start()
                logger.info("Calibration cancelled.")

            cal_overlay.calibration_finished.connect(_on_cal_finished)
            cal_overlay.calibration_cancelled.connect(_on_cal_cancelled)
            cal_overlay.show()

        QTimer.singleShot(0, _calibrate_on_main_thread)

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

    def _show_poker_input() -> None:
        if poker_input is not None:
            QMetaObject.invokeMethod(poker_input, "show", Qt.ConnectionType.QueuedConnection)
            QMetaObject.invokeMethod(poker_input, "raise_", Qt.ConnectionType.QueuedConnection)
            QMetaObject.invokeMethod(poker_input, "activateWindow", Qt.ConnectionType.QueuedConnection)

    def _show_settings() -> None:
        if settings_dialog is not None:
            def _open_on_main_thread() -> None:
                settings_dialog.load_from_config(config)
                settings_dialog.show()
                settings_dialog.raise_()
                settings_dialog.activateWindow()

            QTimer.singleShot(0, _open_on_main_thread)

    def _on_degradation_fix(action: str) -> None:
        """Route degradation Fix button to calibration or manual poker input."""
        if action == "poker_input":
            _show_poker_input()
        else:
            _calibrate()

    if overlay is not None:
        overlay.poker_view.degradation_fix_requested.connect(_on_degradation_fix)

    hotkeys.register(config.hotkey_toggle_hud, _toggle_hud)
    hotkeys.register(config.hotkey_calibrate, _calibrate)
    hotkeys.register(config.hotkey_manual_card, _show_manual_input)
    hotkeys.register(config.hotkey_reset_hand, _reset_hand)
    def _reset_shoe() -> None:
        controller.reset_shoe()
        logger.info("Shoe reset via hotkey.")

    hotkeys.register(config.hotkey_switch_mode, _switch_mode)
    hotkeys.register(config.hotkey_reset_shoe, _reset_shoe)
    hotkeys.register(config.hotkey_poker_input, _show_poker_input)
    hotkeys.register(config.hotkey_settings, _show_settings)
    hotkeys.start()

    logger.info(
        "Hotkeys registered: toggle=%s, calibrate=%s, manual=%s, reset=%s, "
        "switch=%s, shoe=%s, poker_input=%s, settings=%s",
        config.hotkey_toggle_hud,
        config.hotkey_calibrate,
        config.hotkey_manual_card,
        config.hotkey_reset_hand,
        config.hotkey_switch_mode,
        config.hotkey_reset_shoe,
        config.hotkey_poker_input,
        config.hotkey_settings,
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
