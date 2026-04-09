# RiverRater Module Interface Contracts

This document defines the exact interfaces between modules so they can be built and tested independently.

> **Last updated:** After fixes 1-8, confidence display (P0+P1), and calibration GUI (P0).

## Shared Data Types (src/riverrater/game/state.py)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class Suit(Enum):
    HEARTS = "h"
    DIAMONDS = "d"
    CLUBS = "c"
    SPADES = "s"

class Rank(Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"

@dataclass
class Card:
    rank: Rank
    suit: Suit
    
    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.value}"
    
    def __hash__(self) -> int:
        return hash((self.rank, self.suit))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    @classmethod
    def from_string(cls, s: str) -> "Card":
        """Parse a card string like 'Ah', 'Td', '2c'. Handles '10' as 'T'."""

    @property
    def blackjack_value(self) -> int:
        """Ace=11, face cards=10, pips=face value."""

class PokerAction(Enum):
    FOLD = "fold"
    CALL = "call"
    RAISE = "raise"

class BlackjackAction(Enum):
    HIT = "Hit"
    STAND = "Stand"
    DOUBLE = "Double"
    SPLIT = "Split"
    SURRENDER = "Surrender"

class GameMode(Enum):
    POKER = "poker"
    BLACKJACK = "blackjack"

@dataclass
class PokerState:
    hole_cards: list[Card] = field(default_factory=list)       # Player's 2 cards
    community_cards: list[Card] = field(default_factory=list)  # 0, 3, 4, or 5 cards
    pot_size: float = 0.0
    bet_to_call: float = 0.0
    num_opponents: int = 1

@dataclass 
class BlackjackState:
    player_hand: list[Card] = field(default_factory=list)
    dealer_upcard: Optional[Card] = None
    cards_seen: list[Card] = field(default_factory=list)  # All cards dealt from shoe
    num_decks: int = 6  # Standard shoe size

@dataclass
class DetectionMeta:
    """Metadata about vision detection confidence."""
    card_confidences: dict[str, float] = field(default_factory=dict)  # card_str -> score 0.0-1.0
    overall_confidence: float = 0.0  # Average of all card_confidences
    is_manual: bool = False  # True when cards came from manual input

    @classmethod
    def from_detections(cls, detections: list[tuple]) -> "DetectionMeta":
        """Build from TemplateEngine detection tuples (Card, bbox, confidence)."""

    @classmethod
    def manual(cls) -> "DetectionMeta":
        """Create a DetectionMeta indicating manual input (overall=1.0, is_manual=True)."""

@dataclass
class PokerResult:
    win_pct: float = 0.0
    tie_pct: float = 0.0
    required_equity: float = 0.0     # pot_odds as equity threshold
    actual_equity: float = 0.0       # win_pct + tie_pct/2
    ev_call: float = 0.0
    ev_fold: float = 0.0
    ev_raise: float = 0.0
    recommended_action: Optional[PokerAction] = None
    detection_meta: Optional[DetectionMeta] = None  # Confidence from vision pipeline

@dataclass
class BlackjackResult:
    running_count: int = 0
    true_count: float = 0.0
    recommended_action: Optional[BlackjackAction] = None
    recommended_bet: float = 0.0     # Kelly-based bet size
    shoe_favorability: float = 0.0   # 0.0 to 1.0 for heat meter
    hand_total: int = 0
    is_soft: bool = False            # Contains usable ace

FULL_DECK: list[Card]  # All 52 cards, computed at module level
```

## Module Interfaces

### 1. Screen Capture (src/riverrater/capture/screen.py)

```python
class ScreenCapture:
    def __init__(self, region: tuple[int,int,int,int] | None = None):
        """region = (left, top, width, height) or None for full screen.
        Creates a persistent mss.mss() context for the capture loop."""
    
    def set_region(self, region: tuple[int,int,int,int]) -> None:
        """Update capture region."""
    
    def grab_frame(self, sct=None) -> np.ndarray:
        """Capture single frame, return as BGR numpy array.
        Accepts optional sct (mss context) for reuse in tight loops."""
    
    def get_fps(self) -> float:
        """Return current capture FPS."""
    
    def start(self) -> None:
        """Start continuous capture in background thread with persistent mss context."""
    
    def stop(self) -> None:
        """Stop capture and release mss context."""
    
    def get_latest_frame(self) -> np.ndarray | None:
        """Get most recent frame from buffer (non-blocking)."""
```

### 2. Vision Engines

**Template Engine (src/riverrater/vision/template_engine.py):**
```python
class TemplateEngine:
    def __init__(self, profile_path: str | None = None):
        """Load vision profile from JSON, or start empty."""
    
    def detect_cards(self, frame: np.ndarray, confidence: float = 0.8) -> list[tuple[Card, tuple[int,int,int,int], float]]:
        """Returns list of (Card, bounding_box, confidence_score).
        Uses multi-scale matching + non-maximum suppression."""
    
    def add_template(self, card: Card, template_image: np.ndarray) -> None:
        """Add/update a card template (stored as grayscale)."""
    
    def remove_template(self, card: Card) -> None:
        """Remove templates for a card. No-op if card not found."""
    
    def save_profile(self, path: str) -> None:
        """Save current templates + config to profile directory."""
    
    def load_profile(self, path: str) -> None:
        """Load templates + config from profile directory."""
```

**YOLO Engine Stub (src/riverrater/vision/yolo_engine.py):**
```python
class YOLOEngine:
    """Stub — same detect_cards interface as TemplateEngine.
    Will be implemented when trained model is available.
    
    CLASS_MAP dict maps YOLO class indices to (Rank, Suit) tuples.
    """
    
    def __init__(self, model_path: str | None = None):
        """Load YOLO model weights, or None if not yet available.
        try/except ImportError handles missing ultralytics gracefully."""
        self._model = None  # Will be ultralytics.YOLO when ready
    
    def detect_cards(self, frame: np.ndarray, confidence: float = 0.5) -> list[tuple[Card, tuple[int,int,int,int], float]]:
        """Returns same format as TemplateEngine — raises NotImplementedError for now."""
        raise NotImplementedError("YOLO model not yet trained. Use manual input mode.")
    
    @property
    def is_available(self) -> bool:
        return self._model is not None
```

**Calibration Session (src/riverrater/vision/calibration.py):**
```python
class CalibrationCapture:
    """Manages capturing a cropped card image from a frame at a given bbox."""
    
    def crop(self, frame: np.ndarray, bbox: tuple[int,int,int,int]) -> np.ndarray:
        """Crop and return the card region from the frame."""

class CalibrationSession:
    """Collects card calibration entries and commits them to a TemplateEngine."""
    
    def add_calibration(self, card_str: str, bbox: tuple[int,int,int,int], frame: np.ndarray) -> None:
        """Add a calibration entry (card string + bbox + source frame)."""
    
    def finish(self, engine: TemplateEngine) -> None:
        """Commit all entries to the TemplateEngine as templates."""
    
    def cancel(self) -> None:
        """Discard all pending calibration entries."""
```

### 3. Math Engines

**Poker Math (src/riverrater/game/poker_math.py):**
```python
def calculate_equity(hole_cards: list[Card], community_cards: list[Card], num_opponents: int = 1, simulations: int = 5000) -> tuple[float, float]:
    """Monte Carlo simulation. Returns (win_pct, tie_pct)."""

def pot_odds(pot_size: float, bet_to_call: float) -> float:
    """Returns required equity as decimal (0.0-1.0)."""

def expected_value(equity: float, pot_size: float, bet_to_call: float) -> tuple[float, float, float]:
    """Returns (ev_call, ev_fold, ev_raise).
    ev_raise = 1.5 * ev_call when ev_call > 0; ev_call * 2.0 when ev_call < 0."""

def analyze_poker(state: PokerState) -> PokerResult:
    """Full analysis — calls all above, returns populated PokerResult."""
```

**Blackjack Math (src/riverrater/game/blackjack_math.py):**
```python
def hand_value(cards: list[Card]) -> tuple[int, bool]:
    """Returns (total, is_soft)."""

def running_count(cards_seen: list[Card]) -> int:
    """Hi-Lo running count."""

def true_count(running: int, decks_remaining: float) -> float:
    """Running count / decks remaining."""

def basic_strategy(player_cards: list[Card], dealer_upcard: Card) -> BlackjackAction:
    """Standard basic strategy lookup."""

def kelly_bet(true_count: float, min_bet: float, max_bet: float, bankroll: float) -> float:
    """Kelly Criterion bet sizing."""

def shoe_favorability(tc: float) -> float:
    """Map true count to 0.0-1.0 favorability score."""

def analyze_blackjack(state: BlackjackState, *, min_bet: float = 10.0, max_bet: float = 500.0, bankroll: float = 5000.0) -> BlackjackResult:
    """Full analysis — calls all above, returns populated BlackjackResult.
    Kelly config params threaded through from AppConfig."""
```

### 4. HUD Overlay

**Main Overlay (src/riverrater/hud/overlay.py):**
```python
class HUDOverlay(QMainWindow):
    """Transparent always-on-top overlay window using QStackedWidget for mode switching."""
    
    def __init__(self):
        """Create transparent, frameless, always-on-top window."""
    
    def set_mode(self, mode: GameMode) -> None:
        """Switch between poker and blackjack view."""
    
    def update_poker(self, result: PokerResult) -> None:
        """Update poker HUD with new analysis (delegates to PokerView)."""
    
    def update_blackjack(self, result: BlackjackResult) -> None:
        """Update blackjack HUD with new analysis (delegates to BlackjackView)."""
    
    def set_visible(self, visible: bool) -> None:
        """Toggle HUD visibility."""
    
    def toggle_visibility(self) -> None:
        """Invert current visibility (used by hotkey)."""
    
    def set_position(self, x: int, y: int) -> None:
        """Move overlay to position."""
    
    def set_opacity(self, opacity: float) -> None:
        """Set window opacity (0.0-1.0)."""
```

**Poker View (src/riverrater/hud/poker_view.py):**
```python
class PokerView(QWidget):
    """Poker HUD panel displaying equity, EV, action recommendation, and confidence.
    
    Confidence display (driven by PokerResult.detection_meta):
    - Colored dot in title bar: green (>=0.85), yellow (0.7-0.85), red (<0.7)
    - Card detection row: "Ah 94%  Kd 88%" with per-card color coding
    - Manual mode: shows "MANUAL ✓" when detection_meta.is_manual is True
    - Low-confidence warning: yellow banner when any card < 0.7 confidence
    - Hidden when detection_meta is None
    """
    
    def update(self, result: PokerResult) -> None:
        """Update all display elements from a PokerResult."""
```

**Blackjack View (src/riverrater/hud/blackjack_view.py):**
```python
class BlackjackView(QWidget):
    """Blackjack HUD panel displaying count, strategy, heat meter, and bet sizing."""
    
    def update(self, result: BlackjackResult) -> None:
        """Update all display elements from a BlackjackResult."""
```

**Manual Card Input (src/riverrater/hud/manual_input.py):**
```python
class ManualCardInput(QDialog):
    """Dialog for manually entering cards in blackjack mode.
    
    Signals:
        card_added(str, str): Emitted with (card_str, target) where target is
            'player', 'dealer', or 'seen'.
        shoe_reset(): Emitted when the "New Shoe" button is clicked.
    """
    
    card_added = pyqtSignal(str, str)  # (card_str like "Ah", target like "player")
    shoe_reset = pyqtSignal()           # New shoe button clicked
```

**Poker Input Dialog (src/riverrater/hud/poker_input.py):**
```python
class PokerInputDialog(QDialog):
    """Dialog for entering pot size, bet-to-call, and opponent count in poker mode.
    
    Signals:
        values_submitted(float, float, int): Emitted with (pot_size, bet_to_call, num_opponents)
            when the user clicks Submit.
    """
    
    values_submitted = pyqtSignal(float, float, int)
```

**Calibration Overlay (src/riverrater/hud/calibration_overlay.py):**
```python
class CalibrationOverlay(QWidget):
    """Fullscreen semi-transparent overlay for interactive card template calibration.
    
    Takes a frozen screen capture as background. User draws bounding boxes around
    cards with click-and-drag, assigns rank + suit, and commits to the TemplateEngine.
    
    Features:
    - Rubber-band rectangle drawing (mousePressEvent/Move/Release + paintEvent)
    - Right-side control panel: rank buttons (2-A), suit buttons (♥♦♣♠)
    - ROI preview at 2x zoom
    - Undo last entry
    - Display-to-frame coordinate scaling for accurate bbox mapping
    - Escape key cancels
    
    Signals:
        calibration_finished: Emitted after templates are committed and profile is saved.
        calibration_cancelled: Emitted when user cancels (Escape or Cancel button).
    """
    
    calibration_finished = pyqtSignal()
    calibration_cancelled = pyqtSignal()
    
    def __init__(
        self,
        frame: np.ndarray,               # Frozen screen capture (BGR numpy array)
        template_engine: TemplateEngine,  # Engine to commit templates to
        profile_path: str,                # Where to save the profile
        parent: QWidget | None = None,
    ) -> None: ...
```

### 5. Controller (src/riverrater/main.py)

```python
@dataclass
class AppConfig:
    # Screen capture
    capture_region: Optional[tuple[int, int, int, int]] = None
    capture_fps_target: int = 30
    
    # Game mode
    game_mode: str = "poker"  # GameMode.value
    
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
    hotkey_reset_shoe: str = "<ctrl>+<shift>+n"
    hotkey_poker_input: str = "<ctrl>+<shift>+p"
    
    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        """Load from JSON. Missing keys default gracefully."""
    
    def save(self, path: str | Path) -> None:
        """Persist as JSON."""


class GameController:
    """Central controller managing game state and the processing loop.
    
    Key methods:
    - process_frame(): Called at ~30fps by QTimer. Dispatches to poker or blackjack tick.
    - add_card_manual(card_str, target): Handle card from ManualCardInput.
    - set_poker_values(pot_size, bet_to_call, num_opponents): Handle PokerInputDialog values.
    - reset_hand(): Clear current hand (preserves shoe in blackjack).
    - reset_shoe(): Clear entire shoe state.
    - switch_mode(): Toggle between poker and blackjack.
    
    Frame-skip optimization:
    - _frame_changed() compares downsampled grayscale mean against threshold.
    - Detection runs at most every _detect_every_n ticks and only if frame changed.
    
    Confidence tracking:
    - _detection_meta: Optional[DetectionMeta] updated on each detection or manual input.
    - Attached to PokerResult before sending to HUD.
    """
```

### 6. Utilities

**Hotkey Manager (src/riverrater/utils/hotkeys.py):**
```python
class HotkeyManager:
    """Wraps pynput.keyboard.GlobalHotKeys for cross-platform hotkey registration."""
    
    def register(self, hotkey_str: str, callback: Callable) -> None:
        """Register a hotkey combination (pynput format like '<ctrl>+<shift>+h')."""
    
    def start(self) -> None:
        """Start listening for hotkeys in a background thread."""
    
    def stop(self) -> None:
        """Stop the hotkey listener."""
```

**Logging (src/riverrater/utils/logging.py):**
```python
def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure rotating file + console logging."""

def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
```
