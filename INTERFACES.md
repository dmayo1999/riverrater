# RiverRater Module Interface Contracts

This document defines the exact interfaces between modules so they can be built independently.

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
class PokerResult:
    win_pct: float = 0.0
    tie_pct: float = 0.0
    required_equity: float = 0.0     # pot_odds as equity threshold
    actual_equity: float = 0.0       # win_pct + tie_pct/2
    ev_call: float = 0.0
    ev_fold: float = 0.0
    ev_raise: float = 0.0
    recommended_action: Optional[PokerAction] = None

@dataclass
class BlackjackResult:
    running_count: int = 0
    true_count: float = 0.0
    recommended_action: Optional[BlackjackAction] = None
    recommended_bet: float = 0.0     # Kelly-based bet size
    shoe_favorability: float = 0.0   # 0.0 to 1.0 for heat meter
    hand_total: int = 0
    is_soft: bool = False            # Contains usable ace
```

## Module Interfaces

### 1. Screen Capture (src/riverrater/capture/screen.py)

```python
class ScreenCapture:
    def __init__(self, region: tuple[int,int,int,int] | None = None):
        """region = (left, top, width, height) or None for full screen"""
    
    def set_region(self, region: tuple[int,int,int,int]) -> None:
        """Update capture region"""
    
    def grab_frame(self) -> np.ndarray:
        """Capture single frame, return as BGR numpy array"""
    
    def get_fps(self) -> float:
        """Return current capture FPS"""
    
    def start(self) -> None:
        """Start continuous capture in background thread"""
    
    def stop(self) -> None:
        """Stop capture"""
    
    def get_latest_frame(self) -> np.ndarray | None:
        """Get most recent frame from buffer (non-blocking)"""
```

### 2. Vision Engines

**Template Engine (src/riverrater/vision/template_engine.py):**
```python
class TemplateEngine:
    def __init__(self, profile_path: str | None = None):
        """Load vision profile from JSON, or start empty"""
    
    def detect_cards(self, frame: np.ndarray, confidence: float = 0.8) -> list[tuple[Card, tuple[int,int,int,int], float]]:
        """Returns list of (Card, bounding_box, confidence_score)"""
    
    def add_template(self, card: Card, template_image: np.ndarray) -> None:
        """Add/update a card template"""
    
    def save_profile(self, path: str) -> None:
        """Save current templates + config to JSON profile"""
    
    def load_profile(self, path: str) -> None:
        """Load templates + config from JSON profile"""
```

**YOLO Engine Stub (src/riverrater/vision/yolo_engine.py):**
```python
class YOLOEngine:
    """Stub — same interface as TemplateEngine.detect_cards.
    Will be implemented when trained model is available."""
    
    def __init__(self, model_path: str | None = None):
        """Load YOLO model weights, or None if not yet available"""
        self._model = None  # Will be ultralytics.YOLO when ready
    
    def detect_cards(self, frame: np.ndarray, confidence: float = 0.5) -> list[tuple[Card, tuple[int,int,int,int], float]]:
        """Returns same format as TemplateEngine — empty list for now"""
        raise NotImplementedError("YOLO model not yet trained. Use manual input mode.")
    
    @property
    def is_available(self) -> bool:
        return self._model is not None
```

### 3. Math Engines

**Poker Math (src/riverrater/game/poker_math.py):**
```python
def calculate_equity(hole_cards: list[Card], community_cards: list[Card], num_opponents: int = 1, simulations: int = 5000) -> tuple[float, float]:
    """Monte Carlo simulation. Returns (win_pct, tie_pct)"""

def calculate_pot_odds(pot_size: float, bet_to_call: float) -> float:
    """Returns required equity as decimal (0.0-1.0)"""

def calculate_ev(win_pct: float, pot_size: float, bet_to_call: float) -> tuple[float, float, float]:
    """Returns (ev_call, ev_fold, ev_raise_estimate)"""

def analyze_poker(state: PokerState) -> PokerResult:
    """Full analysis — calls all above, returns populated PokerResult"""
```

**Blackjack Math (src/riverrater/game/blackjack_math.py):**
```python
def hand_value(cards: list[Card]) -> tuple[int, bool]:
    """Returns (total, is_soft)"""

def running_count(cards_seen: list[Card]) -> int:
    """Hi-Lo running count"""

def true_count(running: int, decks_remaining: float) -> float:
    """Running count / decks remaining"""

def basic_strategy(player_cards: list[Card], dealer_upcard: Card) -> BlackjackAction:
    """Standard basic strategy lookup"""

def kelly_bet(true_count: float, min_bet: float, max_bet: float, bankroll: float) -> float:
    """Kelly Criterion bet sizing"""

def analyze_blackjack(state: BlackjackState) -> BlackjackResult:
    """Full analysis — calls all above, returns populated BlackjackResult"""
```

### 4. HUD Overlay

**Main Overlay (src/riverrater/hud/overlay.py):**
```python
class HUDOverlay(QMainWindow):
    """Transparent always-on-top overlay window"""
    
    def __init__(self):
        """Create transparent, frameless, always-on-top window"""
    
    def set_mode(self, mode: GameMode) -> None:
        """Switch between poker and blackjack view"""
    
    def update_poker(self, result: PokerResult) -> None:
        """Update poker HUD with new analysis"""
    
    def update_blackjack(self, result: BlackjackResult) -> None:
        """Update blackjack HUD with new analysis"""
    
    def set_visible(self, visible: bool) -> None:
        """Toggle HUD visibility"""
    
    def set_position(self, x: int, y: int) -> None:
        """Move overlay to position"""
```

### 5. Config (src/riverrater/config.py)

```python
@dataclass
class AppConfig:
    # Screen capture
    capture_region: tuple[int,int,int,int] | None = None
    capture_fps_target: int = 30
    
    # Game mode
    game_mode: GameMode = GameMode.POKER
    
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
    def load(cls, path: str) -> "AppConfig": ...
    
    def save(self, path: str) -> None: ...
```
