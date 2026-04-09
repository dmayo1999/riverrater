# RiverRater

Real-time casino game assistant with transparent HUD overlay. Provides live equity calculations for poker and card counting for blackjack, displayed as a non-intrusive overlay on top of your game window.

## Features

### Digital Poker (Module 1)
- **OpenCV template matching** for card detection in 2D poker clients
- **Monte Carlo equity calculator** (5,000 simulations) with win%/tie%
- **Pot odds & EV analysis** — call, fold, raise with recommended action
- **Confidence display** — per-card detection confidence with color-coded indicators
- **Interactive calibration** — draw bounding boxes on screen to teach the vision engine your poker client's card layout
- **Manual poker input** — enter pot size, bet-to-call, and opponent count via dialog (Ctrl+Shift+P)

### Live Blackjack (Module 2)
- **Hi-Lo card counting** with running count and true count
- **Basic strategy engine** — hit/stand/double/split/surrender recommendations
- **Kelly Criterion bet sizing** — optimal bet based on true count, bankroll, and table limits
- **Shoe favorability heat meter** — visual indicator of current edge
- **Manual card input** — enter player hand, dealer upcard, and observed cards
- **Shoe reset** — clear all counts when dealer shuffles a new shoe

### Shared Infrastructure
- **Transparent PyQt6 overlay** — always-on-top, click-through, dark themed
- **Frame-skip optimization** — only runs detection when the screen actually changes
- **YOLO stub** — integration point ready for trained YOLOv8 model (live dealer detection)
- **Global hotkeys** — all features accessible via keyboard shortcuts
- **JSON config** — all settings persisted to `~/.riverrater/config.json`

## Quick Start

```bash
# Install (no ML dependencies)
pip install -e .

# Run in poker mode (default)
riverrater

# Run in blackjack mode
riverrater --mode blackjack

# Debug logging
riverrater --mode poker --debug

# With ML support (when YOLO model is ready)
pip install -e ".[ml]"
```

## Hotkeys

| Shortcut | Action |
|----------|--------|
| Ctrl+Shift+H | Toggle HUD visibility |
| Ctrl+Shift+C | Enter calibration mode (draw bounding boxes for card detection) |
| Ctrl+Shift+M | Manual card input (blackjack) |
| Ctrl+Shift+P | Poker input (pot size, bet-to-call, opponents) |
| Ctrl+Shift+R | Reset current hand |
| Ctrl+Shift+S | Switch between poker and blackjack mode |
| Ctrl+Shift+N | New shoe — reset all card counts (blackjack) |

## Project Structure

```
src/riverrater/
├── main.py                    # Entry point, GameController, AppConfig, hotkey wiring
├── config.py                  # (Legacy — AppConfig lives in main.py)
├── capture/
│   └── screen.py              # MSS screen capture with background thread
├── vision/
│   ├── template_engine.py     # OpenCV multi-scale template matching + NMS
│   ├── calibration.py         # Calibration session system (template capture backend)
│   └── yolo_engine.py         # YOLO stub with CLASS_MAP (awaiting trained model)
├── game/
│   ├── state.py               # Shared types: Card, Rank, Suit, enums, dataclasses, DetectionMeta
│   ├── poker_math.py          # Monte Carlo equity, hand evaluator, pot odds, EV
│   └── blackjack_math.py      # Hi-Lo counting, basic strategy, Kelly, shoe favorability
├── hud/
│   ├── overlay.py             # Transparent PyQt6 overlay (QStackedWidget)
│   ├── poker_view.py          # Poker HUD: win%, equity, EV, action pill, confidence display
│   ├── blackjack_view.py      # Blackjack HUD: count, strategy, heat meter, bet sizing
│   ├── manual_input.py        # Manual card entry dialog (blackjack) with shoe reset
│   ├── poker_input.py         # Poker pot/bet input dialog
│   └── calibration_overlay.py # Fullscreen calibration GUI with rubber-band drawing
└── utils/
    ├── hotkeys.py             # pynput GlobalHotKeys wrapper
    └── logging.py             # Rotating file logger

tests/
├── conftest.py                # Shared fixtures (qapp for PyQt6 tests)
├── test_blackjack_math.py     # Blackjack engine tests
├── test_poker_math.py         # Poker engine tests (equity, EV, pot odds)
├── test_template_engine.py    # Template matching + NMS tests
├── test_calibration.py        # Calibration session tests
├── test_calibration_overlay.py # Calibration GUI widget tests
├── test_detection_meta.py     # DetectionMeta dataclass tests
├── test_poker_input.py        # Poker input dialog tests
└── test_poker_view_confidence.py # Confidence display tests
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run with debug logging
riverrater --debug
```

### Current Test Coverage
- **202 tests**, all passing
- Covers: poker math (equity, EV, pot odds, hand evaluation), blackjack math (counting, strategy, Kelly), template matching (detection, NMS, profiles), calibration, HUD widgets, confidence display

## Configuration

Config is stored at `~/.riverrater/config.json`. A default config is created on first run. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `capture_region` | `null` (full screen) | `[left, top, width, height]` capture area |
| `capture_fps_target` | `30` | Processing loop target FPS |
| `game_mode` | `"poker"` | Starting mode (`"poker"` or `"blackjack"`) |
| `vision_profile` | `"default"` | Template profile name |
| `detection_confidence` | `0.8` | Minimum confidence for card detection |
| `num_decks` | `6` | Blackjack shoe size |
| `min_bet` / `max_bet` | `10.0` / `500.0` | Table bet limits (for Kelly sizing) |
| `bankroll` | `5000.0` | Current bankroll (for Kelly sizing) |
| `hud_position` | `[100, 100]` | HUD overlay position `[x, y]` |
| `hud_opacity` | `0.85` | HUD overlay opacity |

## Architecture

```
                    ┌─────────────────────┐
                    │   GameController     │
                    │   (main.py)          │
                    └──────┬──────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
     ScreenCapture   TemplateEngine    HUDOverlay
     (MSS thread)    (OpenCV/YOLO)     (PyQt6)
              │            │                │
              │            ▼                ├── PokerView (+ confidence)
              └──► frame ──► detections     ├── BlackjackView
                            │               ├── ManualCardInput
                            ▼               ├── PokerInputDialog
                    PokerState/             └── CalibrationOverlay
                    BlackjackState
                            │
                            ▼
                    poker_math.py /
                    blackjack_math.py
                            │
                            ▼
                    PokerResult /
                    BlackjackResult
                            │
                            ▼
                      HUD Display
```

## Roadmap

### Completed
- [x] Screen capture with MSS (background thread, persistent context)
- [x] OpenCV template matching with multi-scale + NMS
- [x] Monte Carlo poker equity calculator
- [x] Pot odds, EV, action recommendation
- [x] Hi-Lo card counting with true count
- [x] Basic strategy engine (full decision table)
- [x] Kelly Criterion bet sizing
- [x] Transparent PyQt6 HUD overlay
- [x] Global hotkey system
- [x] Manual card input (blackjack)
- [x] Poker pot/bet input dialog
- [x] Calibration GUI (bounding-box drawing)
- [x] Confidence display (per-card + overall indicator)
- [x] Frame-skip optimization (pixel change detection)
- [x] Shoe reset (hotkey + button)
- [x] YOLO engine stub with class mapping

### Next Up
- [ ] Train YOLOv8 model for live dealer card detection (Google Colab)
- [ ] Integrate trained model into yolo_engine.py
- [ ] Opponent count input UI
- [ ] Session stats / hand history logging
- [ ] Config UI (replace JSON-only editing)
- [ ] Calibration bounding-box drag-to-reposition

### Hardware
- **Development**: M4 MacBook Air
- **Training**: Google Colab (GPU)
