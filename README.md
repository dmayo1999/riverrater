# RiverRater

Real-time casino game assistant with transparent HUD overlay.

## Modules

- **Digital Poker** — OpenCV template matching for 2D poker clients. Calculates equity, pot odds, EV.
- **Live Blackjack** — YOLOv8 object detection for live dealer streams. Tracks card count, basic strategy, bet sizing. (Manual input mode until model is trained.)

## Quick Start

```bash
# Install (no ML dependencies)
pip install -e .

# Run
riverrater

# With ML support (when model is ready)
pip install -e ".[ml]"
```

## Hotkeys

| Key | Action |
|-----|--------|
| Ctrl+Shift+H | Toggle HUD visibility |
| Ctrl+Shift+C | Enter calibration mode (poker) |
| Ctrl+Shift+M | Manual card input (blackjack) |
| Ctrl+Shift+R | Reset current hand |
| Ctrl+Shift+S | Switch poker/blackjack mode |

## Project Structure

```
src/riverrater/
├── main.py          # Entry point
├── config.py        # Settings
├── capture/         # Screen capture (MSS)
├── vision/          # Card detection (OpenCV + YOLO stub)
├── game/            # Game state + math engines
├── hud/             # PyQt6 overlay
└── utils/           # Hotkeys, logging
```
