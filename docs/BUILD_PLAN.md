# RiverRater — Build Plan & Technical Analysis

> **Last updated:** April 9, 2026 — after fixes 1-8, confidence display, and calibration GUI.

## Progress Summary

### Completed (Phases 0-2 + portions of Phase 4 and 5)

| Component | Status | Notes |
|-----------|--------|-------|
| Project scaffold (pyproject.toml, README, INTERFACES.md) | ✅ Done | |
| MSS screen capture with background thread | ✅ Done | Persistent mss context, frame-skip optimization |
| PyQt6 transparent overlay (QStackedWidget) | ✅ Done | Dark theme, drag-to-move, opacity config |
| Global hotkey system (pynput) | ✅ Done | 7 hotkeys registered |
| OpenCV multi-scale template matching + NMS | ✅ Done | |
| Calibration GUI (rubber-band bounding box) | ✅ Done | Fullscreen overlay, rank/suit panel, undo |
| Vision profile save/load | ✅ Done | JSON profiles in ~/.riverrater/profiles/ |
| Monte Carlo equity calculator | ✅ Done | 5,000 simulations, win%/tie% |
| Pot odds calculator | ✅ Done | |
| EV calculator (call/fold/raise) | ✅ Done | Raise penalizes losing hands (2x negative EV) |
| Poker HUD view | ✅ Done | Win%, equity, EV, action pill, confidence display |
| Confidence display (P0+P1) | ✅ Done | Per-card badges, overall dot, warning banner, manual mode |
| Manual card input (blackjack) | ✅ Done | Rank/suit buttons, shoe_reset signal |
| Poker input dialog (pot/bet/opponents) | ✅ Done | Ctrl+Shift+P hotkey |
| Hi-Lo running count + true count | ✅ Done | |
| Basic strategy engine | ✅ Done | Full decision table |
| Kelly Criterion bet sizing | ✅ Done | Config-driven min/max/bankroll |
| Shoe favorability heat meter | ✅ Done | |
| Blackjack HUD view | ✅ Done | Count, strategy, heat meter, bet sizing |
| Shoe reset (hotkey + button) | ✅ Done | Ctrl+Shift+N |
| YOLO engine stub | ✅ Done | CLASS_MAP, integration docs, graceful fallback |
| JSON config system | ✅ Done | ~/.riverrater/config.json, auto-create on first run |
| Frame-skip optimization | ✅ Done | Pixel change detection, detect every 3rd tick |
| Dead code cleanup | ✅ Done | Stripped all stale try/except fallbacks |
| Deduplication guards | ✅ Done | cards_seen checked before append |
| Test suite | ✅ Done | **202 tests**, all passing |
| GitHub repo | ✅ Done | https://github.com/dmayo1999/riverrater (private) |

### Remaining

| Component | Phase | Blocked By | Est. Hours |
|-----------|-------|------------|------------|
| YOLO model training (YOLOv8n for live dealer cards) | Phase 3 | Training data collection + Google Colab | ~54 |
| Integrate trained YOLO model into yolo_engine.py | Phase 4 | Trained model | ~8 |
| YOLO card deduplication (track "new" vs "already counted") | Phase 4 | YOLO integration | ~6 |
| Pot size OCR (read chip counts from poker client) | Phase 2 | Nice-to-have, manual input works | ~10 |
| Opponent count input UI | Phase 5 | Low priority | ~4 |
| Session stats / hand history logging | Phase 5 | Low priority | ~8 |
| Config UI (replace JSON-only editing) | Phase 5 | Low priority | ~8 |
| Multi-table support | Won't have v1 | — | — |

### Code Statistics

- **Source**: ~5,700 LOC across 17 Python files
- **Tests**: ~1,800 LOC across 11 test files (202 tests)
- **Commits**: 4 commits on master
- **Dependencies**: opencv-python, mss, PyQt6, numpy, pynput (+ torch/ultralytics as optional)

---

## PRD Assessment

### What the PRD Gets Right

The core concept is well-scoped into two distinct modules with different vision pipelines — that's the correct architectural instinct. The acknowledgment of the "Vision Complexity Gap" between Module 1 (template matching) and Module 2 (YOLO object detection) is accurate and honest. The feature set for each module (pot odds/equity for poker, Hi-Lo counting/Kelly Criterion for blackjack) maps directly to established, well-documented gambling math.

### Critical Gaps in the PRD

| Gap | Impact | Recommendation |
|-----|--------|---------------|
| No user stories or acceptance criteria | Can't verify "done" for any feature | Add Given/When/Then criteria per feature |
| No success metrics | No way to measure if the tool works | Define detection accuracy targets (e.g., >95% card ID), latency budget (<500ms end-to-end) |
| No platform-specific architecture | macOS and Windows have completely different screen capture, overlay, and GPU acceleration APIs | Decide primary platform first, port second |
| No data pipeline for YOLO training | The PRD says "trained model" but doesn't address how you get thousands of labeled casino-lighting card images | This is a Phase 0 blocker for Module 2 |
| No error handling / degradation strategy | What happens when vision fails mid-hand? | Define fallback UX (manual input mode, confidence thresholds) |
| TOS section is a warning, not a design constraint | Needs to be an architectural driver, not a footnote | Design detection-evasion into the architecture from day one |

---

## Technical Feasibility on Your Hardware

**Target machine: M4 MacBook Air (16GB unified memory)**

### Screen Capture Pipeline

| Library | macOS FPS (full screen) | macOS FPS (region) | Notes |
|---------|------------------------|-------------------|-------|
| MSS (CoreGraphics) | ~47-50 | ~57-61 | Best option for Mac — uses native CoreGraphics API |
| Pillow/ImageGrab | ~2.8 | ~2.6 | Unusably slow on Mac — shells out to `screencapture` CLI |
| PyAutoGUI | ~2-3 | ~2-3 | Same bottleneck as Pillow |

**Verdict:** MSS is the only viable option on macOS. Capturing a small ROI (region of interest) around the card areas rather than full screen pushes you well above the 30 FPS needed for real-time processing.

Source: [Kyle Fu benchmarks](https://kylefu.me/2023/02/18/python-fast-screen-capture.html), [Poppy Ramblings benchmarks](https://blog.trackmypop.com/2024/01/02/quick-screenshots-in-python/)

### OpenCV Template Matching (Module 1 — Digital Poker)

Template matching with `cv2.matchTemplate` is lightweight and will run at negligible latency on M4. However, there are real limitations:

- **Works well for:** Fixed-theme poker clients where card assets don't change between sessions.
- **Breaks on:** Resized windows, different themes/skins, anti-aliased or animated cards.
- **Multi-scale matching is mandatory.** Resize the source image across multiple scales and run matching at each — [PyImageSearch documents this pattern well](https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/).
- The "Dynamic Vision Calibration" feature in the PRD (user draws bounding box, labels it) is actually the right call — it sidesteps the fragility of pure template matching.

**Verdict:** Feasible as-designed. The user-driven calibration system is the key differentiator that makes this work across arbitrary poker clients.

### YOLOv8 Inference (Module 2 — Live Blackjack)

| Metric | M2 Pro (MPS) | M4 Air (expected) |
|--------|-------------|-------------------|
| YOLOv8n inference | ~33ms/frame | ~25-30ms/frame |
| YOLOv8s inference | ~50ms/frame | ~40-45ms/frame |
| YOLOv8x inference | ~100ms+ | ~80-90ms |

**Key findings from PyTorch MPS research:**

- Inference on MPS is solid — [33ms per frame on M2 Pro for yolov8x](https://github.com/ultralytics/ultralytics/issues/5717), which is well within the <1 second budget.
- Training on MPS is problematic — loss calculations are 40x slower than CPU due to a known PyTorch bug. **Train on Google Colab (free T4 GPU), deploy to local Mac for inference.**
- YOLOv8n (nano) is the right model size for this use case — you're detecting 52 classes (cards) in a constrained visual field, not general object detection.

**Verdict:** Feasible for inference. Train on Colab, run locally. The M4's Neural Engine could further accelerate via CoreML export, but that's an optimization, not a requirement.

Source: [Ultralytics GitHub #5717](https://github.com/ultralytics/ultralytics/issues/5717), [Dev Genius MPS guide](https://blog.devgenius.io/running-yolov8-on-apple-silicon-with-mps-backend-a-simplified-guide-84b1d382f79c)

### PyQt Overlay HUD

Transparent, always-on-top overlays in PyQt5/6 are well-documented:

```python
self.setWindowFlags(
    Qt.WindowStaysOnTopHint |
    Qt.FramelessWindowHint |
    Qt.X11BypassWindowManagerHint  # Linux only
)
self.setAttribute(Qt.WA_TranslucentBackground)
```

On macOS, this works natively. On Windows, you may need `win32gui.SetWindowPos` with `HWND_TOPMOST` for full-screen game contexts.

**Verdict:** Fully feasible. PyQt6 is preferred over Tkinter for the overlay — Tkinter's transparency support is inconsistent across platforms.

Source: [Stack Overflow PyQt overlay](https://stackoverflow.com/questions/25950049/creating-a-transparent-overlay-with-qt), [Microsoft Learn overlay guide](https://learn.microsoft.com/en-us/answers/questions/3918271/keeping-python-application-always-on-top-(even-dur)

### Training Data for YOLO (Module 2 Blocker)

The [Roboflow "Playing Cards" dataset](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d/dataset/4) provides a pre-labeled YOLOv8-format dataset of playing cards. However:

- These are **studio-lit, flat card images** — not live dealer casino feeds.
- Live dealer conditions introduce: glare, motion blur, partial occlusion by the dealer's hand, varying camera angles, and low-contrast felt backgrounds.
- The [Stanford Pokémon card detection paper](https://cs231n.stanford.edu/2024/papers/real-time-pokemon-card-detection-from-tournament-footage.pdf) found that even with aggressive augmentation, mAP stayed below 0.5 for real tournament footage — illustrating the domain gap problem.

**Verdict:** You can bootstrap with the Roboflow dataset + synthetic augmentation (rotation, brightness jitter, motion blur, perspective warp), but you will eventually need to collect and label frames from actual live dealer streams to reach production accuracy. Budget 2-4 weeks for this data pipeline.

Source: [Eran Feit YOLOv8 card training tutorial](https://eranfeit.net/how-to-train-yolov8-object-detection-on-a-custom-dataset-cards-detection/), [Stanford card detection paper](https://cs231n.stanford.edu/2024/papers/real-time-pokemon-card-detection-from-tournament-footage.pdf)

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    RiverRater Core                    │
├──────────────┬───────────────────────────────────────┤
│  Screen IO   │  MSS (region capture) + PyQt6 (HUD)  │
├──────────────┼───────────────────────────────────────┤
│  Vision      │  Module 1: OpenCV Template Matching   │
│  Engines     │  Module 2: YOLOv8n (PyTorch MPS)     │
├──────────────┼───────────────────────────────────────┤
│  Game State  │  Shared state manager (dataclass)     │
│  Manager     │  Tracks cards seen, pot, counts       │
├──────────────┼───────────────────────────────────────┤
│  Math        │  Poker: Equity engine (Monte Carlo)   │
│  Engines     │  Blackjack: Hi-Lo + Kelly Criterion   │
├──────────────┼───────────────────────────────────────┤
│  Config      │  JSON profiles per poker client       │
│              │  Calibration data store                │
└──────────────┴───────────────────────────────────────┘
```

### Key Design Decisions

1. **Primary platform: macOS first.** You're developing on an M4 MacBook Air. Porting to Windows later is straightforward (swap MSS monitor config, adjust overlay flags). Going the other direction is harder.

2. **PyQt6 over Tkinter.** Tkinter can't reliably do transparent overlays cross-platform. PyQt6 handles it natively and gives you a proper widget system for the HUD.

3. **MSS for screen capture, not window hooking.** MSS captures a screen region — it doesn't inject into the target process. This is both simpler and less detectable than memory reading or API hooking.

4. **Separate vision engines, shared game state.** Module 1 and Module 2 feed into the same `GameState` dataclass. The math engines read from GameState. This decouples vision from logic cleanly.

5. **Train on Colab, infer locally.** PyTorch MPS is fine for inference (~30ms) but broken for training. Use your Colab setup for YOLO training, export `best.pt`, load locally.

---

## MoSCoW Prioritization

### P0 — Must Have (MVP)

| # | Feature | Module | Rationale |
|---|---------|--------|-----------|
| 1 | Screen region capture loop | Core | Everything depends on this |
| 2 | OpenCV template matching for digital cards | Poker | Core vision for Module 1 |
| 3 | User-driven calibration (draw box, label card) | Poker | Makes template matching work across clients |
| 4 | Hand equity calculator (Monte Carlo sim) | Poker | Core value proposition |
| 5 | Pot odds vs. required equity display | Poker | Core value proposition |
| 6 | Transparent HUD overlay (PyQt6) | Core | Delivery mechanism for all information |
| 7 | JSON-based vision profiles (save/load calibrations) | Poker | Persistence across sessions |

### P1 — Should Have (Fast Follow)

| # | Feature | Module | Rationale |
|---|---------|--------|-----------|
| 8 | EV calculation for call/fold/raise | Poker | Enhances poker module depth |
| 9 | Pot size OCR (read chip counts from screen) | Poker | Currently requires manual input without this |
| 10 | YOLOv8n card detection (physical cards) | Blackjack | Core vision for Module 2 |
| 11 | Basic Strategy lookup table | Blackjack | Simple dict lookup, low effort |
| 12 | Running Count + True Count tracker | Blackjack | Core blackjack value prop |
| 13 | Hotkey system (global keyboard shortcuts) | Core | UX necessity for calibration and toggling |

### P2 — Could Have (Nice to Have)

| # | Feature | Module | Rationale |
|---|---------|--------|-----------|
| 14 | "Heat meter" shoe favorability gauge | Blackjack | Visual polish |
| 15 | Kelly Criterion bet sizing | Blackjack | Advanced feature, requires accurate count |
| 16 | Multi-table support (track 2+ poker tables) | Poker | Power user feature |
| 17 | Session statistics / hand history logging | Core | Useful but not core |
| 18 | CoreML export for faster YOLO inference | Blackjack | Optimization — MPS is already fast enough |

### Won't Have (v1)

| Feature | Reason |
|---------|--------|
| Mobile/tablet version | Completely different capture/overlay paradigm |
| Automated betting / mouse control | Crosses from "assistant" to "bot" — much higher legal risk |
| Multi-player tracking (opponent hand estimation) | Requires reading multiple unknown hands — out of scope |
| Real-money API integration | Liability and legal exposure |

---

## Phased Build Plan

### Phase 0: Foundation (Week 1-2)

**Goal:** Capture screen, display overlay, prove the pipeline works end-to-end.

| Task | Est. Hours | Dependencies |
|------|-----------|-------------|
| Project scaffold: `pyproject.toml`, venv, directory structure | 2 | None |
| MSS screen capture loop (configurable region) | 4 | None |
| PyQt6 transparent overlay window (stays on top, shows text) | 6 | None |
| Integration: capture frame → display frame count on overlay | 3 | Both above |
| Global hotkey system (`pynput` or `keyboard` library) | 4 | None |
| **Total** | **~19 hours** | |

**Exit criteria:** A transparent overlay window displays the current FPS of your screen capture loop, stays on top of a browser window, and responds to a global hotkey to toggle visibility.

---

### Phase 1: Digital Poker Vision (Week 3-5)

**Goal:** Detect cards on a digital poker client and display hand information.

| Task | Est. Hours | Dependencies |
|------|-----------|-------------|
| Template matching engine: load templates, multi-scale `cv2.matchTemplate` | 10 | Phase 0 |
| Calibration UI: hotkey → user draws bounding box → labels card → saves template | 12 | Phase 0 hotkeys |
| Vision profile system: save/load JSON per poker client | 4 | Template engine |
| Card detection pipeline: frame → preprocess → match → return detected cards | 8 | Template engine |
| Game state manager: track community cards, hole cards, dealer button position | 6 | Card detection |
| **Total** | **~40 hours** | |

**Exit criteria:** Open any digital poker client. Calibrate 10+ card templates. The overlay correctly identifies your hole cards and community cards in real-time with >90% accuracy on the calibrated client.

---

### Phase 2: Poker Math Engine (Week 5-7)

**Goal:** Calculate and display equity, pot odds, and EV.

| Task | Est. Hours | Dependencies |
|------|-----------|-------------|
| Monte Carlo equity calculator (simulate 1000+ runouts) | 10 | Phase 1 game state |
| Pot odds calculator (read pot size — manual input initially) | 4 | Phase 1 game state |
| Required equity vs. actual equity comparison | 3 | Equity calc |
| EV calculator for call/fold/raise decisions | 6 | Equity + pot odds |
| HUD layout: design poker view (Win%, Equity, EV display) | 8 | Phase 0 overlay |
| Pot size OCR (detect chip/number UI elements) | 10 | Phase 1 vision |
| **Total** | **~41 hours** | |

**Exit criteria:** During a live poker session, the overlay displays your win probability, whether a call is +EV or -EV, and the required equity to continue — all updating within 500ms of new cards appearing.

---

### Phase 3: YOLO Training Pipeline (Week 7-10)

**Goal:** Train a YOLOv8n model that can detect physical playing cards under live dealer conditions.

| Task | Est. Hours | Dependencies |
|------|-----------|-------------|
| Download Roboflow playing cards dataset (YOLOv8 format) | 2 | None |
| Data augmentation pipeline: rotation, brightness, blur, perspective warp, casino-felt backgrounds | 12 | Dataset |
| Record and label 500+ frames from live dealer streams | 16 | Access to live dealer platform |
| Train YOLOv8n on Colab (T4 GPU): 50-100 epochs, evaluate mAP | 8 | Augmented dataset |
| Export `best.pt`, test inference locally on MPS | 4 | Trained model |
| Iterate: identify failure cases, add targeted training data, retrain | 12 | First model |
| **Total** | **~54 hours** | |

**Exit criteria:** The YOLO model achieves >85% mAP@0.5 on a held-out test set of live dealer frames. Inference runs at <50ms per frame on the M4 MacBook Air.

---

### Phase 4: Live Blackjack Module (Week 10-13)

**Goal:** Track cards, maintain count, display strategy in real-time.

| Task | Est. Hours | Dependencies |
|------|-----------|-------------|
| YOLO inference pipeline: frame → detect → extract card classes | 8 | Phase 3 model |
| Card deduplication: track which cards are "new" vs. already counted | 6 | Inference pipeline |
| Hi-Lo running count engine | 4 | Card tracking |
| True Count calculator (running count / estimated decks remaining) | 4 | Running count |
| Basic Strategy lookup table (hard totals, soft totals, pairs) | 6 | Card detection |
| HUD layout: blackjack view (Count, Strategy Move, Heat Meter) | 8 | Phase 0 overlay |
| Kelly Criterion bet sizing calculator | 6 | True Count |
| **Total** | **~42 hours** | |

**Exit criteria:** While watching a live dealer blackjack stream, the overlay displays the current running count, true count, recommended Basic Strategy move, and suggested bet size — all updating correctly as new cards are dealt.

---

### Phase 5: Polish & Hardening (Week 13-15)

| Task | Est. Hours | Dependencies |
|------|-----------|-------------|
| Confidence thresholds: only display when vision confidence > X% | 4 | All vision |
| Manual override mode (type cards when vision fails) | 6 | Game state |
| Settings panel (capture region, overlay position, hotkeys) | 8 | Core |
| Error handling, logging, crash recovery | 6 | All |
| Performance profiling and optimization | 6 | All |
| **Total** | **~30 hours** | |

---

## Total Estimated Effort

| Phase | Hours | Weeks |
|-------|-------|-------|
| Phase 0: Foundation | 19 | 1-2 |
| Phase 1: Poker Vision | 40 | 3-5 |
| Phase 2: Poker Math | 41 | 5-7 |
| Phase 3: YOLO Training | 54 | 7-10 |
| Phase 4: Blackjack Module | 42 | 10-13 |
| Phase 5: Polish | 30 | 13-15 |
| **Total** | **~226 hours** | **~15 weeks** |

At ~15 hours/week (student pace), this is a roughly 15-week project. The first shippable product (Poker module only) lands at Week 7 (~100 hours).

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Live dealer card detection accuracy too low | High | Blocks Module 2 | Start data collection early (Phase 3). Accept manual-input fallback as permanent backup. |
| macOS screen capture blocked by poker client anti-cheat | Medium | Blocks Module 1 | MSS uses CoreGraphics (OS-level), not process injection. Most browser-based platforms won't detect it. Native poker clients (PokerStars, GGPoker) may. |
| PyTorch MPS bugs cause inference failures | Medium | Degrades Module 2 | Fall back to CPU inference (still fast enough for YOLOv8n at ~100ms). CoreML export as backup. |
| Template matching fragility across poker skins | High | Degrades Module 1 | The calibration system is the mitigation. Also consider training a small CNN classifier as a Phase 2 upgrade. |
| Overlay detected by casino platform | Medium | Legal/account risk | Don't inject into process. Don't automate betting. Keep overlay as a separate window. Use at your own risk — the PRD already acknowledges this. |

---

## Recommended Project Structure

```
riverrater/
├── pyproject.toml
├── README.md
├── src/
│   ├── riverrater/
│   │   ├── __init__.py
│   │   ├── main.py                  # Entry point, mode selection
│   │   ├── config.py                # Settings, paths, hotkeys
│   │   ├── capture/
│   │   │   ├── __init__.py
│   │   │   └── screen.py            # MSS capture loop
│   │   ├── vision/
│   │   │   ├── __init__.py
│   │   │   ├── template_engine.py   # OpenCV template matching
│   │   │   ├── calibration.py       # User-driven template labeling
│   │   │   ├── yolo_engine.py       # YOLOv8 inference wrapper
│   │   │   └── profiles/            # JSON vision profiles per client
│   │   ├── game/
│   │   │   ├── __init__.py
│   │   │   ├── state.py             # GameState dataclass
│   │   │   ├── poker_math.py        # Equity, pot odds, EV
│   │   │   └── blackjack_math.py    # Hi-Lo, True Count, Kelly
│   │   ├── hud/
│   │   │   ├── __init__.py
│   │   │   ├── overlay.py           # PyQt6 transparent window
│   │   │   ├── poker_view.py        # Poker HUD layout
│   │   │   └── blackjack_view.py    # Blackjack HUD layout
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── hotkeys.py           # Global hotkey management
│   │       └── logging.py           # Structured logging
├── models/
│   └── yolov8n_cards/
│       └── best.pt                  # Trained YOLO weights
├── data/
│   ├── templates/                   # Saved card templates per profile
│   └── training/                    # YOLO training data
└── tests/
    ├── test_template_engine.py
    ├── test_poker_math.py
    └── test_blackjack_math.py
```

---

## Key Dependencies

```toml
[project]
name = "riverrater"
requires-python = ">=3.11"
dependencies = [
    "opencv-python>=4.9",
    "mss>=9.0",
    "PyQt6>=6.6",
    "numpy>=1.26",
    "torch>=2.2",
    "ultralytics>=8.1",
    "pynput>=1.7",       # Global hotkeys
]

[project.optional-dependencies]
train = [
    "roboflow",          # Dataset download
    "albumentations",    # Advanced augmentation
]
```

---

## What to Build First (This Week)

If you want to start coding today, the highest-leverage first step is the **Phase 0 integration proof**:

1. `pip install mss PyQt6 opencv-python pynput`
2. Write `screen.py` — capture a 400x300 region of your screen at 30+ FPS using MSS.
3. Write `overlay.py` — display a transparent PyQt6 window that shows "FPS: XX" text, stays on top of everything.
4. Wire them together: capture loop feeds frame count to overlay.
5. Add a hotkey (e.g., `Ctrl+Shift+R`) to toggle the overlay on/off.

That's ~6 hours of work and proves the entire I/O pipeline before you touch any vision or math logic.
