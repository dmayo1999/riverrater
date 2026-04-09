# RiverRater: Calibration GUI + Confidence Display — Feature Plan

## Feature 1: Calibration Bounding-Box GUI

### Problem Statement
The calibration system (`CalibrationCapture` + `CalibrationSession`) exists but is headless — there's no way for users to draw bounding boxes on screen to define where cards appear. The `_calibrate()` hotkey handler in `main.py` is a no-op placeholder. Without this, users can't create vision profiles for template matching.

### Goals
- Let users visually draw rectangles over card locations on a frozen screen capture
- Capture ROIs and associate them with card labels (rank + suit)
- Save the resulting templates into a `TemplateEngine` profile for reuse
- Make the entire flow accessible via the existing `Ctrl+Shift+C` hotkey

### Non-Goals
- Auto-detection of card regions (requires trained model)
- Multi-monitor support (initial version: single capture region only)
- Video/animated calibration walkthrough

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  CalibrationOverlay (QWidget, fullscreen, semi-transparent) │
│                                                             │
│  ┌─────────────────────────────────────┐   ┌─────────────┐ │
│  │                                     │   │ Card Label  │ │
│  │   Frozen screen capture             │   │  [Rank ▼]   │ │
│  │   User draws bbox with mouse        │   │  [Suit ▼]   │ │
│  │   ┌──────┐                          │   │             │ │
│  │   │ bbox │←drawn rect               │   │ [Confirm]   │ │
│  │   └──────┘                          │   │ [Skip]      │ │
│  │                                     │   │ [Undo]      │ │
│  └─────────────────────────────────────┘   │ [Finish]    │ │
│                                             └─────────────┘ │
│  Status: "Draw a box around the Ace of Hearts"              │
│  Progress: 3/52 cards captured                              │
└─────────────────────────────────────────────────────────────┘
```

### User Stories

1. **As a poker player**, I want to draw boxes around visible cards on my poker client so the template engine can detect them in future hands.
2. **As a first-time user**, I want clear instructions during calibration so I know exactly what to click and where.
3. **As a returning user**, I want to load an existing profile and add/replace individual card templates without recalibrating everything.

### Requirements

#### P0 — Must Have

| # | Requirement | Acceptance Criteria |
|---|-------------|---------------------|
| 1 | **CalibrationOverlay widget** — Fullscreen semi-transparent QWidget overlaid on the frozen screen capture. | Given the user presses Ctrl+Shift+C, When the overlay appears, Then it freezes the current screen frame as the background and accepts mouse input for drawing. |
| 2 | **Rubber-band rectangle drawing** — Click-and-drag to draw a bounding box. Shows live preview rectangle as user drags. | Given the overlay is active, When the user clicks and drags, Then a green-bordered rectangle follows the mouse. When released, the rectangle is stored as a candidate bbox. |
| 3 | **Card label assignment** — Side panel with rank dropdown (2-A) and suit selector (♥♦♣♠). User picks what card the bbox contains. | Given a bbox is drawn, When the user selects rank and suit and clicks Confirm, Then the (card, bbox, frame) triple is added to the `CalibrationSession`. |
| 4 | **Session commit** — "Finish" button calls `CalibrationSession.finish(template_engine)` and saves the profile. | Given at least 1 calibration entry exists, When the user clicks Finish, Then templates are committed to the TemplateEngine, profile is saved to `~/.riverrater/profiles/{name}/`, and the overlay closes. |
| 5 | **Cancel / Escape** — Pressing Escape or clicking Cancel discards all pending calibrations. | Given the overlay is active, When the user presses Escape, Then `CalibrationSession.cancel()` is called and the overlay closes with no side effects. |
| 6 | **Wire to GameController** — `_calibrate()` in main.py opens the CalibrationOverlay with the current capture frame and template engine. | Given the app is running, When Ctrl+Shift+C is pressed, Then capture is paused, a frame is grabbed, and CalibrationOverlay opens. |

#### P1 — Should Have

| # | Requirement | Acceptance Criteria |
|---|-------------|---------------------|
| 7 | **Undo last** — Button to remove the most recently added calibration entry. | Given 3 entries have been added, When Undo is clicked, Then the 3rd entry is removed and the counter shows 2. |
| 8 | **Visual feedback per capture** — Brief green flash or checkmark on the bbox area when a calibration is confirmed. | Given a card is confirmed, Then a 500ms green overlay flashes on the bbox region. |
| 9 | **Progress counter** — Display "N cards captured" and optionally a grid showing which of the 52 cards have templates. | Given 5 calibrations are confirmed, Then the status bar shows "5 cards captured". |
| 10 | **Zoom preview** — Small inset window showing the cropped ROI at 2x magnification so the user can verify the crop is clean. | Given a bbox is drawn, Then a 2x zoom of the cropped ROI appears in the side panel. |

#### P2 — Could Have

| # | Requirement | Acceptance Criteria |
|---|-------------|---------------------|
| 11 | **Guided mode** — System suggests which card to capture next based on what's missing from the profile. | Given 20 cards have templates, Then the prompt says "Try to capture: 3♣ (no template yet)". |
| 12 | **Multi-bbox batch** — Draw multiple boxes at once, then label them left-to-right. | Given 3 boxes are drawn, Then labeling prompts appear for each in sequence. |

### New Files

| File | Purpose |
|------|---------|
| `src/riverrater/hud/calibration_overlay.py` | `CalibrationOverlay(QWidget)` — fullscreen rubber-band drawing + side panel |
| `tests/test_calibration_overlay.py` | Unit tests for the overlay (widget creation, bbox storage, signal emission) |

### Modified Files

| File | Change |
|------|--------|
| `src/riverrater/main.py` | Replace `_calibrate()` no-op with CalibrationOverlay instantiation. Pause capture during calibration, resume on close. Pass template_engine + current frame. |
| `src/riverrater/vision/calibration.py` | No changes needed — `CalibrationSession` already supports the required workflow. |

### Implementation Notes

- **QPainter for rubber-band**: Override `paintEvent()` on CalibrationOverlay. Store mouse-down point in `mousePressEvent`, draw a `QRect` between press and current pos in `mouseMoveEvent`, finalize on `mouseReleaseEvent`.
- **Frozen frame as background**: Convert the numpy frame from `ScreenCapture.grab_frame()` to `QPixmap` via `QImage(data, w, h, stride, QImage.Format.Format_BGR888)`. Paint as background in `paintEvent`.
- **Dark overlay tint**: Paint a semi-transparent dark rect over the entire screen first, then paint the frozen frame underneath the drawn bbox to create a "spotlight" effect (optional P1).
- **macOS considerations**: `Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint` should work. Test with `WA_TranslucentBackground`. The user has an M4 MacBook Air — ensure Retina scaling is handled (`devicePixelRatio()`).
- **Profile path**: Save to `~/.riverrater/profiles/{config.vision_profile}/` matching the existing `TemplateEngine.save_profile()` path convention.

### Effort Estimate
~12-16 hours (P0 only), ~20-24 hours (P0 + P1)

---

## Feature 2: Confidence Display in HUD

### Problem Statement
`TemplateEngine.detect_cards()` returns `(Card, bbox, confidence_score)` tuples, but the confidence score is never surfaced to the user. The HUD shows analysis results as if detections are certain. Users need to know when the vision system is unsure — a low-confidence detection could mean the wrong card was identified, leading to incorrect advice.

### Goals
- Display per-card confidence scores in the poker HUD when using template-based detection
- Show an overall detection confidence indicator (green/yellow/red)
- Help users understand when to trust the automated reading vs. manually correcting

### Non-Goals
- Confidence calibration (mapping raw scores to true probability)
- Auto-fallback to manual input when confidence is low (separate feature)
- Blackjack mode confidence (blackjack currently uses manual input only)

### Architecture

The confidence data flows through an existing gap in the pipeline:

```
TemplateEngine.detect_cards()
  → list[(Card, bbox, confidence)]     ← confidence exists here
    → GameController._apply_poker_detections()
      → PokerState                      ← confidence lost here
        → analyze_poker()
          → PokerResult                 ← no confidence field
            → HUDOverlay.update_poker()
              → PokerView               ← nothing to display
```

The fix threads confidence through the chain:

```
TemplateEngine.detect_cards()
  → list[(Card, bbox, confidence)]
    → GameController._apply_poker_detections()
      → PokerState + DetectionMeta      ← NEW: store confidence per card
        → PokerResult + confidence      ← NEW: pass through
          → PokerView                   ← NEW: display confidence bar + per-card badges
```

### User Stories

1. **As a poker player using template detection**, I want to see how confident the system is in each card reading so I can catch misreads before they affect my decisions.
2. **As a user calibrating templates**, I want real-time confidence feedback so I know whether my templates are good enough or need re-capturing.

### Requirements

#### P0 — Must Have

| # | Requirement | Acceptance Criteria |
|---|-------------|---------------------|
| 1 | **DetectionMeta dataclass** — New type in `state.py` holding per-card confidence and an overall score. | `DetectionMeta` has `card_confidences: dict[str, float]` (card_str → score 0-1) and `overall_confidence: float` (average). |
| 2 | **Thread confidence through GameController** — `_apply_poker_detections()` stores confidence in a `DetectionMeta` instance on the controller. | Given cards are detected at 0.92 and 0.85 confidence, Then `self._detection_meta.card_confidences` contains both scores and `overall_confidence ≈ 0.885`. |
| 3 | **Add `detection_meta` to PokerResult** — Optional field so results carry confidence data to the HUD. | `PokerResult.detection_meta: Optional[DetectionMeta] = None`. |
| 4 | **Overall confidence indicator in PokerView** — Small colored dot or bar next to the title showing overall detection confidence. Green ≥ 0.85, yellow 0.7-0.85, red < 0.7. | Given overall confidence is 0.78, Then a yellow indicator dot appears in the title bar. |
| 5 | **Per-card confidence badges** — Show confidence % next to each detected card label (when cards are displayed). | Given hole card Ah detected at 0.94 confidence, Then "Ah 94%" appears in the HUD. |

#### P1 — Should Have

| # | Requirement | Acceptance Criteria |
|---|-------------|---------------------|
| 6 | **Low-confidence warning** — When any card has confidence < 0.7, show a yellow warning banner: "Low confidence — verify cards". | Given one card at 0.65 confidence, Then a yellow banner appears below the title. |
| 7 | **Confidence in manual mode** — When using manual input, show confidence as "Manual ✓" (100% implied) to distinguish from auto-detection. | Given cards entered manually, Then the confidence indicator shows "Manual ✓" in white. |

#### P2 — Could Have

| # | Requirement | Acceptance Criteria |
|---|-------------|---------------------|
| 8 | **Confidence history sparkline** — Tiny line chart showing confidence trend over the last 10 detections. | Given 10 detection cycles, Then a 50px-wide sparkline updates in the title bar. |

### Modified Files

| File | Change |
|------|--------|
| `src/riverrater/game/state.py` | Add `DetectionMeta` dataclass. Add `detection_meta: Optional[DetectionMeta] = None` to `PokerResult`. |
| `src/riverrater/main.py` | `_apply_poker_detections()` — build `DetectionMeta` from detection tuples. Store on controller. Pass to `PokerResult` before sending to HUD. |
| `src/riverrater/hud/poker_view.py` | Add confidence dot in title bar. Add per-card confidence badges. Add low-confidence warning banner. |
| `tests/test_poker_view.py` (new) | Test confidence display logic. |
| `tests/test_state.py` | Test `DetectionMeta` creation and `overall_confidence` calculation. |

### Implementation Notes

- **Color thresholds**: Use the same design constants already in poker_view.py — `_CLR_GREEN` for ≥ 0.85, `_CLR_YELLOW` for 0.7-0.85, `_CLR_RED` for < 0.7.
- **Card display**: Currently PokerView doesn't show individual card labels — it just shows aggregate stats. Adding a "Cards: Ah(94%) Kd(88%)" row between the title and the win% would be the cleanest placement.
- **No confidence when manual**: The detection_meta field is Optional. When cards come from manual input, it stays None — PokerView should handle this gracefully (hide confidence elements or show "Manual ✓").
- **Backward compatible**: DetectionMeta is optional on PokerResult, so all existing tests continue to pass.

### Effort Estimate
~6-8 hours (P0 only), ~10-12 hours (P0 + P1)

---

## Implementation Order

| Phase | Feature | Priority | Hours | Dependencies |
|-------|---------|----------|-------|-------------|
| 1 | Confidence Display (P0) | Higher | 6-8 | None — data already flows through detect_cards |
| 2 | Calibration GUI (P0) | Lower | 12-16 | Ideally validated with confidence display active |
| 3 | Confidence Display (P1) | — | 2-4 | Phase 1 |
| 4 | Calibration GUI (P1) | — | 8 | Phase 2 |

**Rationale**: Confidence display is smaller, self-contained, and immediately useful even with manual input. Calibration GUI is larger, involves mouse interaction on macOS that benefits from interactive testing, and is more valuable when confidence display is already showing whether templates are working well.

---

## Open Questions

| # | Question | Owner | Blocking? |
|---|----------|-------|-----------|
| 1 | Should calibration support multi-monitor setups (e.g. poker client on external display, HUD on laptop)? | User | No — start with single capture region |
| 2 | Minimum number of templates per card for reliable detection? (1 per card? 2-3 for robustness?) | Engineering | No — start with 1, add guidance later |
| 3 | Should the confidence display also show in blackjack mode when YOLO is eventually integrated? | User | No — design for poker first, extend later |
| 4 | Profile management — should there be a way to select/switch between profiles from the UI? | User | No — JSON config is fine for now |
