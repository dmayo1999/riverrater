# YOLOv8n Card Detector Weights

Trained weights for live-dealer blackjack card detection (Module 2).

## Export path

Place the Colab training artifact here:

```
models/yolov8n_cards/best.pt
```

`best.pt` is gitignored — commit this README only until a human completes training.

## mAP targets

| Metric | Target | Source |
|--------|--------|--------|
| mAP@0.5 | **≥ 85%** | `docs/BUILD_PLAN.md` Phase 3 exit criteria |
| Inference latency | **< 50 ms/frame** | M4 MacBook Air (YOLOv8n, MPS) |

Evaluate on a held-out set of **live-dealer** frames (`data/training/live_dealer/`), not
studio Roboflow cards alone.

## Training (human — Google Colab)

GPU training is **not** run in CI. A human must train separately:

```bash
# Print full Colab workflow
python scripts/train_yolo.py --colab-instructions

# Validate dataset + CLASS_MAP alignment locally (no GPU)
python scripts/train_yolo.py --dry-run
```

### Prerequisites

1. Bootstrap data: `python scripts/download_roboflow.py`
2. Augment: `python scripts/augment_cards.py --variants 3`
3. (Recommended) Label 500+ live-dealer frames in `data/training/live_dealer/`

### Colab quick start

```python
!git clone https://github.com/dmayo1999/riverrater.git /content/RR
%cd /content/RR
!pip install -e ".[train]"

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="/content/RR/data/training/cards.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)
```

Copy `runs/detect/train/weights/best.pt` → `models/yolov8n_cards/best.pt`.

## CLASS_MAP alignment

YOLO class IDs **must** match `data/training/cards.yaml` and
`riverrater.vision.yolo_engine.YOLOEngine.CLASS_MAP`:

- Ranks: `2 3 4 5 6 7 8 9 T J Q K A`
- Suits: `c d h s`
- Example: ID `0` = `2c`, ID `51` = `As`

If a Roboflow or custom export uses a different order, update **both** `cards.yaml`
`names:` and `YOLOEngine.CLASS_MAP` before training or inference will mislabel cards.

Verify alignment:

```bash
python -m pytest tests/test_yolo_smoke.py::TestClassMapAlignment -v
```

## Integration

Once `best.pt` exists locally:

```python
from riverrater.vision.yolo_engine import YOLOEngine

engine = YOLOEngine("models/yolov8n_cards/best.pt")
assert engine.is_available
detections = engine.detect_cards(frame, confidence=0.5)
```

Install ML dependencies: `pip install -e ".[ml]"`.