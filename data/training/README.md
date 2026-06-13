# YOLO Training Data (Phase 3)

Directory layout for training a YOLOv8n card detector for live-dealer blackjack.

## Layout

```
data/training/
├── cards.yaml              # YOLOv8 dataset config (class order matches YOLOEngine)
├── raw/                    # Roboflow bootstrap dataset (YOLOv8 export)
│   ├── train/images|labels
│   ├── valid/images|labels
│   └── test/images|labels
├── augmented/              # Output of scripts/augment_cards.py
│   ├── images/
│   └── labels/
├── live_dealer/            # Your collected stream frames + labels
│   ├── images/
│   └── labels/
└── backgrounds/            # Optional custom felt textures for augmentation
```

`raw/`, `augmented/`, and `live_dealer/` image data are gitignored; template files
(`cards.yaml`, this README) are tracked.

## 1. Bootstrap dataset (Roboflow)

**Automated** (requires free Roboflow API key):

```bash
export ROBOFLOW_API_KEY=your_key
pip install -e ".[train]"
python scripts/download_roboflow.py
```

**Manual** (no API key):

1. Download YOLOv8 export from [Roboflow Universe — playing-cards v4](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d/dataset/4/download/yolov8).
2. Extract into `data/training/raw/`.

## 2. Augment for live-dealer conditions

Simulates glare, blur, rotation, perspective, and green felt backgrounds:

```bash
python scripts/augment_cards.py --variants 3
```

Defaults read `raw/train/` and write to `augmented/`. Use `--no-original` to emit
only synthetic variants.

## 3. Collect live-dealer frames

Studio Roboflow cards alone are not enough for casino streams. Collect **500+**
labelled frames from actual live-dealer blackjack video.

### Capture workflow

1. **Region** — Use the same MSS capture region as production (`~/.riverrater/config.json`).
2. **Record** — Save PNG/JPEG frames when cards are fully visible (avoid heavy motion blur).
3. **Store** — Place raw frames in `live_dealer/images/`.
4. **Label** — Annotate with [Roboflow Annotate](https://roboflow.com/annotate), [CVAT](https://www.cvat.ai/), or [LabelImg](https://github.com/heartexlabs/labelImg). Export YOLO `.txt` labels into `live_dealer/labels/`.
5. **Class names** — Use the 52-class ordering in `cards.yaml` (must match `YOLOEngine.CLASS_MAP`).
6. **Merge** — Copy labelled live-dealer pairs into `augmented/images` and `augmented/labels` before training, or add a second `train:` path in a Colab-specific yaml.

### Labelling tips

| Condition | Why it matters |
|-----------|----------------|
| Dealer hand partial occlusion | Common in live streams |
| Glare / reflections | Studio dataset lacks this |
| Low contrast on green felt | Domain gap vs flat backgrounds |
| Slight motion blur | Cards move during deal |
| Varying scale | Camera zoom / seat position |

### Quick frame grab (example)

```python
from pathlib import Path
import cv2
from riverrater.capture.screen import ScreenCapture  # adjust to your config

out = Path("data/training/live_dealer/images")
out.mkdir(parents=True, exist_ok=True)
# capture = ScreenCapture(...)
# frame = capture.grab()
# cv2.imwrite(str(out / "frame_0001.png"), frame)
```

Run augmentation on live-dealer data once labelled:

```bash
python scripts/augment_cards.py \
  --input-images data/training/live_dealer/images \
  --input-labels data/training/live_dealer/labels \
  --output-images data/training/augmented/images \
  --output-labels data/training/augmented/labels
```

## 4. Train on Google Colab

**Scaffold (local validation + Colab instructions):**

```bash
python scripts/train_yolo.py --dry-run          # CLASS_MAP + paths check
python scripts/train_yolo.py --colab-instructions  # full Colab workflow
```

**Train on Colab (T4 GPU — human step, not CI):**

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="/content/RR/data/training/cards.yaml", epochs=100, imgsz=640)
```

Copy `runs/detect/train/weights/best.pt` to `models/yolov8n_cards/best.pt` in this repo.
Target: **mAP@0.5 ≥ 85%** on held-out live-dealer frames. See `models/yolov8n_cards/README.md`.

## Class order reference

Ranks `2 3 4 5 6 7 8 9 T J Q K A` × suits `c d h s` → IDs `0–51`.
Must match `YOLOEngine.CLASS_MAP` and the `names:` list in `cards.yaml` (verified by
`tests/test_yolo_smoke.py::TestClassMapAlignment`). Update both if your label export
uses a different order.