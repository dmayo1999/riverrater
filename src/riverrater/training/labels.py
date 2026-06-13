"""
YOLO-format label parsing and bounding-box geometry helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class YoloLabel:
    """Single YOLO detection label (normalized coordinates)."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def as_line(self) -> str:
        return (
            f"{self.class_id} {self.cx:.6f} {self.cy:.6f} "
            f"{self.w:.6f} {self.h:.6f}"
        )


def parse_label_line(line: str) -> YoloLabel | None:
    """Parse one YOLO label line; return ``None`` for blanks or comments."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.split()
    if len(parts) < 5:
        return None
    class_id = int(parts[0])
    cx, cy, w, h = (float(parts[i]) for i in range(1, 5))
    return YoloLabel(class_id=class_id, cx=cx, cy=cy, w=w, h=h)


def parse_label_file(path: Path) -> list[YoloLabel]:
    """Load all labels from a ``.txt`` file."""
    if not path.is_file():
        return []
    labels: list[YoloLabel] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        label = parse_label_line(line)
        if label is not None:
            labels.append(label)
    return labels


def write_label_file(path: Path, labels: list[YoloLabel]) -> None:
    """Write labels to a YOLO ``.txt`` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(label.as_line() for label in labels)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def labels_to_pixel_boxes(
    labels: list[YoloLabel],
    width: int,
    height: int,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """
    Convert labels to pixel axis-aligned boxes.

    Returns ``(class_id, (x1, y1, x2, y2))`` tuples.
    """
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for label in labels:
        box_w = label.w * width
        box_h = label.h * height
        cx = label.cx * width
        cy = label.cy * height
        x1 = cx - box_w / 2.0
        y1 = cy - box_h / 2.0
        x2 = cx + box_w / 2.0
        y2 = cy + box_h / 2.0
        boxes.append((label.class_id, (x1, y1, x2, y2)))
    return boxes


def pixel_boxes_to_labels(
    boxes: list[tuple[int, tuple[float, float, float, float]]],
    width: int,
    height: int,
    *,
    min_area_ratio: float = 1e-4,
) -> list[YoloLabel]:
    """
    Convert pixel boxes back to normalized YOLO labels.

    Boxes with negligible area after clipping are dropped.
    """
    if width <= 0 or height <= 0:
        return []

    labels: list[YoloLabel] = []
    min_area = width * height * min_area_ratio

    for class_id, (x1, y1, x2, y2) in boxes:
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue

        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height

        labels.append(
            YoloLabel(
                class_id=class_id,
                cx=_clamp01(cx),
                cy=_clamp01(cy),
                w=_clamp01(w),
                h=_clamp01(h),
            )
        )
    return labels


def transform_box_corners(
    corners: list[tuple[float, float]],
    matrix: object,
) -> list[tuple[float, float]]:
    """Apply an OpenCV 2×3 affine or 3×3 perspective matrix to corner points."""
    import cv2
    import numpy as np

    pts = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
    mat = np.asarray(matrix, dtype=np.float32)
    if mat.shape == (2, 3):
        transformed = cv2.transform(pts, mat)
    elif mat.shape == (3, 3):
        transformed = cv2.perspectiveTransform(pts, mat)
    else:
        raise ValueError(f"Unsupported transform matrix shape: {mat.shape}")
    flat = transformed.reshape(-1, 2)
    return [(float(x), float(y)) for x, y in flat]


def boxes_from_corner_sets(
    corner_sets: list[tuple[int, list[tuple[float, float]]]],
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Build axis-aligned pixel boxes from four transformed corners each."""
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for class_id, corners in corner_sets:
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        boxes.append((class_id, (min(xs), min(ys), max(xs), max(ys))))
    return boxes


def box_corners(x1: float, y1: float, x2: float, y2: float) -> list[tuple[float, float]]:
    """Return the four corners of an axis-aligned box."""
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))