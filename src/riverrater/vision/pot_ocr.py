"""
OpenCV-based OCR for poker pot and bet-to-call UI regions.

Reads numeric chip amounts from calibrated screen ROIs using contour
segmentation and template matching against built-in digit glyphs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)

# Characters rendered into templates and recognised in ROIs.
_DIGITS = "0123456789"
_DECIMAL = "."
_TEMPLATE_CHARS = _DIGITS + _DECIMAL

# Font used for synthetic templates and test image generation.
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.9
_FONT_THICKNESS = 2
_TEMPLATE_CANVAS = (48, 48)
_NORMALIZED_GLYPH_HEIGHT = 22

# Minimum normalised match score to accept a single glyph.
_MIN_GLYPH_SCORE = 0.45

# Projection segmentation: column sums below this fraction of ROI height are gaps.
_PROJECTION_GAP_RATIO = 0.08


@dataclass(frozen=True)
class PotOCRResult:
    """Parsed monetary value and aggregate recognition confidence."""

    value: float
    confidence: float


def _tight_binary_glyph(char: str) -> np.ndarray:
    """Render *char* with the shared font and return a tight binarised patch."""
    width, height = _TEMPLATE_CANVAS
    canvas = render_amount_image(char, size=(width, height))
    binary = _to_binary(canvas)
    rows = np.where(np.any(binary > 0, axis=1))[0]
    cols = np.where(np.any(binary > 0, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return binary
    return binary[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def _normalize_glyph(patch: np.ndarray, target_height: int = _NORMALIZED_GLYPH_HEIGHT) -> np.ndarray:
    """Resize a glyph patch to a fixed height while preserving aspect ratio."""
    if patch.size == 0:
        return patch
    patch_h, patch_w = patch.shape[:2]
    scale = target_height / max(patch_h, 1)
    target_width = max(4, int(round(patch_w * scale)))
    return cv2.resize(patch, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _build_char_templates() -> dict[str, np.ndarray]:
    """Create glyph templates from the same render pipeline used in tests."""
    templates: dict[str, np.ndarray] = {}
    for char in _TEMPLATE_CHARS:
        templates[char] = _normalize_glyph(_tight_binary_glyph(char))
    return templates


def render_amount_image(
    text: str,
    size: tuple[int, int] = (180, 48),
    *,
    background: tuple[int, int, int] = (24, 72, 40),
    foreground: tuple[int, int, int] = (240, 240, 240),
) -> np.ndarray:
    """
    Render a synthetic pot/bet label for tests and calibration previews.

    Parameters
    ----------
    text:
        Amount string such as ``"150.50"`` or ``"$1,250"``.
    size:
        ``(width, height)`` of the output BGR image.
    background, foreground:
        BGR colours for the chip-label background and text.
    """
    width, height = size
    image = np.full((height, width, 3), background, dtype=np.uint8)
    display = "".join(ch for ch in text if ch in _TEMPLATE_CHARS)
    if not display:
        return image

    text_size, _ = cv2.getTextSize(display, _FONT, _FONT_SCALE, _FONT_THICKNESS)
    text_w, text_h = text_size
    origin = (max(4, (width - text_w) // 2), (height + text_h) // 2 - 2)
    cv2.putText(
        image,
        display,
        origin,
        _FONT,
        _FONT_SCALE,
        foreground,
        _FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    return image


def _crop_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Extract a clamped BGR sub-image from *frame*."""
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError(f"ROI dimensions must be positive (got w={w}, h={h}).")

    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"ROI ({x}, {y}, {w}, {h}) does not intersect frame of size ({frame_w}, {frame_h})."
        )
    return frame[y1:y2, x1:x2].copy()


def _to_binary(roi: np.ndarray) -> np.ndarray:
    """Threshold ROI to white glyphs on black background."""
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Normalise so characters are bright on a dark field.
    if float(np.mean(binary)) > 127.0:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary


def _segment_glyphs(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return left-to-right glyph bounding boxes via vertical projection."""
    roi_h, roi_w = binary.shape[:2]
    if roi_h == 0 or roi_w == 0:
        return []

    col_sum = np.sum(binary > 0, axis=0)
    threshold = max(1, int(roi_h * _PROJECTION_GAP_RATIO))
    in_char = False
    segments: list[tuple[int, int]] = []
    start = 0

    for idx, value in enumerate(col_sum):
        if value > threshold and not in_char:
            start = idx
            in_char = True
        elif value <= threshold and in_char:
            if idx - start >= 2:
                segments.append((start, idx))
            in_char = False

    if in_char and roi_w - start >= 2:
        segments.append((start, roi_w))

    boxes: list[tuple[int, int, int, int]] = []
    for x1, x2 in segments:
        column = binary[:, x1:x2]
        rows = np.where(np.any(column > 0, axis=1))[0]
        if len(rows) == 0:
            continue
        y1 = int(rows[0])
        y2 = int(rows[-1]) + 1
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


def _average_digit_width(templates: dict[str, np.ndarray]) -> int:
    """Estimate the average rendered digit width for merged-glyph splitting."""
    widths = [templates[digit].shape[1] for digit in _DIGITS if digit in templates]
    if not widths:
        return 12
    return max(8, int(round(sum(widths) / len(widths))))


def _pad_glyph(patch: np.ndarray, width: int) -> np.ndarray:
    """Pad a glyph patch to a fixed width on the right."""
    if patch.size == 0:
        return np.zeros((_NORMALIZED_GLYPH_HEIGHT, width), dtype=np.uint8)
    height = patch.shape[0]
    canvas = np.zeros((height, width), dtype=np.uint8)
    copy_w = min(width, patch.shape[1])
    canvas[:, :copy_w] = patch[:, :copy_w]
    return canvas


def _match_glyph(
    patch: np.ndarray,
    templates: dict[str, np.ndarray],
) -> tuple[Optional[str], float]:
    """Classify a single glyph patch via normalised cross-correlation."""
    if patch.size == 0:
        return None, 0.0

    patch_gray = patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    normalized = _normalize_glyph(patch_gray)
    compare_width = max(normalized.shape[1], max(tmpl.shape[1] for tmpl in templates.values()))
    padded_patch = _pad_glyph(normalized, compare_width)

    best_char: Optional[str] = None
    best_score = 0.0

    for char, template in templates.items():
        padded_template = _pad_glyph(template, compare_width)
        result = cv2.matchTemplate(padded_patch, padded_template, cv2.TM_CCOEFF_NORMED)
        score = float(result.max()) if result.size else 0.0
        if score > best_score:
            best_score = score
            best_char = char

    if best_char is None or best_score < _MIN_GLYPH_SCORE:
        return None, best_score
    return best_char, best_score


def _split_wide_glyph(
    patch: np.ndarray,
    avg_digit_width: int,
) -> list[np.ndarray]:
    """Split a merged column (e.g. ``00``) into single-digit patches."""
    patch_w = patch.shape[1]
    if patch_w <= int(avg_digit_width * 1.4):
        return [patch]

    digit_count = max(2, int(round(patch_w / avg_digit_width)))
    slice_width = max(1, patch_w // digit_count)
    slices: list[np.ndarray] = []
    for idx in range(digit_count):
        x1 = idx * slice_width
        x2 = patch_w if idx == digit_count - 1 else (idx + 1) * slice_width
        slices.append(patch[:, x1:x2])
    return slices


def _classify_glyph_patch(
    patch: np.ndarray,
    templates: dict[str, np.ndarray],
    avg_digit_width: int,
) -> list[tuple[str, float]]:
    """Classify a projection segment, splitting wide merged digits when needed."""
    patch_h, patch_w = patch.shape[:2]

    if patch_w <= 6 and patch_h <= max(10, _NORMALIZED_GLYPH_HEIGHT // 2):
        return [(_DECIMAL, 1.0)]

    results: list[tuple[str, float]] = []
    for sub_patch in _split_wide_glyph(patch, avg_digit_width):
        char, score = _match_glyph(sub_patch, templates)
        if char is None:
            return []
        results.append((char, score))
    return results


def _parse_amount_from_roi(
    roi: np.ndarray,
    templates: dict[str, np.ndarray],
) -> Optional[PotOCRResult]:
    """Parse a numeric amount from a single ROI crop."""
    binary = _to_binary(roi)
    boxes = _segment_glyphs(binary)
    if not boxes:
        return None

    chars: list[str] = []
    scores: list[float] = []
    avg_digit_width = _average_digit_width(templates)

    for x, y, w, h in boxes:
        patch = binary[y : y + h, x : x + w]
        classified = _classify_glyph_patch(patch, templates, avg_digit_width)
        if not classified:
            return None
        for char, score in classified:
            chars.append(char)
            scores.append(score)

    raw = "".join(chars)
    if _DECIMAL in raw:
        parts = raw.split(_DECIMAL)
        integer_part = parts[0] or "0"
        fractional_part = parts[1] if len(parts) > 1 else ""
        cleaned = integer_part
        if fractional_part:
            cleaned = f"{integer_part}.{fractional_part}"
    else:
        cleaned = raw

    cleaned = "".join(ch for ch in cleaned if ch in _TEMPLATE_CHARS)
    if not cleaned or cleaned == _DECIMAL:
        return None

    try:
        value = float(cleaned)
    except ValueError:
        return None

    confidence = float(sum(scores) / len(scores)) if scores else 0.0
    return PotOCRResult(value=value, confidence=confidence)


def _parse_roi_entry(entry: object) -> Optional[tuple[int, int, int, int]]:
    """Parse a JSON ROI list into ``(x, y, w, h)``."""
    if not isinstance(entry, (list, tuple)) or len(entry) != 4:
        return None
    x, y, w, h = (int(v) for v in entry)
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def load_profile_rois(profile_path: str | Path) -> tuple[Optional[tuple[int, int, int, int]], Optional[tuple[int, int, int, int]]]:
    """
    Load optional ``pot_roi`` and ``bet_roi`` from a vision profile directory.

    Returns ``(None, None)`` when *regions.json* is missing or has no pot/bet
    entries.
    """
    regions_path = Path(profile_path) / "regions.json"
    if not regions_path.exists():
        return None, None

    with regions_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    pot_roi = _parse_roi_entry(payload.get("pot_roi"))
    bet_roi = _parse_roi_entry(payload.get("bet_roi"))
    return pot_roi, bet_roi


def resolve_pot_rois(
    config_pot_roi: Optional[tuple[int, int, int, int]],
    config_bet_roi: Optional[tuple[int, int, int, int]],
    profile_path: Optional[str | Path] = None,
) -> tuple[Optional[tuple[int, int, int, int]], Optional[tuple[int, int, int, int]]]:
    """
    Resolve pot/bet ROIs with AppConfig values overriding profile defaults.
    """
    profile_pot: Optional[tuple[int, int, int, int]] = None
    profile_bet: Optional[tuple[int, int, int, int]] = None
    if profile_path is not None:
        profile_pot, profile_bet = load_profile_rois(profile_path)

    pot_roi = config_pot_roi if config_pot_roi is not None else profile_pot
    bet_roi = config_bet_roi if config_bet_roi is not None else profile_bet
    return pot_roi, bet_roi


class PotOCR:
    """
    Read pot size and bet-to-call amounts from calibrated screen regions.

    Parameters
    ----------
    pot_roi, bet_roi:
        Optional ``(x, y, w, h)`` regions in frame coordinates.  When
        ``None``, the corresponding :meth:`read_pot` / :meth:`read_bet`
        call returns ``None``.
    confidence_threshold:
        Minimum aggregate glyph confidence required to accept a parse.
    templates:
        Optional pre-built glyph templates (mainly for testing).
    """

    def __init__(
        self,
        pot_roi: Optional[tuple[int, int, int, int]] = None,
        bet_roi: Optional[tuple[int, int, int, int]] = None,
        confidence_threshold: float = 0.6,
        templates: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        self.pot_roi = pot_roi
        self.bet_roi = bet_roi
        self.confidence_threshold = confidence_threshold
        self._templates = templates or _build_char_templates()

    def read_pot(self, frame: np.ndarray) -> Optional[PotOCRResult]:
        """Parse the pot-size ROI from *frame*."""
        if self.pot_roi is None:
            return None
        return self._read_roi(frame, self.pot_roi)

    def read_bet(self, frame: np.ndarray) -> Optional[PotOCRResult]:
        """Parse the bet-to-call ROI from *frame*."""
        if self.bet_roi is None:
            return None
        return self._read_roi(frame, self.bet_roi)

    def read_values(
        self,
        frame: np.ndarray,
    ) -> tuple[Optional[PotOCRResult], Optional[PotOCRResult]]:
        """Parse both pot and bet ROIs when configured."""
        return self.read_pot(frame), self.read_bet(frame)

    def _read_roi(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int],
    ) -> Optional[PotOCRResult]:
        if frame is None or frame.size == 0:
            return None

        try:
            crop = _crop_roi(frame, roi)
        except ValueError as exc:
            logger.debug("Pot OCR ROI crop failed: %s", exc)
            return None

        result = _parse_amount_from_roi(crop, self._templates)
        if result is None:
            return None
        if result.confidence < self.confidence_threshold:
            logger.debug(
                "Pot OCR below threshold: value=%.2f confidence=%.3f < %.3f",
                result.value,
                result.confidence,
                self.confidence_threshold,
            )
            return None
        return result