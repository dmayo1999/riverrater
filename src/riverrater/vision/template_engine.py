"""
OpenCV template-matching engine for digital poker card detection.

Each card can have multiple template images registered (for robustness across
different lighting conditions, card backs, or slight size variations).
Detection uses multi-scale matching followed by Non-Maximum Suppression (NMS).
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from riverrater.game.state import Card, Rank, Suit


logger = logging.getLogger(__name__)

# Scale levels used during multi-scale matching.
_SCALES = np.linspace(0.5, 1.5, 7)

# IoU threshold above which two detections are considered duplicates.
_IOU_THRESHOLD = 0.3

# Key used in the .npz archive.
_NPZ_KEY_PREFIX = "card_"

# Extra pixels added around each saved card-slot ROI before template matching.
_ROI_PADDING = 8


def _compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """
    Compute Intersection-over-Union between two axis-aligned bounding boxes.

    Parameters
    ----------
    box_a, box_b:
        ``(x, y, w, h)`` bounding boxes.

    Returns
    -------
    float
        IoU in [0, 1].
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _nms(
    detections: list[tuple[Card, tuple[int, int, int, int], float]],
    iou_threshold: float = _IOU_THRESHOLD,
) -> list[tuple[Card, tuple[int, int, int, int], float]]:
    """
    Apply Non-Maximum Suppression to eliminate duplicate detections.

    Detections are sorted by confidence (descending).  Any detection that
    overlaps an already-kept detection by more than *iou_threshold* is
    discarded.

    Parameters
    ----------
    detections:
        List of ``(Card, bbox, confidence)`` tuples.
    iou_threshold:
        IoU threshold above which overlapping boxes are suppressed.

    Returns
    -------
    list
        Pruned list of ``(Card, bbox, confidence)`` tuples.
    """
    if not detections:
        return []

    # Sort by confidence descending.
    sorted_dets = sorted(detections, key=lambda d: d[2], reverse=True)
    kept: list[tuple[Card, tuple[int, int, int, int], float]] = []

    for det in sorted_dets:
        _, bbox, _ = det
        suppress = False
        for _, kept_bbox, _ in kept:
            if _compute_iou(bbox, kept_bbox) > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(det)

    return kept


def _padded_roi(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_w: int,
    frame_h: int,
    padding: int = _ROI_PADDING,
) -> tuple[int, int, int, int]:
    """
    Expand an ROI by *padding* pixels and clamp to the frame bounds.

    Returns
    -------
    tuple[int, int, int, int]
        ``(x, y, w, h)`` of the padded, clamped region.
    """
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame_w, x + w + padding)
    y2 = min(frame_h, y + h + padding)
    return x1, y1, x2 - x1, y2 - y1


class TemplateEngine:
    """
    Card detection engine based on OpenCV template matching.

    Supports multiple templates per card and multi-scale search so that
    cards appearing at slightly different sizes in the captured frame are
    still reliably detected.

    Parameters
    ----------
    profile_path:
        If provided, :meth:`load_profile` is called immediately with this path.
    """

    def __init__(self, profile_path: Optional[str] = None) -> None:
        # Dict mapping Card -> list of grayscale template images.
        self._templates: dict[Card, list[np.ndarray]] = defaultdict(list)
        # Optional per-slot search regions from calibration (x, y, w, h).
        self._roi_regions: list[tuple[int, int, int, int]] = []

        if profile_path is not None:
            self.load_profile(profile_path)

    @property
    def roi_regions(self) -> list[tuple[int, int, int, int]]:
        """Card-slot ROIs loaded from calibration; empty means full-frame search."""
        return list(self._roi_regions)

    @property
    def uses_roi_scoped_search(self) -> bool:
        """True when detection is limited to saved card-slot regions."""
        return bool(self._roi_regions)

    def set_roi_regions(self, regions: list[tuple[int, int, int, int]]) -> None:
        """
        Define card-slot search regions for :meth:`detect_cards`.

        Parameters
        ----------
        regions:
            List of ``(x, y, w, h)`` bounding boxes in frame coordinates.
            An empty list restores full-frame matching.
        """
        self._roi_regions = list(regions)
        logger.debug("Set %d ROI region(s) for scoped card search.", len(self._roi_regions))

    # ------------------------------------------------------------------ #
    # Template management
    # ------------------------------------------------------------------ #

    def add_template(self, card: Card, template_image: np.ndarray) -> None:
        """
        Register a new template image for *card*.

        The image is converted to grayscale if necessary.

        Parameters
        ----------
        card:
            The :class:`Card` this template represents.
        template_image:
            BGR or grayscale ``uint8`` numpy array.
        """
        gray = _to_gray(template_image)
        self._templates[card].append(gray)
        logger.debug("Added template for %s (total: %d)", card, len(self._templates[card]))

    def remove_template(self, card: Card) -> None:
        """
        Remove all templates for *card*.

        Parameters
        ----------
        card:
            The card whose templates should be deleted.
        """
        if card in self._templates:
            del self._templates[card]
            logger.debug("Removed all templates for %s", card)

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #

    def detect_cards(
        self,
        frame: np.ndarray,
        confidence: float = 0.8,
    ) -> list[tuple[Card, tuple[int, int, int, int], float]]:
        """
        Detect cards in *frame* using multi-scale template matching.

        Algorithm
        ---------
        1. Convert *frame* to grayscale.
        2. For each registered card and each of its templates:
           a. Resize the source frame at :data:`_SCALES` (7 scale levels,
              0.5× to 1.5×).
           b. Run ``cv2.matchTemplate`` with ``TM_CCOEFF_NORMED``.
           c. Record all locations whose score ≥ *confidence*.
        3. Apply NMS across all candidate detections to eliminate duplicates.

        Parameters
        ----------
        frame:
            BGR or grayscale ``uint8`` numpy array (the captured screen frame).
        confidence:
            Minimum normalised cross-correlation score to accept as a
            detection (0–1).

        Returns
        -------
        list of (Card, bbox, score)
            Each entry is ``(card, (x, y, w, h), confidence_score)``.
        """
        gray_frame = _to_gray(frame)
        candidates: list[tuple[Card, tuple[int, int, int, int], float]] = []

        for search_gray, offset_x, offset_y in self._iter_search_areas(gray_frame):
            for card, templates in self._templates.items():
                for tmpl in templates:
                    tmpl_h, tmpl_w = tmpl.shape[:2]
                    if tmpl_h == 0 or tmpl_w == 0:
                        continue

                    for scale in _SCALES:
                        new_h = max(1, int(search_gray.shape[0] * scale))
                        new_w = max(1, int(search_gray.shape[1] * scale))
                        resized = cv2.resize(search_gray, (new_w, new_h))

                        # Template must be smaller than the search image.
                        if resized.shape[0] < tmpl_h or resized.shape[1] < tmpl_w:
                            continue

                        result = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
                        if self.uses_roi_scoped_search:
                            # One card per calibrated slot — peak match is enough.
                            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
                            if max_val >= confidence:
                                px, py = max_loc
                                score = float(max_val)
                                orig_x = int(px / scale) + offset_x
                                orig_y = int(py / scale) + offset_y
                                orig_w = int(tmpl_w / scale)
                                orig_h = int(tmpl_h / scale)
                                candidates.append(
                                    (card, (orig_x, orig_y, orig_w, orig_h), score)
                                )
                        else:
                            loc = np.where(result >= confidence)
                            for py, px in zip(*loc):
                                score = float(result[py, px])
                                orig_x = int(px / scale) + offset_x
                                orig_y = int(py / scale) + offset_y
                                orig_w = int(tmpl_w / scale)
                                orig_h = int(tmpl_h / scale)
                                candidates.append(
                                    (card, (orig_x, orig_y, orig_w, orig_h), score)
                                )

        return _nms(candidates)

    def _iter_search_areas(
        self,
        gray_frame: np.ndarray,
    ) -> list[tuple[np.ndarray, int, int]]:
        """
        Yield search images for template matching.

        When card-slot ROIs are configured, each ROI (with padding) is cropped
        from *gray_frame*.  Otherwise the full frame is searched once.
        """
        if not self._roi_regions:
            return [(gray_frame, 0, 0)]

        frame_h, frame_w = gray_frame.shape[:2]
        areas: list[tuple[np.ndarray, int, int]] = []
        for x, y, w, h in self._roi_regions:
            if w <= 0 or h <= 0:
                continue
            roi_x, roi_y, roi_w, roi_h = _padded_roi(x, y, w, h, frame_w, frame_h)
            crop = gray_frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            if crop.size == 0:
                continue
            areas.append((crop, roi_x, roi_y))

        return areas

    # ------------------------------------------------------------------ #
    # Profile persistence
    # ------------------------------------------------------------------ #

    def save_profile(self, path: str) -> None:
        """
        Save all templates to a profile directory.

        Profile structure::

            <path>/
                metadata.json           # card labels → npz keys
                card_templates.npz      # numpy arrays

        Parameters
        ----------
        path:
            Destination directory.  Created automatically if it does not exist.
        """
        profile_dir = Path(path)
        profile_dir.mkdir(parents=True, exist_ok=True)

        npz_data: dict[str, np.ndarray] = {}
        metadata: dict[str, list[str]] = {}  # card_str -> list of npz keys

        for card, templates in self._templates.items():
            card_str = str(card)
            keys: list[str] = []
            for idx, tmpl in enumerate(templates):
                key = f"{_NPZ_KEY_PREFIX}{card_str}_{idx}"
                npz_data[key] = tmpl
                keys.append(key)
            metadata[card_str] = keys

        npz_path = profile_dir / "card_templates.npz"
        np.savez(str(npz_path), **npz_data)

        meta_path = profile_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        regions_path = profile_dir / "regions.json"
        if self._roi_regions:
            regions_payload: dict[str, object] = {
                "card_slots": [list(region) for region in self._roi_regions],
            }
            existing_pot, existing_bet = self._read_pot_bet_rois(regions_path)
            if existing_pot is not None:
                regions_payload["pot_roi"] = list(existing_pot)
            if existing_bet is not None:
                regions_payload["bet_roi"] = list(existing_bet)
            with regions_path.open("w", encoding="utf-8") as fh:
                json.dump(regions_payload, fh, indent=2)
        elif regions_path.exists():
            regions_path.unlink()

        logger.info("Saved profile to %s (%d cards)", path, len(metadata))

    def load_profile(self, path: str) -> None:
        """
        Load templates from a profile directory.

        Existing templates are replaced by those in the profile.

        Parameters
        ----------
        path:
            Directory created by :meth:`save_profile`.

        Raises
        ------
        FileNotFoundError
            If *metadata.json* or *card_templates.npz* do not exist.
        ValueError
            If a card string in the metadata cannot be parsed.
        """
        profile_dir = Path(path)
        meta_path = profile_dir / "metadata.json"
        npz_path = profile_dir / "card_templates.npz"

        if not meta_path.exists():
            raise FileNotFoundError(f"Profile metadata not found: {meta_path}")
        if not npz_path.exists():
            raise FileNotFoundError(f"Profile npz not found: {npz_path}")

        with meta_path.open("r", encoding="utf-8") as fh:
            metadata: dict[str, list[str]] = json.load(fh)

        archive = np.load(str(npz_path))

        self._templates.clear()
        for card_str, keys in metadata.items():
            card = _parse_card_string(card_str)
            for key in keys:
                if key in archive:
                    self._templates[card].append(archive[key])
                else:
                    logger.warning("Key %s missing from npz archive; skipping.", key)

        self._roi_regions = self._load_roi_regions(profile_dir)

        logger.info(
            "Loaded profile from %s (%d cards, %d ROI region(s))",
            path,
            len(self._templates),
            len(self._roi_regions),
        )

    def _load_roi_regions(self, profile_dir: Path) -> list[tuple[int, int, int, int]]:
        """Load optional card-slot regions saved during calibration."""
        regions_path = profile_dir / "regions.json"
        if not regions_path.exists():
            return []

        with regions_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        raw_regions = payload.get("card_slots", [])
        regions: list[tuple[int, int, int, int]] = []
        for entry in raw_regions:
            if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                logger.warning("Skipping invalid ROI entry in %s: %r", regions_path, entry)
                continue
            x, y, w, h = (int(v) for v in entry)
            if w > 0 and h > 0:
                regions.append((x, y, w, h))

        return regions

    @staticmethod
    def _read_pot_bet_rois(
        regions_path: Path,
    ) -> tuple[Optional[tuple[int, int, int, int]], Optional[tuple[int, int, int, int]]]:
        """Preserve pot/bet ROIs already stored in regions.json."""
        if not regions_path.exists():
            return None, None

        with regions_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        pot_roi = TemplateEngine._parse_optional_roi(payload.get("pot_roi"))
        bet_roi = TemplateEngine._parse_optional_roi(payload.get("bet_roi"))
        return pot_roi, bet_roi

    @staticmethod
    def _parse_optional_roi(entry: object) -> Optional[tuple[int, int, int, int]]:
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            return None
        x, y, w, h = (int(v) for v in entry)
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h


# ---------------------------------------------------------------------------
# Utility helpers (module-private)
# ---------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert *image* to grayscale if it is a colour image."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _parse_card_string(card_str: str) -> Card:
    """
    Parse a two-character card string like ``"Ah"`` or ``"Td"`` into a
    :class:`Card`.

    Parameters
    ----------
    card_str:
        Two-character string: first char is rank, second is suit.

    Returns
    -------
    Card

    Raises
    ------
    ValueError
        If the string cannot be parsed.
    """
    if len(card_str) != 2:
        raise ValueError(f"Card string must be exactly 2 characters: {card_str!r}")

    rank_char = card_str[0].upper()
    suit_char = card_str[1].lower()

    try:
        rank = Rank(rank_char)
    except ValueError:
        raise ValueError(f"Unknown rank character: {rank_char!r}")

    try:
        suit = Suit(suit_char)
    except ValueError:
        raise ValueError(f"Unknown suit character: {suit_char!r}")

    return Card(rank=rank, suit=suit)
