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

# ---------------------------------------------------------------------------
# Import Card / Rank / Suit from game.state if available; otherwise fall back
# to local definitions so this module is importable during isolated testing.
# ---------------------------------------------------------------------------
try:
    from riverrater.game.state import Card, Rank, Suit
except ImportError:
    # Local fallback — these mirror the definitions in game/state.py exactly.
    from enum import Enum  # type: ignore[assignment]
    from dataclasses import dataclass as _dataclass

    class Suit(Enum):  # type: ignore[no-redef]
        HEARTS = "h"
        DIAMONDS = "d"
        CLUBS = "c"
        SPADES = "s"

    class Rank(Enum):  # type: ignore[no-redef]
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

    @_dataclass
    class Card:  # type: ignore[no-redef]
        rank: Rank
        suit: Suit

        def __str__(self) -> str:
            return f"{self.rank.value}{self.suit.value}"

        def __hash__(self) -> int:
            return hash((self.rank, self.suit))

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, Card):
                return False
            return self.rank == other.rank and self.suit == other.suit


logger = logging.getLogger(__name__)

# Scale levels used during multi-scale matching.
_SCALES = np.linspace(0.5, 1.5, 7)

# IoU threshold above which two detections are considered duplicates.
_IOU_THRESHOLD = 0.3

# Key used in the .npz archive.
_NPZ_KEY_PREFIX = "card_"


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

        if profile_path is not None:
            self.load_profile(profile_path)

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

        for card, templates in self._templates.items():
            for tmpl in templates:
                tmpl_h, tmpl_w = tmpl.shape[:2]
                if tmpl_h == 0 or tmpl_w == 0:
                    continue

                for scale in _SCALES:
                    new_h = max(1, int(gray_frame.shape[0] * scale))
                    new_w = max(1, int(gray_frame.shape[1] * scale))
                    resized = cv2.resize(gray_frame, (new_w, new_h))

                    # Template must be smaller than the search image.
                    if resized.shape[0] < tmpl_h or resized.shape[1] < tmpl_w:
                        continue

                    result = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(result >= confidence)

                    for py, px in zip(*loc):
                        score = float(result[py, px])
                        # Map coordinates back to original frame space.
                        orig_x = int(px / scale)
                        orig_y = int(py / scale)
                        orig_w = int(tmpl_w / scale)
                        orig_h = int(tmpl_h / scale)
                        candidates.append((card, (orig_x, orig_y, orig_w, orig_h), score))

        return _nms(candidates)

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

        logger.info("Loaded profile from %s (%d cards)", path, len(self._templates))


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
