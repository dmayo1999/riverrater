"""
Cross-frame card tracking for YOLO blackjack detections.

Physical cards persist on screen across multiple frames.  This module
tracks dealt cards by class + bounding-box proximity so each physical card
is counted at most once in :attr:`~riverrater.game.state.BlackjackState.cards_seen`,
while still allowing multiple copies of the same rank/suit in a multi-deck shoe
when they appear at different table positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from riverrater.game.state import Card
from riverrater.vision.template_engine import _compute_iou


@dataclass
class _TrackedCard:
    card: Card
    bbox: tuple[int, int, int, int]


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _center_distance(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax, ay = _bbox_center(box_a)
    bx, by = _bbox_center(box_b)
    return math.hypot(ax - bx, ay - by)


class CardTracker:
    """Track physical cards across frames using class + bbox proximity.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU between bboxes to treat detections as the same card.
    center_distance_threshold:
        Maximum center-to-center pixel distance to treat detections as the
        same card when IoU is low (e.g. slight motion between frames).
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        center_distance_threshold: float = 40.0,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._center_distance_threshold = center_distance_threshold
        self._tracked: list[_TrackedCard] = []

    def reset(self) -> None:
        """Clear all tracked cards (e.g. on shoe reset)."""
        self._tracked.clear()

    def register_detections(
        self,
        detections: list[tuple[Card, tuple[int, int, int, int], float]],
    ) -> list[tuple[Card, tuple[int, int, int, int], float]]:
        """Register frame detections and return only newly seen physical cards.

        A detection is *new* when no previously tracked card shares the same
        :class:`Card` identity and a nearby bounding box (IoU or center
        distance within threshold).

        Matched detections update the stored bbox to the latest observation.
        """
        new_detections: list[tuple[Card, tuple[int, int, int, int], float]] = []

        for card, bbox, confidence in detections:
            match_idx = self._find_match(card, bbox)
            if match_idx is not None:
                self._tracked[match_idx].bbox = bbox
            else:
                self._tracked.append(_TrackedCard(card=card, bbox=bbox))
                new_detections.append((card, bbox, confidence))

        return new_detections

    def _find_match(
        self,
        card: Card,
        bbox: tuple[int, int, int, int],
    ) -> int | None:
        for idx, tracked in enumerate(self._tracked):
            if tracked.card != card:
                continue
            if self._bboxes_match(tracked.bbox, bbox):
                return idx
        return None

    def _bboxes_match(
        self,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> bool:
        if _compute_iou(box_a, box_b) >= self._iou_threshold:
            return True
        return _center_distance(box_a, box_b) <= self._center_distance_threshold