"""Tests for cross-frame YOLO card deduplication."""

import pytest

from riverrater.game.state import Card, Rank, Suit
from riverrater.vision.card_tracker import CardTracker


def _card(rank: str, suit: str) -> Card:
    return Card(rank=Rank(rank), suit=Suit(suit))


def _det(card: Card, x: int, y: int, w: int = 40, h: int = 60, conf: float = 0.9):
    return (card, (x, y, w, h), conf)


class TestCardTrackerNewDetections:
    def test_first_detection_is_new(self) -> None:
        tracker = CardTracker()
        card = _card("7", "c")
        detections = [_det(card, 10, 20)]

        new = tracker.register_detections(detections)

        assert new == detections

    def test_same_card_same_bbox_across_frames_counted_once(self) -> None:
        tracker = CardTracker()
        card = _card("A", "h")
        detections = [_det(card, 50, 100)]

        first = tracker.register_detections(detections)
        assert len(first) == 1

        for _ in range(9):
            again = tracker.register_detections(detections)
            assert again == []

    def test_same_card_slightly_shifted_bbox_still_matched(self) -> None:
        tracker = CardTracker()
        card = _card("K", "s")

        tracker.register_detections([_det(card, 100, 200)])
        shifted = tracker.register_detections([_det(card, 105, 203)])

        assert shifted == []

    def test_same_rank_suit_different_location_counts_as_new(self) -> None:
        """Multi-deck shoe: two physical 7c at different positions."""
        tracker = CardTracker()
        card = _card("7", "c")

        first = tracker.register_detections([_det(card, 10, 10)])
        second = tracker.register_detections([_det(card, 300, 10)])

        assert len(first) == 1
        assert len(second) == 1

    def test_different_cards_both_new(self) -> None:
        tracker = CardTracker()
        seven = _card("7", "c")
        ace = _card("A", "h")
        detections = [_det(seven, 10, 10), _det(ace, 60, 10)]

        new = tracker.register_detections(detections)

        assert len(new) == 2

    def test_mixed_new_and_repeat_in_one_frame(self) -> None:
        tracker = CardTracker()
        known = _card("9", "d")
        fresh = _card("2", "h")

        tracker.register_detections([_det(known, 20, 80)])
        new = tracker.register_detections([
            _det(known, 20, 80),
            _det(fresh, 80, 80),
        ])

        assert len(new) == 1
        assert new[0][0] == fresh

    def test_reset_clears_tracked_cards(self) -> None:
        tracker = CardTracker()
        card = _card("Q", "d")
        detections = [_det(card, 0, 0)]

        tracker.register_detections(detections)
        tracker.reset()
        after_reset = tracker.register_detections(detections)

        assert len(after_reset) == 1

    def test_ten_frame_sequence_single_card(self) -> None:
        """Acceptance: same card in 10 frames only counted once."""
        tracker = CardTracker()
        card = _card("J", "c")
        detections = [_det(card, 40, 120)]

        total_new = 0
        for _ in range(10):
            total_new += len(tracker.register_detections(detections))

        assert total_new == 1


class TestCardTrackerProximity:
    def test_iou_overlap_matches(self) -> None:
        tracker = CardTracker(iou_threshold=0.3)
        card = _card("4", "s")

        tracker.register_detections([_det(card, 0, 0, 50, 70)])
        # Partial overlap with first bbox
        repeat = tracker.register_detections([_det(card, 25, 0, 50, 70)])

        assert repeat == []

    def test_center_distance_within_threshold_matches(self) -> None:
        tracker = CardTracker(iou_threshold=0.99, center_distance_threshold=30.0)
        card = _card("5", "h")

        tracker.register_detections([_det(card, 0, 0, 40, 60)])
        # No IoU overlap but centers are 20px apart
        repeat = tracker.register_detections([_det(card, 20, 20, 40, 60)])

        assert repeat == []

    def test_far_apart_same_class_not_matched(self) -> None:
        tracker = CardTracker(center_distance_threshold=40.0)
        card = _card("8", "d")

        tracker.register_detections([_det(card, 0, 0)])
        distant = tracker.register_detections([_det(card, 200, 0)])

        assert len(distant) == 1