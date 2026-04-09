"""Tests for DetectionMeta dataclass."""

import pytest

from riverrater.game.state import Card, DetectionMeta, Rank, Suit


def _make_card(rank_val: str, suit_val: str) -> Card:
    return Card(rank=Rank(rank_val), suit=Suit(suit_val))


def test_from_detections_basic():
    """Two detections produce correct card_confidences and overall_confidence."""
    card_ah = _make_card("A", "h")
    card_kd = _make_card("K", "d")
    detections = [
        (card_ah, (0, 0, 50, 50), 0.94),
        (card_kd, (60, 0, 50, 50), 0.88),
    ]
    meta = DetectionMeta.from_detections(detections)
    assert meta.card_confidences == {"Ah": 0.94, "Kd": 0.88}
    assert meta.overall_confidence == pytest.approx((0.94 + 0.88) / 2)
    assert meta.is_manual is False


def test_from_detections_empty():
    """Empty detection list returns defaults."""
    meta = DetectionMeta.from_detections([])
    assert meta.card_confidences == {}
    assert meta.overall_confidence == 0.0
    assert meta.is_manual is False


def test_from_detections_single():
    """Single detection: overall equals that confidence."""
    card = _make_card("T", "s")
    detections = [(card, (10, 10, 40, 60), 0.77)]
    meta = DetectionMeta.from_detections(detections)
    assert meta.card_confidences == {"Ts": 0.77}
    assert meta.overall_confidence == pytest.approx(0.77)


def test_manual_factory():
    """manual() returns is_manual=True, overall=1.0, empty confidences."""
    meta = DetectionMeta.manual()
    assert meta.is_manual is True
    assert meta.overall_confidence == 1.0
    assert meta.card_confidences == {}


def test_overall_is_average():
    """Three detections — overall is the arithmetic mean of their confidences."""
    cards = [
        (_make_card("2", "c"), (0, 0, 1, 1), 0.60),
        (_make_card("5", "h"), (0, 0, 1, 1), 0.80),
        (_make_card("J", "d"), (0, 0, 1, 1), 0.90),
    ]
    meta = DetectionMeta.from_detections(cards)
    expected = (0.60 + 0.80 + 0.90) / 3
    assert meta.overall_confidence == pytest.approx(expected)
