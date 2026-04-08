"""
Tests for TemplateEngine (src/riverrater/vision/template_engine.py).

Covers:
- add_template / remove_template
- detect_cards with a synthetic image
- NMS (overlapping duplicates → single detection)
- save_profile / load_profile round-trip
- Multi-scale detection (template size differs from in-frame card size)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# The imports below work whether or not riverrater is installed as a package;
# pytest's conftest / PYTHONPATH handling makes them reachable from the tests/
# directory when the project is installed with `pip install -e .`.
# ---------------------------------------------------------------------------
from riverrater.vision.template_engine import TemplateEngine, _compute_iou, _nms

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

try:
    from riverrater.game.state import Card, Rank, Suit
except ImportError:
    # Fallback definitions (mirrors game/state.py).
    from enum import Enum
    from dataclasses import dataclass

    class Suit(Enum):
        HEARTS = "h"
        DIAMONDS = "d"
        CLUBS = "c"
        SPADES = "s"

    class Rank(Enum):
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

    @dataclass
    class Card:
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


def _make_card_image(color: tuple[int, int, int], size: tuple[int, int] = (40, 60)) -> np.ndarray:
    """
    Return a structured BGR image that acts as a synthetic card template.

    Rather than a flat solid colour (which can produce false matches at all
    positions when resized to small sizes), the image has a bright border and
    a contrasting inner region so it is visually distinctive at the pixel level.
    """
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Fill with the main colour.
    img[:, :] = color
    # Add a contrasting 2-px white border so the template has clear edge structure.
    img[:2, :] = (255, 255, 255)
    img[-2:, :] = (255, 255, 255)
    img[:, :2] = (255, 255, 255)
    img[:, -2:] = (255, 255, 255)
    # Add a dark cross in the centre for further distinctiveness.
    cy, cx = h // 2, w // 2
    img[cy - 1 : cy + 2, :] = (10, 10, 10)
    img[:, cx - 1 : cx + 2] = (10, 10, 10)
    return img


def _make_scene(
    card_img: np.ndarray,
    pos: tuple[int, int],
    scene_size: tuple[int, int] = (480, 640),
) -> np.ndarray:
    """
    Place *card_img* at *pos* (y, x) inside a grey scene of *scene_size*.
    Returns the scene as a BGR array.
    """
    # Use a mid-grey that differs strongly from the template's white border.
    scene = np.full((scene_size[0], scene_size[1], 3), 128, dtype=np.uint8)
    y, x = pos
    h, w = card_img.shape[:2]
    scene[y : y + h, x : x + w] = card_img
    return scene


@pytest.fixture
def ace_of_hearts() -> Card:
    return Card(rank=Rank.ACE, suit=Suit.HEARTS)


@pytest.fixture
def king_of_spades() -> Card:
    return Card(rank=Rank.KING, suit=Suit.SPADES)


@pytest.fixture
def card_image() -> np.ndarray:
    """A distinct cyan-ish solid-colour rectangle used as a card template."""
    return _make_card_image((0, 200, 255), size=(40, 60))


# ---------------------------------------------------------------------------
# Tests: add_template / remove_template
# ---------------------------------------------------------------------------


class TestAddRemoveTemplate:
    def test_add_single_template(self, ace_of_hearts: Card, card_image: np.ndarray) -> None:
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, card_image)
        assert ace_of_hearts in engine._templates
        assert len(engine._templates[ace_of_hearts]) == 1

    def test_add_multiple_templates_same_card(
        self, ace_of_hearts: Card, card_image: np.ndarray
    ) -> None:
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, card_image)
        engine.add_template(ace_of_hearts, card_image)
        assert len(engine._templates[ace_of_hearts]) == 2

    def test_add_templates_different_cards(
        self,
        ace_of_hearts: Card,
        king_of_spades: Card,
        card_image: np.ndarray,
    ) -> None:
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, card_image)
        engine.add_template(king_of_spades, card_image)
        assert len(engine._templates) == 2

    def test_remove_template(self, ace_of_hearts: Card, card_image: np.ndarray) -> None:
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, card_image)
        engine.remove_template(ace_of_hearts)
        assert ace_of_hearts not in engine._templates

    def test_remove_nonexistent_card_no_error(self, ace_of_hearts: Card) -> None:
        engine = TemplateEngine()
        # Should not raise.
        engine.remove_template(ace_of_hearts)

    def test_add_bgr_image_stored_as_gray(
        self, ace_of_hearts: Card, card_image: np.ndarray
    ) -> None:
        """Templates should be stored in grayscale regardless of input format."""
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, card_image)
        stored = engine._templates[ace_of_hearts][0]
        assert stored.ndim == 2, "Stored template should be grayscale (2-D)."


# ---------------------------------------------------------------------------
# Tests: detect_cards with a synthetic image
# ---------------------------------------------------------------------------


class TestDetectCards:
    """
    Image-based detect_cards tests use small scenes (80×100 px) and small
    templates to keep runtime acceptable on constrained CI hardware.
    """

    def _make_small_scene_with_card(
        self,
        card_img: np.ndarray,
        pos: tuple[int, int] = (20, 25),
        scene_h: int = 80,
        scene_w: int = 100,
        bg_value: int = 128,
    ) -> np.ndarray:
        scene = np.full((scene_h, scene_w, 3), bg_value, dtype=np.uint8)
        y, x = pos
        h, w = card_img.shape[:2]
        scene[y : y + h, x : x + w] = card_img
        return scene

    def test_detects_card_in_scene(
        self, ace_of_hearts: Card
    ) -> None:
        engine = TemplateEngine()
        # Small structured template (15×20) on a 80×100 scene.
        card_img = _make_card_image((0, 180, 255), size=(15, 20))
        engine.add_template(ace_of_hearts, card_img)

        scene = self._make_small_scene_with_card(card_img, pos=(20, 25))
        detections = engine.detect_cards(scene, confidence=0.90)

        cards_found = [d[0] for d in detections]
        assert ace_of_hearts in cards_found, (
            f"Expected {ace_of_hearts} in detections; got {cards_found}"
        )

    def test_detection_bbox_approximately_correct(
        self, ace_of_hearts: Card
    ) -> None:
        engine = TemplateEngine()
        card_img = _make_card_image((0, 180, 255), size=(15, 20))
        engine.add_template(ace_of_hearts, card_img)

        target_y, target_x = 20, 25
        scene = self._make_small_scene_with_card(card_img, pos=(target_y, target_x))
        detections = engine.detect_cards(scene, confidence=0.90)

        assert detections, "No detections returned."
        # Find the best-confidence detection for our card.
        our_dets = [d for d in detections if d[0] == ace_of_hearts]
        assert our_dets, "Card not in detections."
        best = max(our_dets, key=lambda d: d[2])
        x, y, w, h = best[1]
        # Allow ±8 px tolerance for multi-scale rounding.
        assert abs(x - target_x) <= 8, f"x={x} expected ~{target_x}"
        assert abs(y - target_y) <= 8, f"y={y} expected ~{target_y}"

    def test_no_false_positive_on_empty_scene(self, ace_of_hearts: Card) -> None:
        engine = TemplateEngine()
        # Use a structured template (has white border + dark cross).
        tmpl = _make_card_image((50, 100, 200), size=(15, 20))
        engine.add_template(ace_of_hearts, tmpl)

        # Uniform scene at a very different intensity — no card present.
        grey_scene = np.full((80, 100, 3), 200, dtype=np.uint8)
        detections = engine.detect_cards(grey_scene, confidence=0.95)
        assert not any(d[0] == ace_of_hearts for d in detections), (
            "False positive: card detected in a plain scene."
        )

    def test_no_templates_returns_empty(self, ace_of_hearts: Card) -> None:
        engine = TemplateEngine()
        scene = np.full((80, 100, 3), 128, dtype=np.uint8)
        assert engine.detect_cards(scene) == []

    def test_confidence_score_in_range(
        self, ace_of_hearts: Card
    ) -> None:
        engine = TemplateEngine()
        card_img = _make_card_image((0, 180, 255), size=(15, 20))
        engine.add_template(ace_of_hearts, card_img)
        scene = self._make_small_scene_with_card(card_img, pos=(20, 25))
        detections = engine.detect_cards(scene, confidence=0.85)
        for _, _, score in detections:
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


# ---------------------------------------------------------------------------
# Tests: NMS — two overlapping detections → single kept detection
# ---------------------------------------------------------------------------


class TestNMS:
    """
    NMS tests operate on synthetic detection lists (not full image pipelines)
    to keep runtime minimal.  Image-based NMS coverage is provided by
    TestDetectCards.test_detects_card_in_scene which exercises the full path.
    """

    def test_compute_iou_perfect_overlap(self) -> None:
        box = (10, 10, 50, 50)
        assert _compute_iou(box, box) == pytest.approx(1.0)

    def test_compute_iou_no_overlap(self) -> None:
        a = (0, 0, 10, 10)
        b = (20, 20, 10, 10)
        assert _compute_iou(a, b) == pytest.approx(0.0)

    def test_compute_iou_partial_overlap(self) -> None:
        # Two 10×10 boxes offset by 5 px in both axes → 5×5=25 intersection,
        # union = 100+100-25 = 175, IoU = 25/175 ≈ 0.1429.
        a = (0, 0, 10, 10)
        b = (5, 5, 10, 10)
        iou = _compute_iou(a, b)
        assert abs(iou - 25 / 175) < 1e-4

    def test_nms_keeps_highest_confidence(self) -> None:
        card_a = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        card_b = Card(rank=Rank.KING, suit=Suit.SPADES)
        # Two boxes that overlap heavily; second has higher confidence.
        dets = [
            (card_a, (0, 0, 50, 50), 0.80),
            (card_b, (5, 5, 50, 50), 0.95),
        ]
        kept = _nms(dets, iou_threshold=0.3)
        assert len(kept) == 1
        # The one kept should have the higher confidence score.
        assert kept[0][2] == pytest.approx(0.95)

    def test_nms_empty_input(self) -> None:
        assert _nms([]) == []

    def test_nms_single_detection_kept(self) -> None:
        card = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        dets = [(card, (0, 0, 50, 50), 0.90)]
        assert _nms(dets) == dets

    def test_nms_non_overlapping_all_kept(self) -> None:
        card_a = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        card_b = Card(rank=Rank.KING, suit=Suit.SPADES)
        dets = [
            (card_a, (0, 0, 50, 50), 0.90),
            (card_b, (200, 200, 50, 50), 0.85),
        ]
        kept = _nms(dets, iou_threshold=0.3)
        assert len(kept) == 2

    def test_nms_overlapping_collapsed_to_one(self) -> None:
        card_a = Card(rank=Rank.ACE, suit=Suit.HEARTS)
        # Identical boxes — IoU = 1.0, always suppressed.
        dets = [
            (card_a, (10, 10, 50, 50), 0.80),
            (card_a, (10, 10, 50, 50), 0.75),
        ]
        kept = _nms(dets, iou_threshold=0.3)
        assert len(kept) == 1
        assert kept[0][2] == pytest.approx(0.80)  # higher-conf box kept


# ---------------------------------------------------------------------------
# Tests: save_profile / load_profile round-trip
# ---------------------------------------------------------------------------


class TestProfileRoundTrip:
    def test_save_and_reload_templates(
        self, ace_of_hearts: Card, card_image: np.ndarray
    ) -> None:
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, card_image)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.save_profile(tmpdir)

            engine2 = TemplateEngine()
            engine2.load_profile(tmpdir)

        assert ace_of_hearts in engine2._templates
        assert len(engine2._templates[ace_of_hearts]) == 1

    def test_roundtrip_preserves_template_shape(
        self, ace_of_hearts: Card, card_image: np.ndarray
    ) -> None:
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, card_image)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.save_profile(tmpdir)
            engine2 = TemplateEngine()
            engine2.load_profile(tmpdir)

        # Templates are stored grayscale.
        expected_shape = (card_image.shape[0], card_image.shape[1])
        assert engine2._templates[ace_of_hearts][0].shape == expected_shape

    def test_roundtrip_multiple_cards(
        self,
        ace_of_hearts: Card,
        king_of_spades: Card,
        card_image: np.ndarray,
    ) -> None:
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, card_image)
        engine.add_template(king_of_spades, _make_card_image((100, 100, 100)))

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.save_profile(tmpdir)
            engine2 = TemplateEngine()
            engine2.load_profile(tmpdir)

        assert len(engine2._templates) == 2

    def test_roundtrip_multiple_templates_per_card(
        self, ace_of_hearts: Card, card_image: np.ndarray
    ) -> None:
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, card_image)
        engine.add_template(ace_of_hearts, _make_card_image((10, 20, 30)))

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.save_profile(tmpdir)
            engine2 = TemplateEngine()
            engine2.load_profile(tmpdir)

        assert len(engine2._templates[ace_of_hearts]) == 2

    def test_load_missing_profile_raises(self) -> None:
        engine = TemplateEngine()
        with pytest.raises(FileNotFoundError):
            engine.load_profile("/nonexistent/path/to/profile")

    def test_profile_creates_required_files(
        self, ace_of_hearts: Card, card_image: np.ndarray
    ) -> None:
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, card_image)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.save_profile(tmpdir)
            assert (Path(tmpdir) / "metadata.json").exists()
            assert (Path(tmpdir) / "card_templates.npz").exists()

    def test_profile_path_constructor(
        self, ace_of_hearts: Card, card_image: np.ndarray
    ) -> None:
        """TemplateEngine(profile_path=...) should auto-load the profile."""
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, card_image)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.save_profile(tmpdir)
            engine2 = TemplateEngine(profile_path=tmpdir)

        assert ace_of_hearts in engine2._templates


# ---------------------------------------------------------------------------
# Tests: multi-scale detection
# ---------------------------------------------------------------------------


class TestMultiScaleDetection:
    """
    Multi-scale tests use small images (≤80×100) to keep execution time low.
    The structured template (white border + dark cross) ensures the NCC
    response is sufficiently distinct to survive the 0.80 confidence threshold.
    """

    def test_detects_card_at_larger_scale(self, ace_of_hearts: Card) -> None:
        """
        Register a small template; the card should be found even when it
        appears at a larger size in the frame (scale > 1.0).
        """
        # Small structured template (10×14).
        small_tmpl = _make_card_image((0, 100, 200), size=(10, 14))
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, small_tmpl)

        # Large version of the same structured image in a 70×90 scene.
        large_card = _make_card_image((0, 100, 200), size=(15, 21))
        scene = np.full((70, 90, 3), 128, dtype=np.uint8)
        scene[20:35, 30:51] = large_card

        detections = engine.detect_cards(scene, confidence=0.80)
        assert any(d[0] == ace_of_hearts for d in detections), (
            "Multi-scale detection failed to find card at larger scale."
        )

    def test_detects_card_at_smaller_scale(self, ace_of_hearts: Card) -> None:
        """
        Register a large template; the card should be found when it appears
        at a smaller size in the frame (scale < 1.0).
        """
        # Larger structured template (16×22).
        large_tmpl = _make_card_image((200, 50, 0), size=(16, 22))
        engine = TemplateEngine()
        engine.add_template(ace_of_hearts, large_tmpl)

        # Downsized version (≈0.7× → 11×15) embedded in a 70×90 scene.
        small_card = cv2.resize(large_tmpl, (15, 11))
        scene = np.full((70, 90, 3), 128, dtype=np.uint8)
        scene[25:36, 35:50] = small_card

        detections = engine.detect_cards(scene, confidence=0.80)
        assert any(d[0] == ace_of_hearts for d in detections), (
            "Multi-scale detection failed to find card at smaller scale."
        )
