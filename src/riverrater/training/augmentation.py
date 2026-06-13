"""
Card image augmentation for YOLO training (rotation, brightness, blur,
perspective warp, casino-felt backgrounds).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from riverrater.training.labels import (
    YoloLabel,
    box_corners,
    boxes_from_corner_sets,
    labels_to_pixel_boxes,
    parse_label_file,
    pixel_boxes_to_labels,
    transform_box_corners,
    write_label_file,
)
from riverrater.training.paths import discover_image_label_pairs


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class AugmentStats:
    """Summary of a dataset augmentation run."""

    source_images: int
    variants_per_image: int
    include_original: bool
    output_images: int
    output_labels: int
    skipped_empty_labels: int


class CardAugmentor:
    """
    Apply live-dealer-style augmentations while preserving YOLO label integrity.

    All geometric transforms update bounding boxes by warping corner points and
    recomputing axis-aligned envelopes.  Labels that collapse below
    ``min_area_ratio`` are dropped.
    """

    def __init__(
        self,
        *,
        seed: int | None = 42,
        felt_backgrounds: list[np.ndarray] | None = None,
        min_area_ratio: float = 1e-4,
    ) -> None:
        self._rng = random.Random(seed)
        self._felt_backgrounds = felt_backgrounds or []
        self._min_area_ratio = min_area_ratio

    def generate_felt_background(
        self,
        width: int = 640,
        height: int = 480,
        base_color: tuple[int, int, int] = (18, 92, 48),
    ) -> np.ndarray:
        """Synthesize a green felt texture with light noise."""
        bg = np.full((height, width, 3), base_color, dtype=np.uint8)
        np_rng = np.random.default_rng(self._rng.getrandbits(31))
        noise = np_rng.integers(-18, 19, size=(height, width, 3), dtype=np.int16)
        noisy = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Subtle horizontal grain reminiscent of casino table felt.
        grain = np.zeros((height, width), dtype=np.uint8)
        for row in range(0, height, 3):
            shade = self._rng.randint(0, 12)
            grain[row : row + 1, :] = shade
        noisy[:, :, 1] = np.clip(
            noisy[:, :, 1].astype(np.int16) - grain, 0, 255
        ).astype(np.uint8)
        return noisy

    def augment_dataset(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_images_dir: Path,
        output_labels_dir: Path,
        *,
        variants_per_image: int = 3,
        include_original: bool = True,
    ) -> AugmentStats:
        """
        Augment every labelled image in *images_dir* and write to output dirs.

        Output count is ``source * (variants_per_image + (1 if include_original))``
        when every source image has at least one label.
        """
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        pairs = discover_image_label_pairs(images_dir, labels_dir)
        output_images = 0
        output_labels = 0
        skipped_empty = 0

        for image_path, label_path in pairs:
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            labels = parse_label_file(label_path)
            if not labels:
                skipped_empty += 1
                continue

            stem = image_path.stem
            ext = image_path.suffix.lower() or ".jpg"

            if include_original:
                out_img = output_images_dir / f"{stem}_orig{ext}"
                out_lbl = output_labels_dir / f"{stem}_orig.txt"
                cv2.imwrite(str(out_img), image)
                write_label_file(out_lbl, labels)
                output_images += 1
                output_labels += 1

            for variant in range(variants_per_image):
                aug_image, aug_labels = self.augment_single(image, labels, variant)
                if not aug_labels:
                    continue
                out_img = output_images_dir / f"{stem}_aug{variant:02d}{ext}"
                out_lbl = output_labels_dir / f"{stem}_aug{variant:02d}.txt"
                cv2.imwrite(str(out_img), aug_image)
                write_label_file(out_lbl, aug_labels)
                output_images += 1
                output_labels += 1

        return AugmentStats(
            source_images=len(pairs),
            variants_per_image=variants_per_image,
            include_original=include_original,
            output_images=output_images,
            output_labels=output_labels,
            skipped_empty_labels=skipped_empty,
        )

    def augment_single(
        self,
        image: np.ndarray,
        labels: list[YoloLabel],
        variant: int,
    ) -> tuple[np.ndarray, list[YoloLabel]]:
        """Apply a deterministic augmentation pipeline for *variant*."""
        pipeline = _pipeline_for_variant(variant)
        current_image = image
        current_labels = labels

        for step in pipeline:
            if step == "rotation":
                current_image, current_labels = self._apply_rotation(
                    current_image, current_labels
                )
            elif step == "brightness":
                current_image = self._apply_brightness(current_image)
            elif step == "blur":
                current_image = self._apply_blur(current_image)
            elif step == "perspective":
                current_image, current_labels = self._apply_perspective(
                    current_image, current_labels
                )
            elif step == "felt_background":
                current_image, current_labels = self._apply_felt_background(
                    current_image, current_labels
                )

        return current_image, current_labels

    def _apply_rotation(
        self,
        image: np.ndarray,
        labels: list[YoloLabel],
    ) -> tuple[np.ndarray, list[YoloLabel]]:
        height, width = image.shape[:2]
        angle = self._rng.uniform(-25.0, 25.0)
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        new_labels = self._transform_labels(matrix, labels, width, height)
        return rotated, new_labels

    def _apply_brightness(self, image: np.ndarray) -> np.ndarray:
        factor = self._rng.uniform(0.55, 1.45)
        adjusted = np.clip(image.astype(np.float32) * factor, 0, 255)
        return adjusted.astype(np.uint8)

    def _apply_blur(self, image: np.ndarray) -> np.ndarray:
        kernel = self._rng.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel, kernel), 0)

    def _apply_perspective(
        self,
        image: np.ndarray,
        labels: list[YoloLabel],
    ) -> tuple[np.ndarray, list[YoloLabel]]:
        height, width = image.shape[:2]
        margin_x = width * 0.08
        margin_y = height * 0.08

        src = np.float32(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ]
        )
        dst = np.float32(
            [
                [self._rng.uniform(0, margin_x), self._rng.uniform(0, margin_y)],
                [
                    width - 1 - self._rng.uniform(0, margin_x),
                    self._rng.uniform(0, margin_y),
                ],
                [
                    width - 1 - self._rng.uniform(0, margin_x),
                    height - 1 - self._rng.uniform(0, margin_y),
                ],
                [
                    self._rng.uniform(0, margin_x),
                    height - 1 - self._rng.uniform(0, margin_y),
                ],
            ]
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        new_labels = self._transform_labels(matrix, labels, width, height)
        return warped, new_labels

    def _apply_felt_background(
        self,
        image: np.ndarray,
        labels: list[YoloLabel],
    ) -> tuple[np.ndarray, list[YoloLabel]]:
        src_h, src_w = image.shape[:2]
        felt_w = max(src_w, 480)
        felt_h = max(src_h, 360)

        if self._felt_backgrounds:
            felt = self._rng.choice(self._felt_backgrounds).copy()
            felt = cv2.resize(felt, (felt_w, felt_h))
        else:
            felt = self.generate_felt_background(felt_w, felt_h)

        scale = self._rng.uniform(0.45, 0.85)
        new_w = max(32, int(src_w * scale))
        new_h = max(32, int(src_h * scale))
        resized = cv2.resize(image, (new_w, new_h))

        max_x = felt_w - new_w
        max_y = felt_h - new_h
        offset_x = self._rng.randint(0, max_x) if max_x > 0 else 0
        offset_y = self._rng.randint(0, max_y) if max_y > 0 else 0

        felt[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

        pixel_boxes = labels_to_pixel_boxes(labels, src_w, src_h)
        scaled_boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        for class_id, (x1, y1, x2, y2) in pixel_boxes:
            scaled_boxes.append(
                (
                    class_id,
                    (
                        x1 * scale + offset_x,
                        y1 * scale + offset_y,
                        x2 * scale + offset_x,
                        y2 * scale + offset_y,
                    ),
                )
            )

        new_labels = pixel_boxes_to_labels(
            scaled_boxes,
            felt_w,
            felt_h,
            min_area_ratio=self._min_area_ratio,
        )
        return felt, new_labels

    def _transform_labels(
        self,
        matrix: np.ndarray,
        labels: list[YoloLabel],
        width: int,
        height: int,
    ) -> list[YoloLabel]:
        pixel_boxes = labels_to_pixel_boxes(labels, width, height)
        corner_sets: list[tuple[int, list[tuple[float, float]]]] = []

        for class_id, (x1, y1, x2, y2) in pixel_boxes:
            corners = box_corners(x1, y1, x2, y2)
            warped = transform_box_corners(corners, matrix)
            corner_sets.append((class_id, warped))

        new_boxes = boxes_from_corner_sets(corner_sets)
        return pixel_boxes_to_labels(
            new_boxes,
            width,
            height,
            min_area_ratio=self._min_area_ratio,
        )


def _pipeline_for_variant(variant: int) -> list[str]:
    """Deterministic augmentation recipe keyed by variant index."""
    pipelines = [
        ["rotation"],
        ["brightness", "blur"],
        ["perspective"],
        ["felt_background"],
        ["rotation", "brightness"],
        ["perspective", "felt_background", "blur"],
    ]
    return pipelines[variant % len(pipelines)]


def build_class_names() -> list[str]:
    """
    Build the 52-class name list matching ``YOLOEngine.CLASS_MAP`` ordering.

    Order: rank (2→A) × suit (c, d, h, s).
    """
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["c", "d", "h", "s"]
    return [f"{rank}{suit}" for rank in ranks for suit in suits]