"""
YOLO-based card detection engine stub for RiverRater.

This module provides :class:`YOLOEngine`, which implements the same
``detect_cards`` interface as :class:`~riverrater.vision.template_engine.TemplateEngine`.

Current status
--------------
The YOLO model has not yet been trained.  This stub:

* Attempts to load a model file at ``model_path`` if one is provided.
* Returns an empty list from ``detect_cards`` when no model is available.
* Never raises an exception from ``detect_cards`` — callers should check
  :attr:`is_available` first and fall back to template matching or manual
  input when it is ``False``.

Integration path (fill in when a trained model is available)
-------------------------------------------------------------
1. **Train the model**

   .. code-block:: bash

       yolo detect train model=yolov8n.pt data=cards.yaml epochs=100

   Use a labelled dataset of ~1 000 – 5 000 card images per class.  The
   Roboflow ``playing-cards`` dataset is a good starting point.

2. **Export the weights**

   Copy ``runs/detect/train/weights/best.pt`` to
   ``models/yolov8n_cards/best.pt`` (relative to the project root) and pass
   that path to ``YOLOEngine.__init__``.

3. **Update CLASS_MAP**

   YOLO assigns integer class IDs in the order they appear in ``data.yaml``.
   Update :attr:`CLASS_MAP` to match that exact ordering.  The placeholder
   mapping below assumes alphabetical order (2c, 2d, 2h, 2s, 3c, …, As).

4. **Install optional [ml] dependencies**

   .. code-block:: bash

       pip install "riverrater[ml]"

   This installs ``torch`` and ``ultralytics``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from riverrater.game.state import Card, Rank, Suit


logger = logging.getLogger(__name__)


def _build_class_map() -> dict[int, tuple[Rank, Suit]]:
    """
    Build the placeholder CLASS_MAP.

    Cards are ordered alphabetically by their two-character string (rank then
    suit): 2c, 2d, 2h, 2s, 3c, …, Ac, Ad, Ah, As → class IDs 0–51.

    This ordering **must** match the class ordering used during YOLO training.
    Update this function (or override :attr:`YOLOEngine.CLASS_MAP`) once the
    actual training class order is known.
    """
    rank_order = [
        Rank.TWO,
        Rank.THREE,
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.EIGHT,
        Rank.NINE,
        Rank.TEN,
        Rank.JACK,
        Rank.QUEEN,
        Rank.KING,
        Rank.ACE,
    ]
    suit_order = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]

    class_map: dict[int, tuple[Rank, Suit]] = {}
    idx = 0
    for rank in rank_order:
        for suit in suit_order:
            class_map[idx] = (rank, suit)
            idx += 1
    return class_map


class YOLOEngine:
    """
    YOLOv8-based card detection engine (stub).

    Implements the same ``detect_cards`` interface as
    :class:`~riverrater.vision.template_engine.TemplateEngine` so the two
    engines can be used interchangeably.

    Parameters
    ----------
    model_path:
        Path to a trained ``best.pt`` weights file.  If ``None`` or the file
        does not exist, the engine operates in stub mode and returns empty
        detections.

    Attributes
    ----------
    CLASS_MAP:
        Dict mapping YOLO class integer IDs to ``(Rank, Suit)`` tuples.
        Populated with a placeholder alphabetical ordering; update to match
        the actual training class order before deployment.
    """

    #: Placeholder class map: YOLO class ID → (Rank, Suit).
    #: Update indices to match training class order before using a real model.
    CLASS_MAP: dict[int, tuple[Rank, Suit]] = _build_class_map()

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model = None  # Will be ultralytics.YOLO when model is available.
        self._model_path: Optional[str] = model_path

        if model_path is not None:
            self._try_load_model(model_path)

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _try_load_model(self, model_path: str) -> None:
        """
        Attempt to load a YOLO model from *model_path*.

        Failures are logged but never raised so the application can start
        without ML dependencies installed.

        Parameters
        ----------
        model_path:
            Filesystem path to a ``*.pt`` weights file.
        """
        import os

        if not os.path.isfile(model_path):
            logger.warning(
                "YOLO weights not found at %s. Running in stub mode.", model_path
            )
            return

        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]

            self._model = YOLO(model_path)
            logger.info("Loaded YOLO model from %s.", model_path)
        except ImportError:
            logger.warning(
                "ultralytics is not installed.  Install it with "
                '``pip install "riverrater[ml]"`` to enable YOLO detection.'
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to load YOLO model from %s: %s", model_path, exc)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def is_available(self) -> bool:
        """
        ``True`` if a YOLO model is loaded and ready for inference.

        Returns
        -------
        bool
        """
        return self._model is not None

    def detect_cards(
        self,
        frame: np.ndarray,
        confidence: float = 0.5,
    ) -> list[tuple[Card, tuple[int, int, int, int], float]]:
        """
        Detect cards in *frame* using the loaded YOLO model.

        When no model is available this returns an empty list rather than
        raising — callers should check :attr:`is_available` and fall back to
        template matching or manual input as appropriate.

        Parameters
        ----------
        frame:
            BGR ``uint8`` numpy array (the captured screen frame).
        confidence:
            Minimum YOLO detection confidence (0–1).  Detections below this
            threshold are discarded.

        Returns
        -------
        list of (Card, bbox, score)
            Each entry is ``(card, (x, y, w, h), confidence_score)`` — the
            same format returned by :class:`TemplateEngine.detect_cards`.

        Notes for integration
        ---------------------
        When a trained model is available:

        1. ``results = self._model(frame, conf=confidence)`` runs inference.
        2. Each ``result.boxes`` object contains ``xyxy`` coordinates, ``cls``
           class IDs, and ``conf`` scores.
        3. Map ``int(cls)`` through :attr:`CLASS_MAP` to obtain ``(Rank, Suit)``
           and construct a :class:`Card`.
        4. Convert ``xyxy`` → ``(x, y, w, h)`` for the output bbox.
        """
        if self._model is None:
            return []

        detections: list[tuple[Card, tuple[int, int, int, int], float]] = []

        try:
            results = self._model(frame, conf=confidence, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())

                    if cls_id not in self.CLASS_MAP:
                        logger.debug("Unknown class ID %d — skipping.", cls_id)
                        continue

                    rank, suit = self.CLASS_MAP[cls_id]
                    card = Card(rank=rank, suit=suit)

                    # xyxy → (x, y, w, h)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)

                    detections.append((card, (x, y, w, h), score))

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("YOLO inference error: %s", exc)

        return detections
