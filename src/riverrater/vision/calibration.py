"""
User-driven calibration system for RiverRater.

Provides two classes:

* :class:`CalibrationCapture` — low-level helpers for extracting ROI patches
  from screen frames and saving them as templates.
* :class:`CalibrationSession` — higher-level session that accumulates
  (card_str, bbox, frame) tuples and commits them to a
  :class:`~riverrater.vision.template_engine.TemplateEngine` in one shot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from riverrater.game.state import Card, Rank, Suit


logger = logging.getLogger(__name__)


class CalibrationCapture:
    """
    Low-level helpers for extracting and persisting card template ROIs.

    This class is stateless — every method operates on the arguments passed
    to it.  It is designed to be used as a utility by :class:`CalibrationSession`
    or directly from GUI code that already manages state.
    """

    def start_capture(self, frame: np.ndarray) -> np.ndarray:
        """
        Accept a screen frame to be used as the source for ROI selection.

        In a full GUI implementation this would open an overlay that lets the
        user draw a rectangle.  In this headless form it simply validates and
        returns the frame so callers can pass it to :meth:`get_roi`.

        Parameters
        ----------
        frame:
            Current screen capture as a BGR ``uint8`` numpy array.

        Returns
        -------
        np.ndarray
            The same frame (possibly a copy if preprocessing were applied).

        Raises
        ------
        ValueError
            If *frame* is empty or None.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame must be a non-empty numpy array.")
        return frame.copy()

    def get_roi(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> np.ndarray:
        """
        Extract a region of interest from *frame*.

        Parameters
        ----------
        frame:
            Source BGR image.
        x, y:
            Top-left corner of the ROI in frame coordinates.
        w, h:
            Width and height of the ROI.

        Returns
        -------
        np.ndarray
            Cropped BGR sub-image.

        Raises
        ------
        ValueError
            If the requested region is outside *frame* or has zero area.
        """
        if w <= 0 or h <= 0:
            raise ValueError(f"ROI dimensions must be positive (got w={w}, h={h}).")

        frame_h, frame_w = frame.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"ROI ({x}, {y}, {w}, {h}) does not intersect frame of size "
                f"({frame_w}×{frame_h})."
            )

        roi = frame[y1:y2, x1:x2].copy()
        return roi

    def save_template(
        self,
        card: Card,
        roi: np.ndarray,
        profile_path: str,
    ) -> None:
        """
        Save *roi* as a new template for *card* inside *profile_path*.

        This writes the template directly via :meth:`TemplateEngine.save_profile`
        by loading the existing profile (if any), appending the new template,
        and re-saving.  This keeps the profile on disk in sync with each
        new capture.

        Parameters
        ----------
        card:
            The card this ROI represents.
        roi:
            Cropped BGR image of the card.
        profile_path:
            Path to the profile directory.
        """
        # Lazy import to avoid circular dependencies at module level.
        from riverrater.vision.template_engine import TemplateEngine  # noqa: PLC0415

        engine = TemplateEngine()
        profile_dir = Path(profile_path)

        # Load existing templates if the profile already exists.
        if profile_dir.exists() and (profile_dir / "metadata.json").exists():
            try:
                engine.load_profile(profile_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Could not load existing profile at %s: %s. Starting fresh.",
                    profile_path,
                    exc,
                )

        engine.add_template(card, roi)
        engine.save_profile(profile_path)
        logger.info("Saved template for %s to profile %s.", card, profile_path)


class CalibrationSession:
    """
    High-level calibration flow manager.

    Accumulates user-supplied (card_str, bbox, frame) entries during an
    interactive session, then commits them all to a :class:`TemplateEngine`
    via :meth:`finish`.  Calling :meth:`cancel` discards everything without
    modifying the engine.

    Example
    -------
    >>> session = CalibrationSession()
    >>> session.add_calibration("Ah", (100, 200, 60, 90), screen_frame)
    >>> session.add_calibration("Kd", (300, 200, 60, 90), screen_frame)
    >>> session.finish(template_engine)
    """

    def __init__(self) -> None:
        # Each entry: (Card, roi image, bbox in frame coordinates)
        self._pending: list[tuple[Card, np.ndarray, tuple[int, int, int, int]]] = []
        self._capture = CalibrationCapture()
        self._active = True

    # ------------------------------------------------------------------ #
    # Data accumulation
    # ------------------------------------------------------------------ #

    def add_calibration(
        self,
        card_str: str,
        bbox: tuple[int, int, int, int],
        frame: np.ndarray,
    ) -> None:
        """
        Record a calibration sample.

        Parameters
        ----------
        card_str:
            Two-character card identifier, e.g. ``"Ah"`` (Ace of hearts),
            ``"Td"`` (Ten of diamonds), ``"2c"`` (Two of clubs).
        bbox:
            ``(x, y, w, h)`` bounding box of the card within *frame*.
        frame:
            Full screen capture frame in which the card appears.

        Raises
        ------
        RuntimeError
            If the session has already been finalised or cancelled.
        ValueError
            If *card_str* cannot be parsed or *bbox* is invalid.
        """
        if not self._active:
            raise RuntimeError("CalibrationSession is no longer active.")

        card = self.parse_card_string(card_str)
        x, y, w, h = bbox
        roi = self._capture.get_roi(frame, x, y, w, h)
        self._pending.append((card, roi, (x, y, w, h)))
        logger.debug("Queued calibration for %s (%d pending).", card, len(self._pending))

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_card_string(card_str: str) -> Card:
        """
        Parse a short card notation string into a :class:`Card` object.

        Supported formats
        -----------------
        * Rank character (case-insensitive): ``2 3 4 5 6 7 8 9 T J Q K A``
        * Suit character (lowercase): ``h d c s``

        Examples
        --------
        * ``"Ah"`` → Ace of hearts
        * ``"Td"`` → Ten of diamonds
        * ``"2c"`` → Two of clubs
        * ``"Ks"`` → King of spades

        Parameters
        ----------
        card_str:
            Exactly two characters: rank + suit.

        Returns
        -------
        Card

        Raises
        ------
        ValueError
            If *card_str* is not exactly 2 characters or contains an
            unrecognised rank or suit.
        """
        if not isinstance(card_str, str) or len(card_str) != 2:
            raise ValueError(
                f"card_str must be a 2-character string (e.g. 'Ah'), got: {card_str!r}"
            )

        rank_char = card_str[0].upper()
        suit_char = card_str[1].lower()

        try:
            rank = Rank(rank_char)
        except ValueError:
            valid_ranks = [r.value for r in Rank]
            raise ValueError(
                f"Unknown rank character {rank_char!r}. "
                f"Valid options: {valid_ranks}"
            )

        try:
            suit = Suit(suit_char)
        except ValueError:
            valid_suits = [s.value for s in Suit]
            raise ValueError(
                f"Unknown suit character {suit_char!r}. "
                f"Valid options: {valid_suits}"
            )

        return Card(rank=rank, suit=suit)

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #

    def finish(self, template_engine: object) -> None:
        """
        Apply all accumulated calibrations to *template_engine* and close the
        session.

        Parameters
        ----------
        template_engine:
            A :class:`~riverrater.vision.template_engine.TemplateEngine`
            instance (typed as ``object`` here to avoid a circular import at
            class definition time).

        Raises
        ------
        RuntimeError
            If the session has already been finalised or cancelled.
        AttributeError
            If *template_engine* does not expose an ``add_template`` method.
        """
        if not self._active:
            raise RuntimeError("CalibrationSession is no longer active.")

        if not hasattr(template_engine, "add_template"):
            raise AttributeError(
                f"template_engine must have an add_template() method, "
                f"got: {type(template_engine)!r}"
            )

        slot_regions: list[tuple[int, int, int, int]] = []
        seen_regions: set[tuple[int, int, int, int]] = set()
        for card, roi, bbox in self._pending:
            template_engine.add_template(card, roi)
            if bbox not in seen_regions:
                seen_regions.add(bbox)
                slot_regions.append(bbox)
            logger.info("Applied calibration for %s to engine.", card)

        if slot_regions and hasattr(template_engine, "set_roi_regions"):
            template_engine.set_roi_regions(slot_regions)

        count = len(self._pending)
        self._pending.clear()
        self._active = False
        logger.info("CalibrationSession finished — applied %d template(s).", count)

    def cancel(self) -> None:
        """
        Discard all accumulated calibration data without modifying the engine.

        The session is marked inactive; subsequent calls to :meth:`add_calibration`
        or :meth:`finish` will raise :exc:`RuntimeError`.
        """
        self._pending.clear()
        self._active = False
        logger.info("CalibrationSession cancelled — all pending data discarded.")
