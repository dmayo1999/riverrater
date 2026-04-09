"""
MSS-based screen capture module for RiverRater.

Provides :class:`ScreenCapture` which can grab frames synchronously or run a
continuous capture loop in a background daemon thread.  Frames are returned as
BGR ``numpy`` arrays (same convention as OpenCV).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

import mss
import mss.tools
import numpy as np

logger = logging.getLogger(__name__)


class ScreenCapture:
    """
    Screen capture using the ``mss`` library.

    Parameters
    ----------
    region:
        ``(left, top, width, height)`` capture region in screen coordinates.
        Pass ``None`` to capture the full primary monitor.

    Example
    -------
    >>> cap = ScreenCapture(region=(0, 0, 1920, 1080))
    >>> cap.start()
    >>> frame = cap.get_latest_frame()   # BGR numpy array
    >>> cap.stop()
    """

    # Number of frame timestamps to keep for rolling FPS calculation.
    _FPS_WINDOW: int = 30

    def __init__(self, region: Optional[tuple[int, int, int, int]] = None) -> None:
        self._region: Optional[tuple[int, int, int, int]] = region

        # Frame buffer — written by background thread, read by consumers.
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # Rolling FPS tracking.
        self._timestamps: deque[float] = deque(maxlen=self._FPS_WINDOW)
        self._fps_lock = threading.Lock()

        # Background thread state.
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_region(self, region: tuple[int, int, int, int]) -> None:
        """
        Update the capture region.

        Thread-safe — may be called while the background thread is running.

        Parameters
        ----------
        region:
            ``(left, top, width, height)`` in screen pixels.
        """
        with self._frame_lock:
            self._region = region

    def grab_frame(self) -> np.ndarray:
        """
        Capture a single frame synchronously.

        Returns
        -------
        np.ndarray
            BGR image as a ``uint8`` numpy array.

        Raises
        ------
        RuntimeError
            If no monitor is found or the requested region is invalid.
        """
        return self._capture_frame()

    def start(self) -> None:
        """
        Start continuous capture in a background daemon thread.

        If capture is already running this is a no-op.
        """
        if self._running:
            logger.debug("ScreenCapture.start() called but capture already running.")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="ScreenCapture-thread",
            daemon=True,
        )
        self._thread.start()
        logger.info("ScreenCapture background thread started.")

    def stop(self) -> None:
        """
        Stop the background capture thread.

        Blocks until the thread has exited.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("ScreenCapture background thread stopped.")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Return the most recent frame from the buffer (non-blocking).

        Returns
        -------
        np.ndarray or None
            The latest BGR frame, or ``None`` if no frame has been captured
            yet.
        """
        with self._frame_lock:
            return self._latest_frame

    def get_fps(self) -> float:
        """
        Return the rolling average capture rate.

        Computed over the last :attr:`_FPS_WINDOW` frame timestamps.

        Returns
        -------
        float
            Frames per second, or ``0.0`` if fewer than 2 frames have been
            captured.
        """
        with self._fps_lock:
            if len(self._timestamps) < 2:
                return 0.0
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed <= 0.0:
                return 0.0
            return (len(self._timestamps) - 1) / elapsed

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_mss_monitor(
        self, sct: mss.base.MSSBase
    ) -> dict[str, int]:
        """
        Build the ``mss`` monitor dict from the current region.

        Parameters
        ----------
        sct:
            An active ``mss`` context.

        Returns
        -------
        dict
            ``{"left": ..., "top": ..., "width": ..., "height": ...}``

        Raises
        ------
        RuntimeError
            If no monitors are available.
        """
        if not sct.monitors:
            raise RuntimeError("mss found no monitors.")

        # monitors[0] is the "all monitors" virtual screen; monitors[1] is
        # the primary display.
        primary = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

        with self._frame_lock:
            region = self._region

        if region is None:
            return {
                "left": primary["left"],
                "top": primary["top"],
                "width": primary["width"],
                "height": primary["height"],
            }

        left, top, width, height = region

        # Clamp to monitor bounds to avoid mss errors.
        mon_left = primary["left"]
        mon_top = primary["top"]
        mon_right = mon_left + primary["width"]
        mon_bottom = mon_top + primary["height"]

        clamped_left = max(left, mon_left)
        clamped_top = max(top, mon_top)
        clamped_right = min(left + width, mon_right)
        clamped_bottom = min(top + height, mon_bottom)

        clamped_width = clamped_right - clamped_left
        clamped_height = clamped_bottom - clamped_top

        if clamped_width <= 0 or clamped_height <= 0:
            raise RuntimeError(
                f"Capture region {region} is entirely outside the primary "
                f"monitor bounds ({primary})."
            )

        if (clamped_left, clamped_top, clamped_width, clamped_height) != (
            left,
            top,
            width,
            height,
        ):
            logger.warning(
                "Capture region %s was clamped to monitor bounds: left=%d "
                "top=%d width=%d height=%d",
                region,
                clamped_left,
                clamped_top,
                clamped_width,
                clamped_height,
            )

        return {
            "left": clamped_left,
            "top": clamped_top,
            "width": clamped_width,
            "height": clamped_height,
        }

    def _capture_frame(self, sct: Optional[mss.base.MSSBase] = None) -> np.ndarray:
        """
        Grab a single frame using ``mss`` and return it as a BGR array.

        Parameters
        ----------
        sct:
            An existing ``mss`` context to reuse.  If ``None`` a temporary
            context is created (used by :meth:`grab_frame`).

        Raises
        ------
        RuntimeError
            If capture fails.
        """
        if sct is not None:
            monitor = self._build_mss_monitor(sct)
            screenshot = sct.grab(monitor)
            bgra = np.array(screenshot, dtype=np.uint8)
            return bgra[:, :, :3]

        # Fallback: open a one-shot context (for synchronous grab_frame calls)
        with mss.mss() as tmp_sct:
            monitor = self._build_mss_monitor(tmp_sct)
            screenshot = tmp_sct.grab(monitor)
            bgra = np.array(screenshot, dtype=np.uint8)
            return bgra[:, :, :3]

    def _capture_loop(self) -> None:
        """Main loop executed in the background daemon thread.

        Holds a single persistent ``mss`` context for the lifetime of the
        loop to avoid the overhead of creating and tearing down the
        platform screen-capture handle on every frame.
        """
        with mss.mss() as sct:
            while self._running:
                try:
                    frame = self._capture_frame(sct)
                    ts = time.monotonic()

                    with self._frame_lock:
                        self._latest_frame = frame

                    with self._fps_lock:
                        self._timestamps.append(ts)

                except RuntimeError as exc:
                    logger.error("ScreenCapture error: %s", exc)
                    # Brief pause before retrying to avoid hammering the system.
                    time.sleep(0.1)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception("Unexpected error in capture loop: %s", exc)
                    time.sleep(0.1)
