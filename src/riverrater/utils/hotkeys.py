"""
Global hotkey manager for RiverRater using pynput.

Platform notes:
- macOS: pynput requires Accessibility permissions. Go to System Settings →
  Privacy & Security → Accessibility and add your terminal / Python executable.
- Linux: pynput may require X11 (not Wayland). Install python-xlib if needed.
- Windows: Should work out of the box; run as administrator if hotkeys are blocked.

Thread safety: pynput invokes callbacks from the listener thread. All registered
callbacks must be thread-safe. If they interact with PyQt6 widgets, use
QMetaObject.invokeMethod or signals/slots to marshal calls to the main thread.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

try:
    from pynput import keyboard as _pynput_keyboard
    _HAS_PYNPUT = True
except ImportError:
    _HAS_PYNPUT = False
    _pynput_keyboard = None  # type: ignore

logger = logging.getLogger(__name__)


class HotkeyManager:
    """Manages global hotkeys using pynput.GlobalHotKeys.

    Hotkey strings use pynput notation, e.g.:
        "<ctrl>+<shift>+h"
        "<alt>+f4"
        "a"

    Example::

        mgr = HotkeyManager()
        mgr.register("<ctrl>+<shift>+h", lambda: print("toggle!"))
        mgr.start()
        # ... later ...
        mgr.stop()
    """

    def __init__(self) -> None:
        """Initialise an empty hotkey registry."""
        self._registry: dict[str, Callable[[], None]] = {}
        self._listener: "_pynput_keyboard.GlobalHotKeys | None" = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register(self, hotkey_str: str, callback: Callable[[], None]) -> None:
        """Register a hotkey with its callback.

        Args:
            hotkey_str: pynput hotkey notation, e.g. ``"<ctrl>+<shift>+h"``.
            callback: Zero-argument callable invoked (from the listener thread)
                when the hotkey is pressed.  Must be thread-safe.

        Note:
            If a listener is already running, you must call :meth:`stop` then
            :meth:`start` again for the new hotkey to take effect.
        """
        with self._lock:
            self._registry[hotkey_str] = callback
            logger.debug("Registered hotkey: %s", hotkey_str)

    def unregister(self, hotkey_str: str) -> None:
        """Remove a hotkey from the registry.

        Args:
            hotkey_str: The exact string used when the hotkey was registered.
        """
        with self._lock:
            removed = self._registry.pop(hotkey_str, None)
            if removed is not None:
                logger.debug("Unregistered hotkey: %s", hotkey_str)
            else:
                logger.warning("Attempted to unregister unknown hotkey: %s", hotkey_str)

    # ------------------------------------------------------------------
    # Listener lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the pynput GlobalHotKeys listener in a daemon thread.

        If pynput is unavailable (e.g. missing system dependencies) this method
        logs a warning and returns without raising, so the rest of the
        application can still function.
        """
        if not _HAS_PYNPUT:
            logger.warning(
                "pynput not available — hotkeys disabled. "
                "Install pynput: pip install pynput"
            )
            return

        if self._listener is not None and self._listener.is_alive():
            logger.debug("Hotkey listener already running; ignoring start().")
            return

        with self._lock:
            hotkeys_snapshot = dict(self._registry)

        if not hotkeys_snapshot:
            logger.debug("No hotkeys registered; listener not started.")
            return

        def _safe_wrapper(hk: str, cb: Callable[[], None]) -> Callable[[], None]:
            """Wrap callback with exception handling."""
            def _wrapper() -> None:
                try:
                    cb()
                except Exception:  # noqa: BLE001
                    logger.exception("Exception in hotkey callback for '%s'", hk)
            return _wrapper

        safe_map = {hk: _safe_wrapper(hk, cb) for hk, cb in hotkeys_snapshot.items()}

        try:
            self._listener = _pynput_keyboard.GlobalHotKeys(safe_map)
            self._listener.daemon = True
            self._listener.start()
            logger.info("Hotkey listener started with %d hotkeys.", len(safe_map))
        except Exception:  # noqa: BLE001
            logger.exception("Failed to start hotkey listener.")
            self._listener = None

    def stop(self) -> None:
        """Stop the pynput listener and release system resources.

        Safe to call even if the listener was never started.
        """
        if self._listener is None:
            return
        try:
            self._listener.stop()
            logger.info("Hotkey listener stopped.")
        except Exception:  # noqa: BLE001
            logger.exception("Error while stopping hotkey listener.")
        finally:
            self._listener = None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "running" if (self._listener and self._listener.is_alive()) else "stopped"
        return f"HotkeyManager(hotkeys={list(self._registry)!r}, status={status!r})"
