"""
Structured logging setup for RiverRater.

Usage::

    from riverrater.utils.logging import setup_logging, get_logger

    setup_logging(level="DEBUG", log_file="riverrater.log")

    log = get_logger(__name__)
    log.info("Application started")
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Log format shared across all handlers
_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Directory for log files: ~/.riverrater/logs/
_LOG_DIR = Path.home() / ".riverrater" / "logs"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger with a consistent format.

    Args:
        level: Logging level name, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
            Case-insensitive.
        log_file: Optional filename (not full path) for a rotating log file.
            The file is created inside ``~/.riverrater/logs/``.  Pass ``None``
            to disable file logging.

    Calling this function multiple times is safe — existing handlers are
    replaced rather than duplicated.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any pre-existing handlers to avoid duplicate output when this
    # function is called more than once (e.g., from tests).
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # --- Console handler ---------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # --- Optional file handler --------------------------------------------
    if log_file is not None:
        _ensure_log_directory()
        log_path = _LOG_DIR / log_file
        try:
            file_handler = _build_file_handler(log_path, numeric_level, formatter)
            root_logger.addHandler(file_handler)
        except OSError:
            root_logger.warning(
                "Could not create log file at %s — file logging disabled.",
                log_path,
            )

    root_logger.debug(
        "Logging configured: level=%s, file=%s",
        level.upper(),
        str(_LOG_DIR / log_file) if log_file else "None",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a :class:`logging.Logger` for *name*.

    This is a thin convenience wrapper around :func:`logging.getLogger` so
    callers only need to import from this module.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_log_directory() -> None:
    """Create ``~/.riverrater/logs/`` if it does not already exist."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # Non-fatal — warn but continue.
        logging.getLogger(__name__).warning(
            "Cannot create log directory %s: %s", _LOG_DIR, exc
        )


def _build_file_handler(
    path: Path,
    level: int,
    formatter: logging.Formatter,
) -> logging.FileHandler:
    """Build a :class:`logging.FileHandler` for *path*.

    Uses a :class:`logging.handlers.RotatingFileHandler` (max 5 MB × 3 files)
    to prevent unbounded disk growth.
    """
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(
        filename=str(path),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler
