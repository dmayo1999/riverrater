#!/usr/bin/env python3
"""
Baseline profiler for a single GameController poker tick.

Measures wall-clock time for capture, template detection, pot OCR, and
poker math so regressions in the processing loop are easy to spot.

Usage::

    python scripts/profile_tick.py
    python scripts/profile_tick.py --ticks 50 --mode poker
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from riverrater.game.state import Card, GameMode, Rank, Suit  # noqa: E402
from riverrater.main import AppConfig, GameController  # noqa: E402
from riverrater.vision.template_engine import TemplateEngine  # noqa: E402


def _make_card(rank_val: str, suit_val: str) -> Card:
    return Card(rank=Rank(rank_val), suit=Suit(suit_val))


def _build_controller(mode: str) -> GameController:
    """Construct a controller with synthetic capture for profiling."""
    config = AppConfig(game_mode=mode)
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    capture = MagicMock()
    capture.get_latest_frame.return_value = frame

    template_engine = TemplateEngine()
    overlay = MagicMock()

    controller = GameController(config, capture, template_engine, overlay)
    controller._frame_count = 0
    controller._prev_frame_gray = None

    if mode == GameMode.POKER.value:
        controller.poker_state.hole_cards = [_make_card("A", "h"), _make_card("K", "d")]
        controller.poker_state.community_cards = [_make_card("2", "c")]
        controller.poker_state.pot_size = 100.0
        controller.poker_state.bet_to_call = 25.0

    return controller


def _time_call(label: str, fn: Callable[[], None], buckets: dict[str, list[float]]) -> None:
    start = time.perf_counter()
    fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    buckets[label].append(elapsed_ms)


def profile_ticks(controller: GameController, *, ticks: int) -> dict[str, list[float]]:
    """Run *ticks* processing iterations and collect per-phase timings."""
    buckets: dict[str, list[float]] = {
        "process_frame": [],
        "tick_poker": [],
        "tick_blackjack": [],
    }

    for i in range(ticks):
        controller._frame_count = i

        if controller.mode == GameMode.POKER:
            _time_call("tick_poker", controller._tick_poker, buckets)
        else:
            _time_call("tick_blackjack", controller._tick_blackjack, buckets)

        _time_call("process_frame", controller.process_frame, buckets)

    return buckets


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(values)
    p95_idx = max(0, int(len(ordered) * 0.95) - 1)
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[p95_idx],
        "max_ms": max(values),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile a single GameController tick.")
    parser.add_argument("--ticks", type=int, default=30, help="Number of iterations (default: 30)")
    parser.add_argument(
        "--mode",
        choices=["poker", "blackjack"],
        default="poker",
        help="Game mode to profile (default: poker)",
    )
    args = parser.parse_args(argv)

    controller = _build_controller(args.mode)
    buckets = profile_ticks(controller, ticks=args.ticks)

    print(f"RiverRater tick profiler — mode={args.mode}, ticks={args.ticks}")
    print("-" * 56)
    for label, values in buckets.items():
        stats = _summarize(values)
        if stats["count"] == 0:
            continue
        print(
            f"{label:16s}  mean={stats['mean_ms']:7.2f}ms  "
            f"p50={stats['p50_ms']:7.2f}ms  "
            f"p95={stats['p95_ms']:7.2f}ms  "
            f"max={stats['max_ms']:7.2f}ms"
        )
    print("-" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())