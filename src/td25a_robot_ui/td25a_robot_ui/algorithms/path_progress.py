"""Helpers for displaying executed progress along dense coverage paths."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

Point = Tuple[float, float]


def advance_path_progress_index(
    path: Sequence[Point],
    start_idx: int,
    pose: Point,
    max_snap_m: float,
    max_advance_m: float,
) -> int:
    """Return a conservative forward-only progress index.

    Coverage paths contain nearby parallel lanes. Searching hundreds of points
    ahead by Euclidean distance can jump to a future lane and erase a large
    uncleaned section. This helper only searches a short integrated distance
    ahead of the current progress index.
    """
    if not path:
        return 0
    start = max(0, min(int(start_idx), len(path) - 1))
    end = _advance_end_index(path, start, max(0.0, max_advance_m))
    best_i = start
    best_d2 = float("inf")
    px, py = pose
    for i in range(start, end + 1):
        x, y = path[i]
        d2 = (x - px) * (x - px) + (y - py) * (y - py)
        if d2 < best_d2:
            best_i = i
            best_d2 = d2
    if best_i > start and best_d2 <= max_snap_m * max_snap_m:
        return best_i
    return start


def _advance_end_index(path: Sequence[Point], start: int, max_distance_m: float) -> int:
    accum = 0.0
    i = start
    while i + 1 < len(path):
        a = path[i]
        b = path[i + 1]
        accum += math.hypot(b[0] - a[0], b[1] - a[1])
        if accum > max_distance_m:
            break
        i += 1
    return i
