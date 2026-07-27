"""Helpers for rolling execution of long coverage paths."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple


Point2 = Tuple[float, float]


@dataclass(frozen=True)
class PathChunk:
    start_idx: int
    end_idx: int
    next_start_idx: int
    start_distance_m: float = 0.0
    ends_at_hard_stop: bool = False


def _forward_search_limit(
    path: Sequence[Point2],
    start_idx: int,
    search_length_m: float,
    end_idx: Optional[int] = None,
) -> int:
    if search_length_m <= 0.0:
        return start_idx

    limit = start_idx
    accum = 0.0
    prev_x, prev_y = path[start_idx]
    final = len(path) - 1 if end_idx is None else min(
        len(path) - 1, max(start_idx, int(end_idx)))
    for i in range(start_idx + 1, final + 1):
        x, y = path[i]
        accum += math.hypot(float(x) - float(prev_x), float(y) - float(prev_y))
        prev_x, prev_y = x, y
        limit = i
        if accum >= search_length_m:
            break
    return limit


def _nearest_forward_index(
    path: Sequence[Point2],
    start_idx: int,
    current_xy: Point2,
    search_length_m: float,
    end_idx: Optional[int] = None,
) -> Tuple[int, float]:
    limit = _forward_search_limit(
        path, start_idx, search_length_m, end_idx=end_idx)
    cx, cy = current_xy
    best_idx = start_idx
    best_dist = float("inf")
    for i in range(start_idx, limit + 1):
        x, y = path[i]
        dist = math.hypot(float(x) - float(cx), float(y) - float(cy))
        if dist < best_dist:
            best_idx = i
            best_dist = dist
    return best_idx, best_dist


def choose_path_chunk(
    path: Sequence[Point2],
    start_idx: int,
    max_length_m: float,
    current_xy: Optional[Point2] = None,
    max_snap_m: float = 0.75,
    search_length_m: float = 4.0,
    hard_stop_indices: Optional[Sequence[int]] = None,
) -> PathChunk:
    """Pick a short contiguous window from a dense coverage path.

    Nav2's DWB controller transforms and prunes the submitted global plan into
    a local window. Sending one long boustrophedon path with many nearby future
    lanes makes it much easier for pruning to jump or leave the controller with
    an empty transformed plan after drift. This helper keeps each FollowPath
    goal local while preserving the original point order.
    """
    if not path:
        return PathChunk(0, 0, 0)

    last_idx = len(path) - 1
    start = max(0, min(int(start_idx), last_idx))
    stops = sorted({
        max(0, min(int(index), last_idx))
        for index in (hard_stop_indices or ())
    })

    def next_hard_stop(after: int) -> Optional[int]:
        return next((index for index in stops if index > after), None)

    snap_limit = next_hard_stop(start)
    start_distance = 0.0
    if current_xy is not None:
        nearest_idx, nearest_dist = _nearest_forward_index(
            path, start, current_xy, search_length_m,
            end_idx=snap_limit)
        start_distance = nearest_dist
        if nearest_dist <= max_snap_m:
            start = nearest_idx

    if start >= last_idx or max_length_m <= 0.0:
        return PathChunk(
            start, start, start, start_distance,
            ends_at_hard_stop=start in stops)

    hard_limit = next_hard_stop(start)

    accum = 0.0
    prev_x, prev_y = path[start]
    end = start
    # [2026-07-07 弯口拆段] U形掉头(累计转角>=120°后回直)处切段: MPPI 的路径前瞻(~1.4m)
    # 在弯口穿过半圆落到【反向车道】上, "往前开"与"目标在身后"冲突 → 实测弯口 cmd≈0
    # 躺平22-97s。段尾=弧出口后, 前瞻被段边界截断, 弯=减速到段尾+滚动交接, 直道干净起步。
    # 普通拐角(圆角<120°)不拆; 段长<1.2m 不拆(防碎段)。
    _prev_h = None
    _in_turn = False
    _turn_acc = 0.0
    _straight_len = 0.0
    final = last_idx if hard_limit is None else hard_limit
    for i in range(start + 1, final + 1):
        x, y = path[i]
        _dx = float(x) - float(prev_x)
        _dy = float(y) - float(prev_y)
        _sl = math.hypot(_dx, _dy)
        accum += _sl
        prev_x, prev_y = x, y
        end = i
        if _sl > 1e-4:
            _h = math.atan2(_dy, _dx)
            if _prev_h is not None:
                _d = _h - _prev_h
                while _d > math.pi:
                    _d -= 2.0 * math.pi
                while _d < -math.pi:
                    _d += 2.0 * math.pi
                if _in_turn:
                    _turn_acc += _d
                    _straight_len = _straight_len + _sl if abs(_d) < math.radians(4.0) else 0.0
                    if abs(_turn_acc) >= math.radians(120.0) and abs(_d) < math.radians(8.0):
                        if accum >= 0.6:   # [codex审B3] 1.2→0.6: 0.9m短带+吸附前移0.75会绕过硬门槛
                            break   # 段切在弧出口(已回直)
                        _in_turn = False
                        _turn_acc = 0.0
                        _straight_len = 0.0
                    elif _straight_len > 0.5:
                        # [codex审W4] 回直0.5m仍没凑够120° → 复位, 防两个分离90°角(沿边环/
                        # 补漏dogleg)被跨直道累计成假U弯乱切段。
                        _in_turn = False
                        _turn_acc = 0.0
                        _straight_len = 0.0
                elif abs(_d) > math.radians(12.0):
                    _in_turn = True
                    _turn_acc = _d
                    _straight_len = 0.0
            _prev_h = _h
        if accum >= max_length_m:
            break

    return PathChunk(
        start, end, end, start_distance,
        ends_at_hard_stop=(hard_limit is not None and end == hard_limit))


def next_index_after_block(
    path: Sequence[Point2], blocked_start_idx: int, skip_length_m: float,
    stop_at_reversal_deg: Optional[float] = None,
) -> int:
    """被堵段后, 沿弧长前跳 skip_length_m 得到新的起始 index(夹到末点)。
    [2026-07-07 掉头感知] stop_at_reversal_deg 不为 None 时: 累计转角超此阈值(=前方
    有掉头/U弯)立即停在拐弯前——防前跳沿半圆弧跨到反向【隔壁车道】(实测跳1m跨0.82m
    半圆落隔壁道→偏离剩余路径1.31m整单停)。停在掉头前若未前进→调用方落全局绕行。"""
    last = len(path) - 1
    if last <= 0:
        return 0
    start = max(0, min(int(blocked_start_idx), last))
    if start >= last:
        return last
    accum = 0.0
    px, py = path[start]
    prev_h = None
    cum_turn = 0.0
    for i in range(start + 1, len(path)):
        x, y = path[i]
        dx, dy = float(x) - float(px), float(y) - float(py)
        sl = math.hypot(dx, dy)
        if stop_at_reversal_deg is not None and sl > 1e-4:
            h = math.atan2(dy, dx)
            if prev_h is not None:
                dh = (h - prev_h + math.pi) % (2.0 * math.pi) - math.pi
                cum_turn += abs(dh)
                if math.degrees(cum_turn) > stop_at_reversal_deg:
                    return max(start, i - 1)   # 停在掉头前, 不跨到反向车道
            prev_h = h
        accum += sl
        px, py = x, y
        if accum >= skip_length_m:
            return i
    return last


def select_safe_rejoin_index(
    path: Sequence[Point2],
    blocked_start_idx: int,
    clearance_at: Callable[[Point2], Optional[float]],
    *,
    min_skip_m: float = 0.8,
    max_skip_m: float = 6.0,
    required_clear_m: float = 0.55,
) -> Optional[int]:
    """Select a confirmed-clear rejoin point after a blocked path segment.

    Unlike a fixed-distance index skip, this scans forward along path arc length
    and accepts the first point whose live-obstacle clearance is confirmed.  A
    ``None`` clearance is deliberately not accepted: perimeter recovery must not
    drive toward an unobserved/unknown rejoin point.  The bounded window keeps a
    closed perimeter ring from silently jumping to another side of the room.
    """
    if len(path) < 2 or max_skip_m < min_skip_m:
        return None
    last = len(path) - 1
    start = max(0, min(int(blocked_start_idx), last))
    if start >= last:
        return None
    arc = 0.0
    px, py = path[start]
    for i in range(start + 1, len(path)):
        x, y = path[i]
        arc += math.hypot(float(x) - float(px), float(y) - float(py))
        px, py = x, y
        if arc > max_skip_m:
            break
        if arc < min_skip_m:
            continue
        clearance = clearance_at((float(x), float(y)))
        if clearance is not None and clearance >= required_clear_m:
            return i
    return None


def select_same_sweep_rejoin_index(
    path: Sequence[Point2],
    blocked_start_idx: int,
    clearance_at: Callable[[Point2], Optional[float]],
    *,
    min_skip_m: float = 0.8,
    max_skip_m: float = 3.0,
    required_clear_m: float = 0.45,
    max_turn_deg: float = 55.0,
) -> Optional[int]:
    """Pick a live-clear rejoin without crossing into the next coverage lane.

    Coverage paths are densely sampled lawnmower curves.  A spatial nearest
    point is unsafe here because the adjacent return lane is only one swath
    away.  This selector advances *only* along path arc length and stops before
    the accumulated heading change reaches a U-turn / lane connector.  The
    resulting point therefore remains on the same sweep (or straight
    connector) as the blocked point.

    ``None`` clearance is fail-closed: proactive diversion is optional, so an
    unobserved rejoin is rejected and the normal MPPI/reactive recovery chain
    remains in charge.
    """
    if len(path) < 2 or max_skip_m < min_skip_m:
        return None
    last = len(path) - 1
    start = max(0, min(int(blocked_start_idx), last))
    if start >= last:
        return None

    px, py = path[start]
    prev_h = None
    if start > 0:
        qx, qy = path[start - 1]
        dx, dy = float(px) - float(qx), float(py) - float(qy)
        if math.hypot(dx, dy) > 1e-4:
            prev_h = math.atan2(dy, dx)

    arc = 0.0
    turn = 0.0
    for i in range(start + 1, len(path)):
        x, y = path[i]
        dx, dy = float(x) - float(px), float(y) - float(py)
        step = math.hypot(dx, dy)
        px, py = x, y
        if step <= 1e-4:
            continue
        h = math.atan2(dy, dx)
        if prev_h is not None:
            dh = (h - prev_h + math.pi) % (2.0 * math.pi) - math.pi
            turn += abs(dh)
            if math.degrees(turn) > max_turn_deg:
                break
        prev_h = h
        arc += step
        if arc > max_skip_m:
            break
        if arc < min_skip_m:
            continue
        clearance = clearance_at((float(x), float(y)))
        if clearance is not None and clearance >= required_clear_m:
            return i
    return None


def splice_detour(
    coverage: Sequence[Point2],
    detour: Sequence[Point2],
    reroute_idx: int,
    min_step: float = 0.015,
) -> list:
    """拼接遇障绕行: detour 段(全局规划器绕过障碍) + 覆盖路径从 reroute_idx 起的剩余部分。

    用于 Stage 2 闭环重规划: 被堵时不再盲目跳段, 而是把"绕行段 + 续扫"拼成一条
    新路径继续执行。去掉相邻重复点(detour 末点常≈coverage[reroute_idx])。
    """
    if not coverage:
        idx = 0
    else:
        idx = max(0, min(int(reroute_idx), len(coverage)))
    merged = list(detour) + list(coverage[idx:])
    out: list = []
    for p in merged:
        q = (float(p[0]), float(p[1]))
        if not out or math.hypot(q[0] - out[-1][0], q[1] - out[-1][1]) > min_step:
            out.append(q)
    return out


def remap_hard_stops_after_splice(
    coverage_length: int,
    spliced_length: int,
    reroute_idx: int,
    hard_stop_indices: Sequence[int],
) -> list:
    """Map future lane boundaries onto ``detour + remaining coverage``.

    ``splice_detour`` discards the already-travelled prefix.  Its retained
    coverage tail is unchanged, apart from a possible duplicate at the join,
    so the tail start is recoverable from the two lengths.  The join itself is
    also a hard stop: finish the temporary obstacle connection, align with the
    original sweep, and only then expose that sweep to MPPI.
    """
    old_length = max(0, int(coverage_length))
    new_length = max(0, int(spliced_length))
    if old_length <= 0 or new_length <= 0:
        return []
    rejoin = max(0, min(int(reroute_idx), old_length - 1))
    retained_tail_length = old_length - rejoin
    tail_start = max(0, new_length - retained_tail_length)
    last = new_length - 1
    remapped = {tail_start}
    for stop in hard_stop_indices:
        index = int(stop)
        if index < rejoin or index >= old_length:
            continue
        remapped.add(tail_start + index - rejoin)
    return sorted(index for index in remapped if 0 < index <= last)
