"""Cleaned-area tracker.

We maintain a uint16 counts grid aligned with the ``/map`` grid; each cell
holds the number of times the robot's footprint covered it (saturating). The map is updated on every
robot pose update (10 Hz typical) so the UI can render "已清扫 / 未清扫" the
same way commercial cleaning bots do.

Why not subscribe to a topic the planner provides? opennav_coverage only
emits the *planned* path; it does not feed back "actually cleaned". We compute
it client-side so the visualisation works even when the robot is being driven
by another stack (teleop, fast-lio, ...).
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np


@dataclass
class CleanedSnapshot:
    """Read-only view of the cleaned mask for the renderer.

    Aligned to the map ``info`` it was created from.
    """
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    counts: np.ndarray            # shape (H, W), uint16  (visits per cell)
    target_mask: np.ndarray       # shape (H, W), bool    (cleanable cells)


# Current CAD bounds in base_link at X-stage q=0.  The ground-contacting
# cleaner footprint is asymmetric; the X stage can extend only body-right.
BRUSH_X0_M = 0.2045
BRUSH_X1_M = 0.56726
BRUSH_RIGHT_M = 0.5548
BRUSH_LEFT_M = 0.4270


def paint_brush(counts: np.ndarray, res: float, ox: float, oy: float,
                x: float, y: float, yaw: float,
                lateral_offset_m: float = 0.0) -> None:
    """把扫把有效区(随航向旋转的前伸矩形)+1 盖进 counts(uint16饱和)。
    供实车图章(tracker.stamp)与预规划承诺盖章共用——两边模型必须一致,
    否则预规划的 ratio 校验(≤3%)会系统性失配。"""
    h, w = counts.shape
    # Body-frame +y is robot-left.  Keep the standard base Pose unchanged and
    # translate only the cleaning tool frame; offset=0 preserves every legacy
    # tracker/planner call exactly.
    tool_x = x - math.sin(yaw) * float(lateral_offset_m)
    tool_y = y + math.cos(yaw) * float(lateral_offset_m)
    mid = (BRUSH_X0_M + BRUSH_X1_M) * 0.5
    cx = tool_x + math.cos(yaw) * mid
    cy = tool_y + math.sin(yaw) * mid
    rad = math.hypot(
        (BRUSH_X1_M - BRUSH_X0_M) * 0.5,
        max(BRUSH_RIGHT_M, BRUSH_LEFT_M),
    ) + res
    c0 = max(0, int((cx - rad - ox) / res))
    c1 = min(w, int((cx + rad - ox) / res) + 1)
    r0 = max(0, int((cy - rad - oy) / res))
    r1 = min(h, int((cy + rad - oy) / res) + 1)
    if c0 >= c1 or r0 >= r1:
        return
    xs = ox + (np.arange(c0, c1) + 0.5) * res - tool_x
    ys = oy + (np.arange(r0, r1) + 0.5) * res - tool_y
    gx, gy = np.meshgrid(xs, ys)
    ct, st = math.cos(yaw), math.sin(yaw)
    lon = gx * ct + gy * st
    lat = -gx * st + gy * ct
    m = (
        (lon >= BRUSH_X0_M)
        & (lon <= BRUSH_X1_M)
        & (lat >= -BRUSH_RIGHT_M)
        & (lat <= BRUSH_LEFT_M)
    )
    sub = counts[r0:r1, c0:c1]
    ceil = np.iinfo(np.uint16).max if counts.dtype == np.uint16 else (1 << 30)
    inc = m & (sub < ceil)
    sub[inc] += 1


class CleanedAreaTracker:
    """Per-pose footprint stamping. Thread-safe."""

    def __init__(self, footprint_radius_m: float = 0.40):
        self._lock = threading.RLock()
        self._w = 0; self._h = 0
        self._res = 0.05
        self._ox = 0.0; self._oy = 0.0
        self._counts: Optional[np.ndarray] = None
        self._target_mask: Optional[np.ndarray] = None
        self._radius = footprint_radius_m
        self._stamp_template: Optional[np.ndarray] = None    # circular bool stencil
        self._stamp_radius_cells = 0
        self._listeners: List[Callable[[], None]] = []
        self._dirty = False

    # ---------- configuration ----------

    def set_footprint_radius(self, r: float):
        with self._lock:
            if abs(r - self._radius) < 1e-3:
                return
            self._radius = float(r)
            self._rebuild_stamp_locked()

    # ---------- map intake ----------

    def configure_from_map(self, width: int, height: int, resolution: float,
                            origin_x: float, origin_y: float,
                            occupancy: np.ndarray):
        """Re-initialise the mask for a new ``/map``.

        ``occupancy`` is the int8 OccupancyGrid data reshaped to (H, W). Cells
        with value in [0, 50] are considered cleanable (free + low-cost).
        Everything else (unknown / occupied) stays out of the target mask.
        """
        with self._lock:
            cleanable = (occupancy >= 0) & (occupancy < 50)
            # P1: preserve the counts grid when the SAME map is re-published
            # (identical geometry AND identical cleanable mask) so a mid-session
            # /map republish doesn't wipe restored/accumulated coverage. A
            # genuinely new map (different geometry or content) zeros it.
            same_map = (self._counts is not None
                        and int(width) == self._w and int(height) == self._h
                        and abs(float(resolution) - self._res) < 1e-6
                        and abs(float(origin_x) - self._ox) < 1e-3
                        and abs(float(origin_y) - self._oy) < 1e-3
                        and self._target_mask is not None
                        and self._target_mask.shape == cleanable.shape
                        and np.array_equal(self._target_mask, cleanable))
            self._w = int(width); self._h = int(height)
            self._res = float(resolution)
            self._ox = float(origin_x); self._oy = float(origin_y)
            self._target_mask = cleanable
            if not same_map:
                self._counts = np.zeros((self._h, self._w), dtype=np.uint16)
            self._rebuild_stamp_locked()
            self._dirty = True
        self._fire()

    def _target_fingerprint_locked(self) -> str:
        """Stable hash of the cleanable mask — distinguishes a changed floor that
        shares the map name + geometry (P1 stale-map guard)."""
        import hashlib
        if self._target_mask is None:
            return ""
        return hashlib.blake2b(np.ascontiguousarray(self._target_mask).tobytes(),
                               digest_size=16).hexdigest()

    def _rebuild_stamp_locked(self):
        if self._res <= 0:
            return
        rc = max(1, int(round(self._radius / self._res)))
        self._stamp_radius_cells = rc
        d = 2 * rc + 1
        ys, xs = np.ogrid[-rc:rc + 1, -rc:rc + 1]
        self._stamp_template = (xs * xs + ys * ys <= rc * rc)

    # ---------- pose intake ----------

    def stamp(self, world_x: float, world_y: float,
              yaw: Optional[float] = None):
        # [1245实测 扫把建模] yaw已知 → 按前伸扫把矩形盖章(真实清扫区);
        # yaw未知 → 退回旧车心圆章(兼容, 现仅防御性保留)。
        if yaw is not None:
            with self._lock:
                if self._counts is None:
                    return
                paint_brush(self._counts, self._res, self._ox, self._oy,
                            world_x, world_y, yaw)
                self._dirty = True
            self._fire()
            return
        with self._lock:
            if self._counts is None or self._stamp_template is None:
                return
            col = int(np.floor((world_x - self._ox) / self._res))
            row = int(np.floor((world_y - self._oy) / self._res))
            r = self._stamp_radius_cells
            x0 = max(0, col - r);   y0 = max(0, row - r)
            x1 = min(self._w, col + r + 1); y1 = min(self._h, row + r + 1)
            if x0 >= x1 or y0 >= y1:
                return
            tx0 = x0 - (col - r); tx1 = tx0 + (x1 - x0)
            ty0 = y0 - (row - r); ty1 = ty0 + (y1 - y0)
            mask = self._stamp_template[ty0:ty1, tx0:tx1]
            sub = self._counts[y0:y1, x0:x1]
            # Add 1 only where mask is True, saturating at uint16 max. A plain
            # np.add on uint16 wraps 65535->0 (a cell visited 65536 times would
            # flip back to "未清扫" and trigger a spurious reclean), so we only
            # increment cells still below the ceiling.
            incr = mask & (sub < np.iinfo(np.uint16).max)
            sub[incr] += 1
            self._dirty = True
        self._fire()

    # ---------- snapshot ----------

    def snapshot(self) -> Optional[CleanedSnapshot]:
        with self._lock:
            if self._counts is None or self._target_mask is None:
                return None
            return CleanedSnapshot(
                width=self._w, height=self._h, resolution=self._res,
                origin_x=self._ox, origin_y=self._oy,
                counts=self._counts.copy(),
                target_mask=self._target_mask.copy(),
            )

    def stats(self) -> dict:
        """Return coverage ratio + areas in m²."""
        with self._lock:
            if self._counts is None or self._target_mask is None:
                return {"target_m2": 0.0, "cleaned_m2": 0.0,
                        "remaining_m2": 0.0, "ratio": 0.0}
            cell_a = self._res * self._res
            cleaned = int(((self._counts >= 1) & self._target_mask).sum())
            target = int(self._target_mask.sum())
            return {
                "target_m2":     target * cell_a,
                "cleaned_m2":    cleaned * cell_a,
                "remaining_m2": (target - cleaned) * cell_a,
                "ratio":         (cleaned / target) if target > 0 else 0.0,
            }

    def reset(self):
        with self._lock:
            if self._counts is not None:
                self._counts.fill(0)
                self._dirty = True
        self._fire()

    # ---------- persistence (P1: resume cleaning after interruption) ----------
    # Persist the per-cell counts grid keyed to the map so a 2-3h mission that
    # loses power / reboots / is exited can resume instead of re-cleaning from
    # zero. Kept off the 10 Hz stamp path — call save_npz() on a timer / at stop.
    _PERSIST_VERSION = 1

    def save_npz(self, path: str, map_key: str) -> bool:
        """Atomically write counts + geometry + map_key to ``path`` (.npz)."""
        with self._lock:
            if self._counts is None:
                return False
            counts = self._counts.copy()
            w, h, res, ox, oy = self._w, self._h, self._res, self._ox, self._oy
            fp = self._target_fingerprint_locked()
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = path + ".tmp"
            # Write via a file handle: np.savez_compressed APPENDS ".npz" to a
            # path argument (so "x.npz.tmp" -> "x.npz.tmp.npz"), but leaves a
            # file object alone — keeping the atomic-rename name exact.
            with open(tmp, "wb") as f:
                np.savez_compressed(
                    f, counts=counts, map_key=np.array(map_key),
                    fingerprint=np.array(fp),
                    version=np.array(self._PERSIST_VERSION),
                    w=np.array(w), h=np.array(h), res=np.array(res),
                    ox=np.array(ox), oy=np.array(oy))
                f.flush()
                os.fsync(f.fileno())          # durable temp before rename (power-loss)
            os.replace(tmp, path)             # atomic on POSIX
            try:                              # fsync the dir so the rename survives a cut
                dfd = os.open(d or ".", os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)             # never leak the fd if dir-fsync raises
            except OSError:
                pass
            return True
        except Exception:
            return False

    def load_npz(self, path: str, map_key: str) -> bool:
        """Restore counts IFF the saved map_key + geometry match the CURRENT map
        (call AFTER configure_from_map, which zeros counts). Returns True on
        restore; False (and no change) if absent / different map / mismatched
        grid — never restores misaligned coverage onto the wrong map."""
        if not path or not os.path.exists(path):
            return False
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["map_key"]) != map_key:
                    return False
                cw, ch = int(z["w"]), int(z["h"])
                cres, cox, coy = float(z["res"]), float(z["ox"]), float(z["oy"])
                saved_fp = str(z["fingerprint"]) if "fingerprint" in z else ""
                counts = z["counts"]
        except Exception:
            return False
        with self._lock:
            if self._counts is None:
                return False
            if (cw != self._w or ch != self._h or abs(cres - self._res) > 1e-6
                    or abs(cox - self._ox) > 1e-3 or abs(coy - self._oy) > 1e-3
                    or counts.shape != self._counts.shape):
                return False
            # Stale-map guard: same name+geometry but a changed floor (different
            # cleanable mask) must NOT restore old coverage onto the new layout.
            if saved_fp and saved_fp != self._target_fingerprint_locked():
                return False
            self._counts = counts.astype(np.uint16, copy=True)
            self._dirty = True
        self._fire()
        return True

    # ---------- listeners ----------

    def on_change(self, cb: Callable[[], None]):
        with self._lock:
            self._listeners.append(cb)

    def _fire(self):
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb()
            except Exception:
                pass
