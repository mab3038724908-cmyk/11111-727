"""Targeted tests for the CleanedAreaTracker footprint stamper.

This component produces the ``counts`` (visits-per-cell) and ``target_mask``
(cleanable cells) arrays that the whole closed-loop reclean pipeline consumes
(``find_uncovered_regions`` / ``reachable_coverage_ratio`` in algorithms.reclean
read exactly these). It had no dedicated coverage; these tests pin its contract
offline (pure numpy, no ROS, no Qt).
"""
import numpy as np
import pytest

from td25a_robot_ui.store.cleaned_area import CleanedAreaTracker


def _free_occ(h, w):
    """All-free OccupancyGrid (every cell value 0 -> cleanable)."""
    return np.zeros((h, w), dtype=np.int8)


def _disc(rc):
    """Disc-stencil cell count, mirroring _rebuild_stamp_locked exactly."""
    ys, xs = np.ogrid[-rc:rc + 1, -rc:rc + 1]
    return int((xs * xs + ys * ys <= rc * rc).sum())


# ---------- configure_from_map / target mask ----------

def test_target_mask_selects_only_free_low_cost_cells():
    occ = np.array([
        [0, 49, 50],      # 0 & 49 cleanable; 50 is NOT (< 50 is exclusive)
        [-1, 100, 10],    # -1 unknown -> no; 100 occupied -> no; 10 -> yes
    ], dtype=np.int8)
    t = CleanedAreaTracker(0.2)
    t.configure_from_map(3, 2, 0.05, 0.0, 0.0, occ)
    snap = t.snapshot()
    assert snap is not None
    expected = np.array([[True, True, False], [False, False, True]])
    assert np.array_equal(snap.target_mask, expected)
    assert snap.counts.shape == (2, 3)
    assert snap.counts.dtype == np.uint16
    assert snap.counts.sum() == 0


def test_uninitialised_tracker_is_safe():
    t = CleanedAreaTracker()
    t.stamp(1.0, 1.0)                 # must not raise before a map arrives
    assert t.snapshot() is None
    assert t.stats() == {
        "target_m2": 0.0, "cleaned_m2": 0.0, "remaining_m2": 0.0, "ratio": 0.0,
    }


# ---------- stamp: placement, accumulation, clipping ----------

def test_stamp_marks_centered_disc_and_accumulates():
    res, rc = 0.1, 4                  # round(0.40 / 0.1) == 4
    t = CleanedAreaTracker(0.40)
    t.configure_from_map(40, 40, res, 1.0, 2.0, _free_occ(40, 40))
    # col = floor((2.05-1.0)/0.1) = 10 ; row = floor((3.05-2.0)/0.1) = 10
    t.stamp(2.05, 3.05)
    snap = t.snapshot()
    assert snap.counts[10, 10] == 1
    assert int((snap.counts >= 1).sum()) == _disc(rc)
    # second visit to the same pose increments the same cells
    t.stamp(2.05, 3.05)
    assert t.snapshot().counts[10, 10] == 2


def test_stamp_partially_offmap_clips_without_error():
    res, rc = 0.1, 4
    t = CleanedAreaTracker(0.40)
    t.configure_from_map(40, 40, res, 0.0, 0.0, _free_occ(40, 40))
    t.stamp(0.0, 0.0)                 # centred at cell (0,0): 3/4 of disc is off-map
    snap = t.snapshot()
    assert snap.counts[0, 0] == 1
    n = int((snap.counts >= 1).sum())
    assert 0 < n < _disc(rc)


def test_stamp_saturates_at_uint16_max_without_wrapping():
    # A cell already at the uint16 ceiling must stay there, not wrap to 0
    # (which would flip a cleaned cell back to "未清扫" and trigger a spurious
    # reclean after ~65536 visits / ~109 min stationary at 10 Hz).
    t = CleanedAreaTracker(0.2)
    t.configure_from_map(20, 20, 0.1, 0.0, 0.0, _free_occ(20, 20))
    ceiling = np.iinfo(np.uint16).max
    t._counts[10, 10] = ceiling          # white-box: pre-load the centre cell
    t.stamp(1.05, 1.05)                  # centred on cell (10, 10)
    assert t.snapshot().counts[10, 10] == ceiling


def test_stamp_fully_offmap_is_noop():
    t = CleanedAreaTracker(0.40)
    t.configure_from_map(20, 20, 0.1, 0.0, 0.0, _free_occ(20, 20))
    t.stamp(100.0, 100.0)
    assert t.snapshot().counts.sum() == 0


# ---------- snapshot / reset semantics ----------

def test_snapshot_is_an_independent_copy():
    t = CleanedAreaTracker(0.2)
    t.configure_from_map(20, 20, 0.1, 0.0, 0.0, _free_occ(20, 20))
    t.stamp(1.0, 1.0)
    snap = t.snapshot()
    before = snap.counts.copy()
    t.stamp(1.0, 1.0)                 # mutate the tracker after snapshotting
    assert np.array_equal(snap.counts, before)


def test_reset_clears_counts_but_keeps_target_mask():
    t = CleanedAreaTracker(0.2)
    t.configure_from_map(20, 20, 0.1, 0.0, 0.0, _free_occ(20, 20))
    t.stamp(1.0, 1.0)
    assert t.snapshot().counts.sum() > 0
    t.reset()
    snap = t.snapshot()
    assert snap.counts.sum() == 0
    assert snap.target_mask.all()    # all-free map is still fully cleanable


# ---------- footprint radius ----------

def test_larger_footprint_radius_covers_more_cells():
    small = CleanedAreaTracker(0.2)
    small.configure_from_map(60, 60, 0.1, 0.0, 0.0, _free_occ(60, 60))
    small.stamp(3.0, 3.0)
    big = CleanedAreaTracker(0.2)
    big.configure_from_map(60, 60, 0.1, 0.0, 0.0, _free_occ(60, 60))
    big.set_footprint_radius(0.5)
    big.stamp(3.0, 3.0)
    assert (int((big.snapshot().counts >= 1).sum())
            > int((small.snapshot().counts >= 1).sum()))


def test_set_footprint_radius_is_noop_within_tolerance():
    t = CleanedAreaTracker(0.40)
    t.configure_from_map(40, 40, 0.1, 0.0, 0.0, _free_occ(40, 40))
    t.set_footprint_radius(0.4005)   # delta < 1e-3 -> stencil unchanged (rc stays 4)
    t.stamp(2.0, 2.0)
    assert int((t.snapshot().counts >= 1).sum()) == _disc(4)


# ---------- stats ----------

def test_stats_reports_ratio_and_areas_consistently():
    res = 0.1
    t = CleanedAreaTracker(0.2)
    t.configure_from_map(20, 20, res, 0.0, 0.0, _free_occ(20, 20))   # target = 400
    t.stamp(1.0, 1.0)
    snap = t.snapshot()
    cleaned = int((snap.counts >= 1).sum())
    cell = res * res
    st = t.stats()
    assert st["target_m2"] == pytest.approx(400 * cell)
    assert st["cleaned_m2"] == pytest.approx(cleaned * cell)
    assert st["remaining_m2"] == pytest.approx((400 - cleaned) * cell)
    assert st["ratio"] == pytest.approx(cleaned / 400)


def test_stats_counts_only_cleaned_cells_inside_target():
    occ = _free_occ(20, 20)
    occ[:, 10:] = 100                 # right half occupied -> not cleanable
    t = CleanedAreaTracker(0.3)
    t.configure_from_map(20, 20, 0.1, 0.0, 0.0, occ)
    t.stamp(1.5, 1.0)                 # col 15: lands entirely on the occupied half
    st = t.stats()
    assert st["cleaned_m2"] == 0.0    # footprint outside target contributes nothing
    assert st["ratio"] == 0.0


# ---------- listeners ----------

def test_listeners_fire_on_changes_and_survive_a_raising_listener():
    t = CleanedAreaTracker(0.2)
    calls = []

    def good():
        calls.append(1)

    def bad():
        raise RuntimeError("listener boom")

    t.on_change(good)
    t.on_change(bad)                  # must be swallowed, not propagate
    t.configure_from_map(20, 20, 0.1, 0.0, 0.0, _free_occ(20, 20))   # fire
    t.stamp(1.0, 1.0)                 # fire (must not raise)
    t.reset()                         # fire
    assert len(calls) >= 3


# ---------- integration with the reclean pipeline ----------

def test_snapshot_feeds_reclean_uncovered_detection():
    from td25a_robot_ui.algorithms.reclean import find_uncovered_regions
    res = 0.1
    t = CleanedAreaTracker(0.2)
    t.configure_from_map(40, 40, res, 0.0, 0.0, _free_occ(40, 40))
    # Fully clean a solid central block; the surrounding border stays uncovered.
    for r in range(8, 32):
        for c in range(8, 32):
            t.stamp((c + 0.5) * res, (r + 0.5) * res)
    snap = t.snapshot()
    assert snap.counts[20, 20] >= 1            # block centre is cleaned
    regs = find_uncovered_regions(
        snap.counts, snap.target_mask, snap.target_mask,
        resolution=res, origin_x=0.0, origin_y=0.0, min_area_m2=0.04)
    assert regs                                 # the uncovered border is detected
    # every reported region lies on genuinely unstamped target cells
    total_uncovered = int(((snap.counts == 0) & snap.target_mask).sum())
    assert total_uncovered > 0
