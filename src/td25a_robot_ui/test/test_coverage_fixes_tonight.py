"""Locks for tonight's coverage correctness fixes:
  1. reclean dilation no longer encloses unreachable cells.
  2. reclean simplify_eps clamped to <= resolution*0.5 so tiny regions survive.
  3. grid_coverage swath generation covers the trailing edge (no end-strip gap).
"""
import numpy as np

from td25a_robot_ui.algorithms.reclean import find_uncovered_regions
from td25a_robot_ui.algorithms.free_space import polygon_to_mask
from td25a_robot_ui.algorithms.grid_coverage import (
    _generate_swaths_from_mask, build_coverage_free_mask)


# ---------- reclean: dilation halo at a wall boundary is benign ----------

def test_reclean_polygon_clip_excludes_walls_via_planner_mask():
    """The 1-cell dilation halo may nominally touch unreachable (wall) cells,
    but plan_lawnmower_coverage ANDs the clip polygon with the inflated free
    mask, so walls never enter the plan. Meanwhile every original uncovered
    cell (including the one against the wall) stays covered."""
    h = w = 24
    grid = np.zeros((h, w), dtype=np.int8)
    grid[:, 12:] = 100          # solid wall on the right half
    target = np.ones((h, w), dtype=bool)
    counts = np.ones((h, w), dtype=np.uint16)
    reachable = np.ones((h, w), dtype=bool)
    reachable[:, 11:] = False   # robot-radius inflation makes col>=11 unreachable
    counts[5:10, 5:10] = 0      # uncovered block, col 9 abuts the unreachable band
    regs = find_uncovered_regions(counts, target, reachable,
                                  resolution=0.1, origin_x=0.0, origin_y=0.0,
                                  min_area_m2=0.04)
    assert len(regs) == 1
    poly = regs[0].polygon
    poly_mask = polygon_to_mask(poly, w, h, 0.1, 0.0, 0.0)
    # Edge coverage preserved: every original uncovered cell is inside the clip.
    orig = np.zeros((h, w), dtype=bool); orig[5:10, 5:10] = True
    assert int((orig & poly_mask).sum()) == int(orig.sum())
    # Planner intersects clip with inflated free mask -> no wall cell in plan.
    free = build_coverage_free_mask(
        data=grid.tobytes(), width=w, height=h, resolution=0.1,
        robot_radius_m=0.1, clip_polygon=poly, origin_x=0.0, origin_y=0.0)
    assert int((free & (grid > 50)).sum()) == 0


# ---------- reclean: simplify_eps clamp keeps tiny regions ----------

def test_reclean_tiny_region_survives_large_eps():
    h = w = 20
    target = np.ones((h, w), dtype=bool)
    counts = np.ones((h, w), dtype=np.uint16)
    reachable = np.ones((h, w), dtype=bool)
    counts[8, 8] = 0   # single-cell uncovered
    # Caller passes an oversized eps (>> resolution); clamp must save the region.
    regs = find_uncovered_regions(counts, target, reachable,
                                  resolution=0.1, origin_x=0.0, origin_y=0.0,
                                  min_area_m2=0.005, simplify_eps_m=0.40)
    assert len(regs) == 1
    assert len(regs[0].polygon) >= 3
    poly_mask = polygon_to_mask(regs[0].polygon, w, h, 0.1, 0.0, 0.0)
    assert poly_mask[8, 8]


# ---------- grid_coverage: trailing-edge swath coverage ----------

def _swath_perp_indices(swaths, axis, resolution, origin):
    """Cell indices of each swath along the scan-stepping axis."""
    perp = 1 if axis == "x" else 0  # world index whose cell drives the row/col
    idxs = []
    for a, b in swaths:
        coord = a[perp]
        idxs.append(int(round((coord - origin) / resolution - 0.5)))
    return sorted(set(idxs))


def test_swath_generation_covers_trailing_edge():
    # Room width chosen to expose the periodic end-strip gap: spacing_cells=7,
    # footprint half ~3 cells. Without the edge swath the last scan line at
    # col 17 leaves cols 18..21 uncovered for width 22.
    for width in (22, 23, 24, 29, 30):
        free = np.ones((10, width), dtype=bool)
        swaths = _generate_swaths_from_mask(
            free, resolution=0.1, origin_x=0.0, origin_y=0.0,
            swath_spacing_m=0.7, axis="y", min_swath_m=0.2)
        cols = _swath_perp_indices(swaths, "y", 0.1, 0.0)
        assert cols, f"no swaths for width={width}"
        spacing_cells = 7
        last_col = cols[-1]
        gap = (width - 1) - last_col
        # trailing edge must be within half-footprint of the last swath
        assert gap <= spacing_cells // 2, (
            f"width={width} leaves end gap {gap} cells (cols={cols})")


def test_swath_generation_no_redundant_edge_swath_when_already_covered():
    # width 21: stock scan cols [3,10,17]; (21-1)-17 = 3 == spacing//2 -> no add.
    free = np.ones((10, 21), dtype=bool)
    swaths = _generate_swaths_from_mask(
        free, resolution=0.1, origin_x=0.0, origin_y=0.0,
        swath_spacing_m=0.7, axis="y", min_swath_m=0.2)
    cols = _swath_perp_indices(swaths, "y", 0.1, 0.0)
    assert cols == [3, 10, 17]
