import numpy as np

from td25a_robot_ui.algorithms.free_space import (
    extract_clean_field_result,
    polygon_to_mask,
    recover_saved_map_known_free_mask,
)
from td25a_robot_ui.algorithms.nav2_params import _footprint_inscribed_radius
from td25a_robot_ui.algorithms.grid_coverage import (
    build_coverage_free_mask,
    plan_lawnmower_coverage,
)


def _seed_world(row, col, resolution=1.0):
    return ((col + 0.5) * resolution, (row + 0.5) * resolution)


def test_polygon_footprint_uses_inscribed_clearance_not_inflation_radius():
    footprint = "[[0.33, 0.34], [0.33, -0.34], [-0.72, -0.34], [-0.72, 0.34]]"

    assert abs(_footprint_inscribed_radius(footprint) - 0.33) < 1e-9


def test_robot_seed_selects_its_reachable_room_not_larger_disconnected_room():
    grid = np.full((40, 80), 100, dtype=np.int8)
    grid[2:20, 2:25] = 0       # smaller room containing the robot
    grid[2:36, 40:78] = 0      # much larger but disconnected room

    result = extract_clean_field_result(
        grid.tobytes(), 80, 40, 1.0, 0.0, 0.0,
        robot_radius_m=1.0,
        seed_world=_seed_world(10, 10),
        min_cluster_cells=1,
    )

    assert result.polygon
    mask = polygon_to_mask(result.polygon, 80, 40, 1.0, 0.0, 0.0)
    assert mask[10, 10]
    assert not mask[10, 50]
    assert result.reachable_area_m2 < 500.0


def test_eroded_seed_snaps_inside_same_room_instead_of_global_largest():
    grid = np.full((36, 72), 100, dtype=np.int8)
    grid[2:24, 2:24] = 0
    grid[2:34, 36:70] = 0

    result = extract_clean_field_result(
        grid.tobytes(), 72, 36, 1.0, 0.0, 0.0,
        robot_radius_m=2.0,
        seed_world=_seed_world(10, 2),  # free, but removed by clearance erosion
        max_seed_snap_m=3.0,
        min_cluster_cells=1,
    )

    assert result.polygon
    assert 0.0 < result.seed_snap_m <= 3.0
    mask = polygon_to_mask(result.polygon, 72, 36, 1.0, 0.0, 0.0)
    assert mask[10, 8]
    assert not mask[10, 50]


def test_seed_room_without_clearance_fails_closed_not_to_other_room():
    grid = np.full((36, 72), 100, dtype=np.int8)
    grid[5:8, 3:6] = 0         # robot room disappears after 2-cell erosion
    grid[2:34, 36:70] = 0      # tempting large fallback component

    result = extract_clean_field_result(
        grid.tobytes(), 72, 36, 1.0, 0.0, 0.0,
        robot_radius_m=2.0,
        seed_world=_seed_world(6, 4),
        max_seed_snap_m=4.0,
        min_cluster_cells=1,
    )

    assert result.polygon == []
    assert result.failure_reason == "seed_component_has_no_safe_clearance"


def test_unclosed_free_map_is_bounded_by_structure_envelope():
    grid = np.zeros((64, 80), dtype=np.int8)
    # Four structural clusters define the observed facility support, but gaps
    # leave the raw free space connected all the way to the map border.
    for ys, xs in ((slice(10, 14), slice(12, 16)),
                   (slice(10, 14), slice(60, 64)),
                   (slice(48, 52), slice(12, 16)),
                   (slice(48, 52), slice(60, 64))):
        grid[ys, xs] = 100

    result = extract_clean_field_result(
        grid.tobytes(), 80, 64, 1.0, 0.0, 0.0,
        robot_radius_m=1.0,
        seed_world=_seed_world(30, 38),
        min_cluster_cells=1,
    )

    assert result.polygon
    assert result.used_open_guard
    assert result.reachable_area_m2 < 0.65 * grid.size
    mask = polygon_to_mask(result.polygon, 80, 64, 1.0, 0.0, 0.0)
    assert mask[30, 38]
    assert not mask[2, 2]


def test_saved_map_palette_recovers_gray_unknown_cells(tmp_path):
    from PIL import Image

    image = np.full((6, 8), 205, dtype=np.uint8)
    image[1:4, 2:6] = 254
    image[2, 3] = 0
    image_path = tmp_path / "map.png"
    yaml_path = tmp_path / "map.yaml"
    Image.fromarray(image, mode="L").save(image_path)
    yaml_path.write_text(
        "image: map.png\nresolution: 0.1\norigin: [0, 0, 0]\n"
        "occupied_thresh: 0.65\nfree_thresh: 0.25\n",
        encoding="utf-8",
    )

    known, count = recover_saved_map_known_free_mask(str(yaml_path), 8, 6)

    assert known is not None
    assert count == int((image == 205).sum())
    assert np.array_equal(known, np.flipud(image >= 250))


def test_fill_islands_are_both_planned_when_travel_corridor_connects_them():
    """A doorway may be safe for the chassis but too narrow for fill lanes."""
    resolution = 0.10
    grid = np.full((70, 120), 100, dtype=np.int8)
    grid[8:62, 5:48] = 0       # left cleaning room
    grid[8:62, 72:115] = 0     # right cleaning room
    grid[30:40, 48:72] = 0     # 1.0m doorway/corridor
    robot = _seed_world(35, 20, resolution)

    plan = plan_lawnmower_coverage(
        data=grid.tobytes(), width=120, height=70,
        resolution=resolution, origin_x=0.0, origin_y=0.0,
        robot_radius_m=0.50,       # fill clearance closes the corridor
        travel_radius_m=0.30,      # physical footprint can traverse it
        swath_spacing_m=0.80,
        robot_world=robot,
        axis="x", no_rotate=True,
        min_swath_m=0.45, path_step_m=0.10,
    )

    assert plan.path
    assert any(max(a[0], b[0]) < 4.8 for a, b in plan.swaths)
    assert any(min(a[0], b[0]) > 7.2 for a, b in plan.swaths)

    travel = build_coverage_free_mask(
        grid.tobytes(), 120, 70, resolution, 0.30)
    for x, y in plan.path:
        col = int(np.floor(x / resolution))
        row = int(np.floor(y / resolution))
        assert 0 <= row < 70 and 0 <= col < 120
        assert travel[row, col]
