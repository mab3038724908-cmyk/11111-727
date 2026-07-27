import math

import numpy as np

from td25a_robot_ui.algorithms.grid_coverage import (
    build_coverage_free_mask,
    plan_lawnmower_coverage,
    _smooth_path_corners,
)


def _open_room(width=60, height=40):
    grid = np.zeros((height, width), dtype=np.int8)
    grid[0, :] = 100
    grid[-1, :] = 100
    grid[:, 0] = 100
    grid[:, -1] = 100
    return grid


def _world_to_cell(x, y, resolution=0.1, origin_x=0.0, origin_y=0.0):
    return int((y - origin_y) / resolution), int((x - origin_x) / resolution)


def _max_turn_deg(path):
    max_turn = 0.0
    for a, b, c in zip(path, path[1:], path[2:]):
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        max_turn = max(max_turn, math.degrees(math.acos(dot)))
    return max_turn


def _first_segment_heading(path):
    for a, b in zip(path, path[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        if math.hypot(dx, dy) > 0.04:
            return math.atan2(dy, dx)
    return 0.0


def test_lawnmower_path_starts_at_robot_and_stays_continuous():
    grid = _open_room()
    result = plan_lawnmower_coverage(
        data=grid.tobytes(),
        width=grid.shape[1],
        height=grid.shape[0],
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
        robot_radius_m=0.2,
        swath_spacing_m=0.6,
        robot_world=(2.8, 1.8),
        axis="x",
    )

    assert result.path
    assert result.swaths
    assert math.hypot(result.path[0][0] - 2.8, result.path[0][1] - 1.8) < 0.15
    assert max(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(result.path, result.path[1:])
    ) <= 0.16


def test_lawnmower_start_direction_prefers_robot_heading_when_split_mid_swath():
    grid = _open_room(width=80, height=50)
    robot_yaw = 0.0
    result = plan_lawnmower_coverage(
        data=grid.tobytes(),
        width=grid.shape[1],
        height=grid.shape[0],
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
        robot_radius_m=0.2,
        swath_spacing_m=0.6,
        robot_world=(3.0, 1.85),
        robot_yaw=robot_yaw,
        axis="x",
    )

    assert result.path
    first_heading = _first_segment_heading(result.path)
    heading_error = abs(math.atan2(
        math.sin(first_heading - robot_yaw),
        math.cos(first_heading - robot_yaw),
    ))
    assert heading_error <= math.radians(90.0)


def test_lawnmower_mid_swath_start_does_not_create_backtracking_cusp():
    grid = _open_room(width=80, height=50)
    result = plan_lawnmower_coverage(
        data=grid.tobytes(),
        width=grid.shape[1],
        height=grid.shape[0],
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
        robot_radius_m=0.2,
        swath_spacing_m=0.6,
        robot_world=(3.0, 1.85),
        robot_yaw=0.0,
        axis="x",
    )

    assert result.path
    assert _max_turn_deg(result.path) <= 95.0


def test_corner_smoothing_removes_short_out_and_back_spur():
    free = np.ones((40, 40), dtype=bool)
    path = [
        (1.0, 1.0),
        (1.4, 1.0),
        (1.8, 1.0),
        (1.6, 1.0),
        (1.4, 1.0),
        (1.4, 1.4),
    ]

    smoothed = _smooth_path_corners(
        path,
        free_mask=free,
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
        turn_radius_m=0.3,
        path_step_m=0.1,
    )

    assert _max_turn_deg(smoothed) <= 95.0
    assert all(x <= 1.61 for x, _ in smoothed)


def test_lawnmower_path_rounds_lane_changes_for_controller_tracking():
    grid = _open_room(width=80, height=50)
    result = plan_lawnmower_coverage(
        data=grid.tobytes(),
        width=grid.shape[1],
        height=grid.shape[0],
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
        robot_radius_m=0.2,
        swath_spacing_m=0.6,
        robot_world=(1.0, 1.0),
        axis="x",
    )

    assert result.path
    assert _max_turn_deg(result.path) <= 70.0


def test_lawnmower_path_never_enters_inflated_obstacles():
    grid = _open_room()
    grid[8:32, 29:32] = 100
    resolution = 0.1
    robot_radius = 0.2

    result = plan_lawnmower_coverage(
        data=grid.tobytes(),
        width=grid.shape[1],
        height=grid.shape[0],
        resolution=resolution,
        origin_x=0.0,
        origin_y=0.0,
        robot_radius_m=robot_radius,
        swath_spacing_m=0.6,
        robot_world=(1.0, 1.0),
        axis="x",
    )
    free = build_coverage_free_mask(
        grid.tobytes(), grid.shape[1], grid.shape[0], resolution, robot_radius)

    assert result.path
    for x, y in result.path:
        cy, cx = _world_to_cell(x, y, resolution)
        assert free[cy, cx]


def test_disconnected_selected_zone_does_not_fallback_to_straight_connector():
    grid = _open_room(width=80, height=50)
    grid[1:-1, 39:41] = 100
    resolution = 0.1
    zone = [(5.0, 1.0), (7.0, 1.0), (7.0, 4.0), (5.0, 4.0)]

    result = plan_lawnmower_coverage(
        data=grid.tobytes(),
        width=grid.shape[1],
        height=grid.shape[0],
        resolution=resolution,
        origin_x=0.0,
        origin_y=0.0,
        robot_radius_m=0.2,
        swath_spacing_m=0.6,
        robot_world=(1.0, 2.5),
        clip_polygon=zone,
        axis="x",
    )

    assert result.swaths
    assert result.path == []


def test_lawnmower_path_respects_blocked_polygons():
    grid = _open_room()
    resolution = 0.1
    blocked = [[(2.5, 1.0), (3.5, 1.0), (3.5, 3.0), (2.5, 3.0)]]

    result = plan_lawnmower_coverage(
        data=grid.tobytes(),
        width=grid.shape[1],
        height=grid.shape[0],
        resolution=resolution,
        origin_x=0.0,
        origin_y=0.0,
        robot_radius_m=0.2,
        swath_spacing_m=0.6,
        robot_world=(1.0, 1.0),
        blocked_polygons=blocked,
        axis="x",
    )

    assert result.path
    for x, y in result.path:
        assert not (2.5 <= x <= 3.5 and 1.0 <= y <= 3.0)


def test_selected_zone_plan_keeps_robot_pose_as_path_start_when_outside_zone():
    grid = _open_room(width=80, height=50)
    resolution = 0.1
    zone = [(4.0, 1.0), (7.0, 1.0), (7.0, 4.0), (4.0, 4.0)]
    robot = (1.0, 2.5)

    result = plan_lawnmower_coverage(
        data=grid.tobytes(),
        width=grid.shape[1],
        height=grid.shape[0],
        resolution=resolution,
        origin_x=0.0,
        origin_y=0.0,
        robot_radius_m=0.2,
        swath_spacing_m=0.6,
        robot_world=robot,
        clip_polygon=zone,
        axis="x",
    )

    assert result.path
    assert math.hypot(result.path[0][0] - robot[0], result.path[0][1] - robot[1]) < 0.15
    assert any(4.0 <= x <= 7.0 and 1.0 <= y <= 4.0 for x, y in result.path)
    assert all(
        (4.0 <= a[0][0] <= 7.0 and 1.0 <= a[0][1] <= 4.0
         and 4.0 <= a[1][0] <= 7.0 and 1.0 <= a[1][1] <= 4.0)
        for a in result.swaths
    )


def test_selected_zone_can_be_far_from_robot_pose():
    grid = _open_room(width=220, height=70)
    resolution = 0.1
    zone = [(16.0, 1.0), (20.0, 1.0), (20.0, 5.0), (16.0, 5.0)]
    robot = (1.0, 3.0)

    result = plan_lawnmower_coverage(
        data=grid.tobytes(),
        width=grid.shape[1],
        height=grid.shape[0],
        resolution=resolution,
        origin_x=0.0,
        origin_y=0.0,
        robot_radius_m=0.2,
        swath_spacing_m=0.6,
        robot_world=robot,
        clip_polygon=zone,
        axis="x",
    )

    assert result.path
    assert math.hypot(result.path[0][0] - robot[0], result.path[0][1] - robot[1]) < 0.15
    assert any(16.0 <= x <= 20.0 and 1.0 <= y <= 5.0 for x, y in result.path)
    assert all(
        (16.0 <= a[0][0] <= 20.0 and 1.0 <= a[0][1] <= 5.0
         and 16.0 <= a[1][0] <= 20.0 and 1.0 <= a[1][1] <= 5.0)
        for a in result.swaths
    )
