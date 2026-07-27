from td25a_robot_ui.algorithms.path_chunks import (
    choose_path_chunk,
    remap_hard_stops_after_splice,
    select_safe_rejoin_index,
    select_same_sweep_rejoin_index,
)


def _line(length_m: float, step_m: float = 0.1):
    n = int(length_m / step_m) + 1
    return [(i * step_m, 0.0) for i in range(n)]


def test_chunk_limits_long_coverage_path_to_local_window():
    path = _line(10.0)

    chunk = choose_path_chunk(path, start_idx=0, max_length_m=2.0)

    assert chunk.start_idx == 0
    assert 18 <= chunk.end_idx <= 21
    assert chunk.next_start_idx == chunk.end_idx
    assert chunk.end_idx < len(path) - 1


def test_chunk_finishes_when_remaining_path_is_short():
    path = _line(1.2)

    chunk = choose_path_chunk(path, start_idx=0, max_length_m=2.0)

    assert chunk.end_idx == len(path) - 1
    assert chunk.next_start_idx == len(path) - 1


def test_chunk_skips_duplicate_points_and_still_advances():
    path = [(0.0, 0.0), (0.0, 0.0)] + _line(3.0)

    chunk = choose_path_chunk(path, start_idx=0, max_length_m=1.0)

    assert chunk.end_idx > 2
    assert chunk.next_start_idx == chunk.end_idx


def test_chunk_recouples_to_current_pose_ahead_of_old_start():
    path = _line(10.0)

    chunk = choose_path_chunk(
        path,
        start_idx=10,
        max_length_m=2.0,
        current_xy=(3.2, 0.05),
        max_snap_m=0.40,
        search_length_m=4.0,
    )

    assert 30 <= chunk.start_idx <= 33
    assert chunk.start_distance_m < 0.10
    assert chunk.end_idx > chunk.start_idx
    assert chunk.next_start_idx == chunk.end_idx


def test_chunk_keeps_old_start_when_current_pose_is_not_near_remaining_path():
    path = _line(10.0)

    chunk = choose_path_chunk(
        path,
        start_idx=10,
        max_length_m=2.0,
        current_xy=(3.2, 1.0),
        max_snap_m=0.40,
        search_length_m=4.0,
    )

    assert chunk.start_idx == 10
    assert chunk.start_distance_m > 0.40


def test_partitioned_chunk_stops_at_lane_boundary_before_length_limit():
    path = _line(10.0)

    chunk = choose_path_chunk(
        path, start_idx=0, max_length_m=8.0,
        hard_stop_indices=[20, 55, 90])

    assert chunk.start_idx == 0
    assert chunk.end_idx == 20
    assert chunk.next_start_idx == 20
    assert chunk.ends_at_hard_stop


def test_partitioned_recouple_cannot_snap_across_next_lane_boundary():
    # The current pose is spatially closest to a later return lane.  A hard
    # stop at the first lane end bounds recoupling as well as chunk length.
    first_lane = [(i * 0.1, 0.0) for i in range(51)]
    next_lane = [(5.0 - i * 0.1, 0.5) for i in range(51)]
    path = first_lane + next_lane

    chunk = choose_path_chunk(
        path, start_idx=0, max_length_m=30.0,
        current_xy=(0.3, 0.49), max_snap_m=0.8,
        search_length_m=30.0, hard_stop_indices=[50, 101])

    assert chunk.start_idx <= 4
    assert chunk.end_idx == 50
    assert chunk.ends_at_hard_stop


def test_next_partitioned_chunk_ignores_the_stop_it_starts_on():
    path = _line(10.0)

    chunk = choose_path_chunk(
        path, start_idx=20, max_length_m=8.0,
        hard_stop_indices=[20, 55, 90])

    assert chunk.start_idx == 20
    assert chunk.end_idx == 55
    assert chunk.ends_at_hard_stop


def test_partitioned_detour_preserves_future_stops_and_stops_at_rejoin():
    # Original points 40..99 survive after a 15-point detour whose last point
    # duplicates point 40: 15 + 60 - 1 = 74 points, retained tail starts at 14.
    stops = remap_hard_stops_after_splice(
        coverage_length=100,
        spliced_length=74,
        reroute_idx=40,
        hard_stop_indices=[20, 50, 75, 99],
    )

    assert stops == [14, 24, 49, 73]


def test_short_recouple_window_cannot_snap_to_nearby_future_lane():
    # Two parallel lanes are only 0.5 m apart. A long forward search prefers the
    # spatially closest point on the much later return lane; the fill-skip guard's
    # 1.5 m window must keep execution on the requested lane.
    outbound = [(i * 0.1, 0.0) for i in range(101)]
    returning = [(10.0 - i * 0.1, 0.5) for i in range(101)]
    path = outbound + returning
    current = (0.5, 0.49)

    long_window = choose_path_chunk(
        path, start_idx=0, max_length_m=2.0, current_xy=current,
        max_snap_m=1.75, search_length_m=30.0)
    bounded = choose_path_chunk(
        path, start_idx=0, max_length_m=2.0, current_xy=current,
        max_snap_m=1.75, search_length_m=1.5)

    assert long_window.start_idx > 100
    assert bounded.start_idx <= 15


def test_perimeter_rejoin_scans_past_obstacle_instead_of_fixed_skip():
    path = _line(8.0)

    def clearance_at(point):
        # A 2.4 m-wide dynamic obstacle occupies the boundary.  A fixed 1.2 m
        # jump would still land in it; the rejoin selector must continue to the
        # first confirmed-clear point after the obstacle.
        return 0.10 if point[0] < 2.4 else 0.80

    idx = select_safe_rejoin_index(
        path, 0, clearance_at, min_skip_m=0.8, max_skip_m=4.0,
        required_clear_m=0.55)

    assert idx is not None
    assert path[idx][0] >= 2.4


def test_perimeter_rejoin_rejects_unknown_clearance():
    path = _line(4.0)

    assert select_safe_rejoin_index(
        path, 0, lambda _point: None, min_skip_m=0.8,
        max_skip_m=3.0, required_clear_m=0.55) is None


def test_perimeter_rejoin_is_bounded_on_closed_or_self_near_path():
    path = _line(10.0)

    # The first clear point lies beyond the allowed forward arc window.  It
    # must not silently jump across most of a perimeter ring.
    idx = select_safe_rejoin_index(
        path, 0, lambda p: 0.8 if p[0] >= 7.0 else 0.1,
        min_skip_m=0.8, max_skip_m=4.0, required_clear_m=0.55)

    assert idx is None


def test_fill_rejoin_stays_on_same_straight_sweep():
    path = _line(8.0)

    def clearance_at(point):
        return 0.1 if point[0] < 2.0 else 0.8

    idx = select_same_sweep_rejoin_index(
        path, 10, clearance_at, min_skip_m=0.8, max_skip_m=3.0,
        required_clear_m=0.45, max_turn_deg=55.0)
    assert idx is not None
    assert 20 <= idx <= 21


def test_fill_rejoin_refuses_to_cross_lane_turn():
    # The obstacle covers the rest of the horizontal sweep.  A spatial-nearest
    # search could select the adjacent connector/return lane; arc selection
    # must stop before the 90-degree turn.
    path = [(i * 0.1, 0.0) for i in range(11)]
    path += [(1.0, i * 0.1) for i in range(1, 11)]

    def clearance_at(point):
        return 0.1 if point[1] == 0.0 else 1.0

    assert select_same_sweep_rejoin_index(
        path, 6, clearance_at, min_skip_m=0.3, max_skip_m=2.0,
        required_clear_m=0.45, max_turn_deg=55.0) is None


def test_fill_rejoin_rejects_unknown_clearance():
    assert select_same_sweep_rejoin_index(
        _line(4.0), 5, lambda _point: None,
        min_skip_m=0.5, max_skip_m=2.0) is None
