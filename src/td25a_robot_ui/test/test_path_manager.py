import math
import time

import pytest

from td25a_robot_ui.algorithms.path_manager import (
    BlockageDetector,
    ExecutionState,
    PathManager,
    PathManagerConfig,
)


def _line(length=100.0, step=0.1, y=0.0):
    return [(i * step, y) for i in range(int(length / step) + 1)]


def test_complete_path_is_cached_and_window_uses_arc_length():
    manager = PathManager(PathManagerConfig(window_length=50.0))
    version = manager.set_complete_path([(0.0, 0.0), (0.2, 0.0), (5.0, 0.0), (100.0, 0.0)])
    manager.start_tracking()
    manager.update_progress((10.0, 0.02, 0.0), speed=0.3)
    manager.update_progress((10.01, 0.02, 0.0), speed=0.3)
    window = manager.build_window()

    assert version == 1
    assert manager.total_length == pytest.approx(100.0)
    assert window.start_s == pytest.approx(9.0, abs=0.05)
    assert window.end_s == pytest.approx(60.0, abs=0.05)
    assert window.poses[0].x == pytest.approx(9.0, abs=0.05)
    assert window.poses[-1].x == pytest.approx(60.0, abs=0.05)


def test_progress_does_not_jump_to_close_parallel_future_lane():
    first = [(x * 0.1, 0.0) for x in range(101)]
    turn = [(10.0, y * 0.1) for y in range(1, 6)]
    second = [(10.0 - x * 0.1, 0.5) for x in range(1, 101)]
    path = first + turn + second
    manager = PathManager()
    manager.set_complete_path(path, hard_stop_indices=[100, 105])
    manager.start_tracking()
    manager.current_progress_s = 4.5
    manager.current_progress_index = 45

    projection = manager.update_progress((5.0, 0.46, 0.0), speed=0.3)

    assert projection is not None
    assert projection.segment_id == 0
    assert manager.current_progress_s < 6.0


def test_heading_gate_does_not_fall_back_while_robot_is_moving():
    manager = PathManager(PathManagerConfig(max_heading_error=math.radians(45.0)))
    manager.set_complete_path(_line(20.0))
    manager.start_tracking()

    projection = manager.update_progress((0.2, 0.0, math.pi), speed=0.2)

    assert projection is None
    assert manager.current_progress_s == 0.0


def test_progress_is_monotonic_and_small_reverse_projection_does_not_reopen_path():
    manager = PathManager()
    manager.set_complete_path(_line(20.0))
    manager.start_tracking()
    manager.update_progress((3.0, 0.0, 0.0), speed=0.2)
    manager.update_progress((3.01, 0.0, 0.0), speed=0.2)
    before = manager.current_progress_s
    raw = manager.update_progress((2.7, 0.0, 0.0), speed=0.0)

    assert raw is not None
    assert raw.s <= before
    assert manager.current_progress_s == pytest.approx(before)
    assert manager.current_progress_index >= 29


def test_large_progress_jump_requires_consecutive_confirmation():
    config = PathManagerConfig(
        progress_search_forward_distance=5.0,
        progress_jump_confirm_distance=0.8,
        progress_jump_confirm_frames=2,
    )
    manager = PathManager(config)
    manager.set_complete_path(_line(20.0))
    manager.start_tracking()

    first = manager.update_progress((2.0, 0.0, 0.0), speed=0.2)
    assert first is not None and not first.confirmed
    assert manager.current_progress_s == 0.0

    second = manager.update_progress((2.02, 0.0, 0.0), speed=0.2)
    assert second is not None and second.confirmed
    assert manager.current_progress_s == pytest.approx(2.02, abs=0.06)


def test_window_is_continuous_and_pose_yaw_follows_tangent():
    manager = PathManager(PathManagerConfig(window_length=20.0, min_window_length=5.0))
    manager.set_complete_path([(0, 0), (5, 0), (5, 5), (10, 5)])
    manager.start_tracking()
    manager.current_progress_s = 4.0
    window = manager.build_window(lookahead_distance=10.0)

    gaps = [
        math.hypot(b.x - a.x, b.y - a.y)
        for a, b in zip(window.poses, window.poses[1:])
    ]
    assert gaps and max(gaps) <= 5.0
    assert window.poses[0].yaw == pytest.approx(0.0)
    corner = next(pose for pose in window.poses if pose.x == pytest.approx(5.0) and pose.y == pytest.approx(0.0))
    assert corner.yaw == pytest.approx(math.pi / 2.0)


def test_folded_window_endpoint_is_not_allowed_beside_current_pose():
    # 50m desired endpoint returns to x=0.2 beside the start, while nearby
    # arc-length candidates on the upper leg are safely farther away.
    path = [(0.0, 0.0), (25.0, 0.0), (25.0, 0.2), (0.2, 0.2), (0.2, 5.0)]
    manager = PathManager(PathManagerConfig(
        window_length=50.0,
        min_window_length=20.0,
        max_window_length=60.0,
        min_goal_endpoint_distance=0.75,
    ))
    manager.set_complete_path(path)
    manager.start_tracking()

    window = manager.build_window()
    endpoint_distance = math.hypot(window.poses[-1].x, window.poses[-1].y)

    assert window.end_s != pytest.approx(50.0)
    assert endpoint_distance >= 0.75


def test_mppi_reference_honours_five_metre_input_and_costmap_cap():
    manager = PathManager(PathManagerConfig(
        mppi_reference_length=5.0,
        mppi_reference_min_length=5.0,
        mppi_reference_max_length=5.0,
        mppi_prediction_horizon=7.968,
        mppi_safety_margin=0.5,
        mppi_costmap_radius=5.5,
        mppi_costmap_margin=0.5,
    ))
    manager.set_complete_path(_line(100.0))
    manager.start_tracking()
    manager.current_progress_s = 10.0

    assert manager.effective_mppi_reference_length(speed=0.5) == pytest.approx(5.0)
    reference = manager.build_mppi_reference(speed=0.5)
    assert reference.end_s - manager.current_progress_s == pytest.approx(5.0)


def test_dynamic_window_shrinks_smoothly_for_turns_and_obstacles():
    config = PathManagerConfig(
        window_length=50.0,
        min_window_length=20.0,
        max_window_length=60.0,
        window_length_change_rate=5.0,
    )
    manager = PathManager(config)
    manager.set_complete_path([(0, 0), (2, 0), (2, 2), (4, 2), (4, 4), (30, 4)])
    manager.start_tracking()

    baseline = manager.adapt_window_length(now=0.0, speed=0.3)
    after_one_second = manager.adapt_window_length(
        now=1.0, speed=0.1, obstacle_density=0.8, narrow_area=True)

    assert after_one_second >= baseline - 5.0 - 1e-6
    assert config.min_window_length <= after_one_second <= config.max_window_length


def test_window_update_is_distance_or_remaining_driven_not_republished_every_tick():
    manager = PathManager(PathManagerConfig(path_update_rate=5.0, path_update_distance=0.75))
    manager.set_complete_path(_line(100.0))
    manager.start_tracking()
    first = manager.build_window()
    assert manager.needs_window_update(0.0, first)
    manager.mark_window_published(first, 0.0)
    assert not manager.needs_window_update(0.05, manager.build_window())

    manager.current_progress_s = 0.8
    assert manager.needs_window_update(0.10, manager.build_window())


def test_blockage_detector_distinguishes_rotation_from_true_blockage():
    config = PathManagerConfig(blocked_timeout=5.0, obstacle_confirmation_time=2.0)
    detector = BlockageDetector(config)
    detector.reset(now=0.0, progress_s=10.0)

    turning = detector.update(
        now=6.0,
        valid_path=True,
        distance_to_goal=20.0,
        linear_speed=0.0,
        angular_speed=0.3,
        progress_s=10.0,
        commanded_speed=0.2,
        path_obstructed=True,
    )
    assert not turning.blocked

    detector.reset(now=0.0, progress_s=10.0)
    detector.update(
        now=1.0,
        valid_path=True,
        distance_to_goal=20.0,
        linear_speed=0.0,
        angular_speed=0.0,
        progress_s=10.0,
        commanded_speed=0.2,
        path_obstructed=True,
    )
    blocked = detector.update(
        now=7.0,
        valid_path=True,
        distance_to_goal=20.0,
        linear_speed=0.0,
        angular_speed=0.0,
        progress_s=10.0,
        commanded_speed=0.2,
        path_obstructed=True,
    )
    assert blocked.blocked
    assert "persistent_obstacle" in blocked.reason


def test_ranges_versioning_and_completion_gate():
    manager = PathManager(PathManagerConfig(skipped_segment_max_length=3.0))
    manager.set_complete_path(_line(20.0), reason="initial")
    manager.start_tracking()
    manager.mark_completed(0.0, 4.0)
    manager.mark_blocked(5.0, 6.0, "chair")
    manager.mark_skipped(5.0, 6.0, "chair")
    manager.set_detour([(4.0, 0.0), (4.5, 0.4), (6.0, 0.0)], reason="detour")

    assert manager.path_version == 2
    assert manager.active_execution_path == manager.detour_path
    assert manager.pending_recovery_areas
    with pytest.raises(ValueError):
        manager.mark_skipped(7.0, 11.0, "too_long")

    manager.current_progress_s = manager.total_length
    manager.set_state(ExecutionState.TRACKING)
    assert not manager.completion_ready(coverage_ratio=0.94)
    assert manager.completion_ready(coverage_ratio=0.96, reachable_pending_areas=0)

    manager.resume_complete_path()
    assert manager.path_version == 3
    assert manager.active_execution_path == manager.complete_path
    assert not manager.detour_path


def test_duplicate_points_are_removed_without_shifting_hard_stop_semantics():
    manager = PathManager()
    manager.set_complete_path(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        hard_stop_indices=[2],
    )

    assert manager.complete_path == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    assert manager.hard_stop_indices == [1]


def test_thousands_of_points_projection_has_bounded_runtime():
    manager = PathManager()
    manager.set_complete_path(_line(500.0, step=0.05))
    manager.start_tracking()
    started = time.perf_counter()
    for index in range(400):
        x = index * 0.02
        manager.update_progress((x, 0.01, 0.0), speed=0.3)
    elapsed = time.perf_counter() - started

    # This is intentionally generous for shared CI and Jetson debug builds;
    # the search remains local and must not scale as a full-path scan.
    assert elapsed < 1.0
