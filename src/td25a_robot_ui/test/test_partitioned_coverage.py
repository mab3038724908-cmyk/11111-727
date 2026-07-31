import math

import numpy as np
import td25a_robot_ui.algorithms.grid_coverage as grid_coverage_module

from td25a_robot_ui.algorithms.grid_coverage import (
    PartitionedCoveragePlan,
    _boustrophedon_cells,
    _canonical_swath_multiset,
    _explicit_path_yaw_pairs,
    _forward_brush_cleaning_mask,
    _generate_component_swaths,
    _polyline_cleaning_mask,
    _polylines_cleaning_mask,
    _strictly_prefer_exit_candidate,
    coverage_swath_spacing,
    plan_partitioned_coverage,
    validate_explicit_pose_transition,
    validate_explicit_yaw_polyline,
)


RESOLUTION = 0.10


def _world(row, col):
    return ((col + 0.5) * RESOLUTION, (row + 0.5) * RESOLUTION)


def test_batched_independent_polyline_mask_matches_union_without_false_links():
    paths = [
        [(1.05, 1.05), (5.05, 1.05)],
        [(1.05, 5.05), (5.05, 5.05)],
    ]
    expected = np.zeros((70, 70), dtype=bool)
    for path in paths:
        expected |= _polyline_cleaning_mask(
            expected.shape, path, RESOLUTION, 0.0, 0.0, 0.12)

    actual = _polylines_cleaning_mask(
        expected.shape, paths, RESOLUTION, 0.0, 0.0, 0.12)

    assert np.array_equal(actual, expected)
    assert not actual[30, 30]


def _plan(grid, robot, selection_policy="strict"):
    return plan_partitioned_coverage(
        data=grid.tobytes(),
        width=grid.shape[1],
        height=grid.shape[0],
        resolution=RESOLUTION,
        origin_x=0.0,
        origin_y=0.0,
        robot_world=robot,
        robot_yaw=0.0,
        swath_spacing_m=coverage_swath_spacing(0.70, 0.75),
        path_step_m=0.10,
        min_swath_m=0.35,
        min_region_area_m2=3.0,
        max_regions=12,
        clean_width_m=0.70,
        selection_policy=selection_policy,
    )


def test_open_square_is_one_complete_body_safe_region_above_95_percent():
    grid = np.full((110, 140), 100, dtype=np.int8)
    grid[8:102, 8:132] = 0

    plan = _plan(grid, _world(25, 25))

    assert plan.path
    assert len(plan.regions) == 1
    assert plan.visit_order == [0]
    assert plan.footprint_valid
    assert plan.footprint_violation_count == 0
    assert plan.path_continuous
    assert plan.reachable_coverage_ratio >= 0.95
    assert plan.actual_brush_coverage_ratio >= 0.95
    assert plan.serviceable_target_mask is not None
    assert plan.serviceable_target_mask.shape == grid.shape
    assert plan.coverage_complete


def test_small_short_side_room_is_excluded_from_cleanable_target():
    grid = np.full((130, 170), 100, dtype=np.int8)
    grid[10:120, 10:105] = 0
    grid[60:68, 105:125] = 0
    grid[48:76, 125:150] = 0

    plan = _plan(grid, _world(25, 25))

    assert plan.coverage_complete
    assert plan.discarded_small_component_count >= 1
    assert plan.discarded_small_area_m2 > 0.0
    assert not plan.serviceable_target_mask[62, 137]
    assert len(plan.regions) == 1


def test_long_narrow_side_area_is_kept_even_when_center_area_is_small():
    grid = np.full((130, 200), 100, dtype=np.int8)
    grid[10:120, 10:105] = 0
    grid[60:68, 105:125] = 0
    grid[54:74, 125:170] = 0

    plan = _plan(grid, _world(25, 25))

    assert plan.coverage_complete
    assert plan.discarded_small_component_count == 0
    assert len(plan.regions) == 2
    assert any(
        max(start[0], finish[0]) > 12.5
        for start, finish in plan.swaths
    )


def test_large_l_is_decomposed_and_each_region_finishes_before_next():
    grid = np.full((160, 160), 100, dtype=np.int8)
    grid[10:150, 10:72] = 0
    grid[88:150, 10:150] = 0

    plan = _plan(grid, _world(120, 25))

    assert plan.path
    assert len(plan.regions) >= 2
    assert len(plan.visit_order) == len(plan.regions)
    assert plan.footprint_valid
    assert plan.path_continuous

    # Once a region's segments end, it is never revisited.  BCD fill remains
    # room-scoped, while one red perimeter follows every fill phase.
    compressed = []
    for segment in plan.segments:
        if not compressed or compressed[-1] != segment.region_id:
            compressed.append(segment.region_id)
    assert compressed == plan.visit_order
    assert all(any(
        segment.kind == "fill" and segment.region_id == region_id
        for segment in plan.segments)
        for region_id in plan.visit_order)
    perimeter_indices = [
        index for index, segment in enumerate(plan.segments)
        if segment.kind == "perimeter"
    ]
    fill_indices = [
        index for index, segment in enumerate(plan.segments)
        if segment.kind == "fill"
    ]
    assert len(perimeter_indices) == 1
    assert perimeter_indices[0] > max(fill_indices)
    assert plan.segments[perimeter_indices[0]].region_id == plan.visit_order[-1]


def test_lane_and_phase_boundaries_are_monotonic_hard_stops():
    grid = np.full((90, 120), 100, dtype=np.int8)
    grid[8:82, 8:112] = 0

    plan = _plan(grid, _world(20, 20))

    assert plan.hard_stop_indices
    assert plan.hard_stop_indices == sorted(set(plan.hard_stop_indices))
    assert all(0 < index < len(plan.path) for index in plan.hard_stop_indices)
    # There is at least one boundary per cleaning lane; extra boundaries are
    # transfers and the fill->perimeter phase handoff.
    assert len(plan.hard_stop_indices) >= len(plan.swaths)
    for first, second in zip(plan.hard_stop_indices, plan.hard_stop_indices[1:]):
        assert second > first

    # Dense sampling remains continuous even though execution goals stop at the
    # shared indices.
    assert max(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(plan.path, plan.path[1:])
    ) <= 0.16
    assert len(plan.arrival_yaws) == len(plan.path)
    assert len(plan.departure_yaws) == len(plan.path)
    assert plan.raw_free_mask is not None
    assert plan.raw_free_mask.shape == grid.shape
    assert plan.free_mask.shape == grid.shape


def test_explicit_yaws_keep_distinct_arrival_and_departure_at_corner():
    arrivals, departures = _explicit_path_yaw_pairs([
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])

    assert math.isclose(arrivals[1], 0.0, abs_tol=1e-9)
    assert math.isclose(departures[1], math.pi / 2.0, abs_tol=1e-9)


def test_forward_brush_mask_stamps_both_hard_stop_orientations():
    mask = _forward_brush_cleaning_mask(
        (80, 80), [(3.0, 3.0)], [0.0], [math.pi / 2.0],
        RESOLUTION, 0.0, 0.0)

    def covered(x, y):
        row = int(math.floor(y / RESOLUTION))
        col = int(math.floor(x / RESOLUTION))
        return bool(mask[row, col])

    assert covered(3.30, 3.00)       # arrival pose: brush points east
    assert covered(3.00, 3.30)       # departure pose: brush points north
    assert not covered(2.70, 3.00)   # brush never extends behind the body
    assert not covered(3.00, 2.70)


def test_current_pose_lead_in_checks_full_asymmetric_rotation_sweep():
    raw_free = np.ones((80, 80), dtype=bool)
    safe, violations = validate_explicit_pose_transition(
        raw_free,
        (3.0, 3.0), 0.0,
        (4.0, 3.0), math.pi / 2.0,
        RESOLUTION, 0.0, 0.0,
    )
    assert safe and violations == 0

    # At the goal, TD25A's 0.72m rear overhang sweeps through the lower-left
    # quadrant while taking the shortest 0 -> +90 degree rotation.
    raw_free[24:27, 34:37] = False
    safe, violations = validate_explicit_pose_transition(
        raw_free,
        (3.0, 3.0), 0.0,
        (4.0, 3.0), math.pi / 2.0,
        RESOLUTION, 0.0, 0.0,
    )
    assert not safe and violations > 0


def test_execution_polyline_never_accepts_only_the_uncommanded_long_turn():
    raw_free = np.ones((80, 80), dtype=bool)
    raw_free[24, 34] = False
    path = [(3.0, 3.0), (4.0, 3.0), (4.0, 4.0)]

    safe, violations = validate_explicit_yaw_polyline(
        raw_free, path, RESOLUTION, 0.0, 0.0)
    candidate_only, _ = validate_explicit_yaw_polyline(
        raw_free, path, RESOLUTION, 0.0, 0.0,
        allow_staged_long_rotation=True)

    assert not safe and violations > 0
    assert candidate_only


def test_bcd_cells_cover_an_irregular_mask_once_without_raster_chips():
    mask = np.zeros((80, 120), dtype=bool)
    mask[8:72, 8:112] = True
    mask[25:55, 45:58] = False
    mask[15:38, 82:95] = False

    cells = _boustrophedon_cells(
        mask, "x", min_cell_area_m2=0.60, resolution=RESOLUTION)

    assert len(cells) >= 3
    combined = np.zeros_like(mask)
    for cell in cells:
        assert not (combined & cell).any()
        combined |= cell
        assert float(cell.sum()) * RESOLUTION ** 2 >= 0.50
    assert np.array_equal(combined, mask)


def test_near_rectangular_region_sweeps_along_its_long_edge():
    mask = np.zeros((70, 130), dtype=bool)
    mask[20:50, 15:115] = True

    axis, angle, swaths = _generate_component_swaths(
        mask, (20, 49, 15, 114), RESOLUTION, 0.0, 0.0,
        swath_spacing_m=0.525, min_swath_m=0.35)

    assert axis == "x"
    assert math.isclose(angle, 0.0, abs_tol=1e-9)
    assert swaths
    assert all(abs(end[1] - start[1]) <= 1e-9 for start, end in swaths)
    assert min(math.dist(start, end) for start, end in swaths) >= 9.5


def test_axis_aligned_room_with_asymmetric_furniture_is_not_falsely_rotated():
    mask = np.zeros((120, 160), dtype=bool)
    mask[25:95, 15:145] = True
    # A large furniture/no-data bite biases PCA by about 14 degrees even
    # though the physical room walls remain axis aligned.  Rotating the sweep
    # in this case creates diagonal fill connectors and avoidable crossings.
    mask[25:60, 85:145] = False

    axis, angle, swaths = _generate_component_swaths(
        mask, (25, 94, 15, 144), RESOLUTION, 0.0, 0.0,
        swath_spacing_m=0.525, min_swath_m=0.35)

    assert axis == "x"
    assert math.isclose(angle, 0.0, abs_tol=1e-9)
    assert swaths
    assert all(abs(end[1] - start[1]) <= 1e-9 for start, end in swaths)


def test_genuinely_rotated_near_rectangle_keeps_long_edge_sweeps():
    mask = np.zeros((160, 160), dtype=bool)
    rows, cols = np.indices(mask.shape)
    theta = math.radians(30.0)
    x = cols - 80.0
    y = rows - 80.0
    along = math.cos(theta) * x + math.sin(theta) * y
    across = -math.sin(theta) * x + math.cos(theta) * y
    mask[(np.abs(along) <= 50.0) & (np.abs(across) <= 15.0)] = True

    axis, angle, swaths = _generate_component_swaths(
        mask, (43, 117, 30, 130), RESOLUTION, 0.0, 0.0,
        swath_spacing_m=0.525, min_swath_m=0.35)

    assert axis == "rotated"
    assert math.isclose(angle, theta, abs_tol=math.radians(1.0))
    assert swaths
    lengths = sorted(math.dist(start, end) for start, end in swaths)
    assert lengths[len(lengths) // 2] >= 9.5


def test_four_rooms_use_door_connected_order_and_clean_corridor_once():
    grid = np.full((220, 260), 100, dtype=np.int8)
    grid[15:205, 108:152] = 0
    grid[25:92, 20:101] = 0
    grid[115:192, 20:101] = 0
    grid[25:92, 159:240] = 0
    grid[115:192, 159:240] = 0
    grid[55:68, 101:159] = 0
    grid[148:161, 101:159] = 0

    plan = _plan(grid, _world(180, 130))

    assert plan.coverage_complete
    assert plan.footprint_valid and plan.path_continuous
    assert plan.serviceable_coverage_ratio >= 0.95
    assert plan.actual_brush_coverage_ratio >= 0.95
    assert len(plan.regions) == 5
    assert len(plan.visit_order) == 5
    corridor = max(
        plan.regions,
        key=lambda region: (
            region.bbox_cells[1] - region.bbox_cells[0]
            - (region.bbox_cells[3] - region.bbox_cells[2])))
    assert plan.visit_order.index(corridor.region_id) == 2

    compressed = []
    for segment in plan.segments:
        if not compressed or compressed[-1] != segment.region_id:
            compressed.append(segment.region_id)
    assert compressed == plan.visit_order
    assert all(any(
        segment.kind == "fill" and segment.region_id == region_id
        for segment in plan.segments)
        for region_id in plan.visit_order)
    perimeter_indices = [
        index for index, segment in enumerate(plan.segments)
        if segment.kind == "perimeter"]
    fill_indices = [
        index for index, segment in enumerate(plan.segments)
        if segment.kind == "fill"]
    assert len(perimeter_indices) == 1
    assert perimeter_indices[0] > max(fill_indices)
    assert plan.segments[perimeter_indices[0]].region_id == plan.visit_order[-1]


def test_cluttered_bcd_connectors_reduce_crossings_and_absolute_retrace():
    rng = np.random.default_rng(0)
    grid = np.full((180, 240), 100, dtype=np.int8)
    grid[8:172, 8:232] = 0
    for _ in range(12):
        obstacle_height = int(rng.integers(4, 13))
        obstacle_width = int(rng.integers(4, 13))
        row = int(rng.integers(20, 160 - obstacle_height))
        col = int(rng.integers(35, 220 - obstacle_width))
        grid[
            row:row + obstacle_height,
            col:col + obstacle_width,
        ] = 100

    plan = _plan(
        grid, _world(25, 20), selection_policy="sparse_graph")

    assert plan.coverage_complete
    assert plan.footprint_valid and plan.footprint_violation_count == 0
    assert plan.path_continuous
    assert plan.serviceable_coverage_ratio >= 0.99
    assert plan.actual_brush_coverage_ratio >= 0.95
    assert plan.quality_metrics["avoidable_crossings"] <= 57
    assert plan.quality_metrics["avoidable_repeat_samples"] <= 163


def test_exit_aware_candidate_is_selected_only_when_both_quality_axes_improve():
    grid = np.full((220, 260), 100, dtype=np.int8)
    grid[15:205, 108:152] = 0
    grid[25:92, 20:101] = 0
    grid[115:192, 20:101] = 0
    grid[25:92, 159:240] = 0
    grid[115:192, 159:240] = 0
    grid[55:68, 101:159] = 0
    grid[148:161, 101:159] = 0
    robot = _world(180, 130)

    baseline = _plan(grid, robot, selection_policy="baseline")
    strict = _plan(grid, robot, selection_policy="strict")

    assert baseline.selection_mode == "baseline"
    assert strict.selection_mode == "exit_aware_strict"
    assert strict.footprint_valid and strict.path_continuous
    assert strict.serviceable_coverage_ratio >= 0.95
    assert strict.actual_brush_coverage_ratio >= 0.95
    assert strict.quality_metrics["avoidable_crossings"] <= (
        baseline.quality_metrics["avoidable_crossings"])
    assert strict.quality_metrics["avoidable_repeat_ratio"] < (
        baseline.quality_metrics["avoidable_repeat_ratio"])
    assert _canonical_swath_multiset(strict.swaths) == (
        _canonical_swath_multiset(baseline.swaths))


def test_shadow_policy_returns_the_exact_baseline_geometry():
    grid = np.full((220, 260), 100, dtype=np.int8)
    grid[15:205, 108:152] = 0
    grid[25:92, 20:101] = 0
    grid[115:192, 20:101] = 0
    grid[25:92, 159:240] = 0
    grid[115:192, 159:240] = 0
    grid[55:68, 101:159] = 0
    grid[148:161, 101:159] = 0
    robot = _world(180, 130)

    baseline = _plan(grid, robot, selection_policy="baseline")
    shadow = _plan(grid, robot, selection_policy="shadow")

    assert shadow.selection_mode == "baseline_shadow"
    assert shadow.path == baseline.path
    assert shadow.hard_stop_indices == baseline.hard_stop_indices
    assert shadow.visit_order == baseline.visit_order
    assert shadow.alternative_quality_metrics


def test_quality_gate_rejects_crossing_repeat_tradeoffs_and_tiny_ties():
    baseline = {
        "avoidable_crossings": 20.0,
        "avoidable_repeat_ratio": 0.08,
        "avoidable_repeat_samples": 80.0,
        "transfer_length_m": 40.0,
        "long_transfer_count": 5.0,
        "max_transfer_length_m": 8.0,
        "path_length_m": 300.0,
        "hard_stop_count": 20.0,
    }
    more_retrace = dict(
        baseline,
        avoidable_crossings=19.0,
        avoidable_repeat_ratio=0.081,
    )
    more_crossings = dict(
        baseline,
        avoidable_crossings=21.0,
        avoidable_repeat_ratio=0.07,
    )
    numerical_tie = dict(
        baseline,
        avoidable_repeat_ratio=0.07999,
    )
    extra_hard_stop = dict(
        baseline,
        avoidable_crossings=19.0,
        hard_stop_count=21.0,
    )
    more_repeat_samples = dict(
        baseline,
        avoidable_crossings=19.0,
        avoidable_repeat_ratio=0.079,
        avoidable_repeat_samples=81.0,
    )
    strict_gain = dict(
        baseline,
        avoidable_crossings=19.0,
        avoidable_repeat_ratio=0.075,
        transfer_length_m=35.0,
    )
    bounded_crossing_trade = dict(
        baseline,
        avoidable_crossings=14.0,
        avoidable_repeat_ratio=0.0808,
        avoidable_repeat_samples=84.0,
        path_length_m=299.0,
    )
    excessive_crossing_trade = dict(
        bounded_crossing_trade,
        avoidable_repeat_ratio=0.082,
        avoidable_repeat_samples=86.0,
    )

    assert not _strictly_prefer_exit_candidate(baseline, more_retrace)
    assert not _strictly_prefer_exit_candidate(baseline, more_crossings)
    assert not _strictly_prefer_exit_candidate(baseline, numerical_tie)
    assert not _strictly_prefer_exit_candidate(baseline, extra_hard_stop)
    assert not _strictly_prefer_exit_candidate(baseline, more_repeat_samples)
    assert _strictly_prefer_exit_candidate(baseline, strict_gain)
    assert _strictly_prefer_exit_candidate(
        baseline, bounded_crossing_trade)
    assert not _strictly_prefer_exit_candidate(
        baseline, excessive_crossing_trade)


def test_short_swath_safety_failure_falls_back_to_proven_lane_floor():
    original = grid_coverage_module._plan_partitioned_coverage_once
    calls = []

    def fake_once(**kwargs):
        lane_floor = float(kwargs["min_swath_m"])
        calls.append(lane_floor)
        safe = lane_floor >= 0.45 - 1e-9
        mask = np.ones((2, 2), dtype=bool)
        return PartitionedCoveragePlan(
            path=[(0.0, 0.0), (1.0, 0.0)],
            segments=[], regions=[], visit_order=[], free_mask=mask,
            snapped_start=(0.0, 0.0),
            footprint_valid=safe,
            footprint_violation_count=0 if safe else 1,
            path_continuous=safe,
            turn_safe_coverage_ratio=1.0 if safe else 0.0,
            serviceable_coverage_ratio=1.0 if safe else 0.0,
            actual_brush_coverage_ratio=1.0 if safe else 0.0,
            reachable_coverage_ratio=1.0 if safe else 0.0,
            coverage_complete=safe,
        )

    grid_coverage_module._plan_partitioned_coverage_once = fake_once
    try:
        grid = np.zeros((2, 2), dtype=np.int8)
        plan = plan_partitioned_coverage(
            grid.tobytes(), 2, 2, 0.10, 0.0, 0.0,
            robot_world=(0.05, 0.05), swath_spacing_m=0.525,
            min_swath_m=0.35, selection_policy="baseline")
    finally:
        grid_coverage_module._plan_partitioned_coverage_once = original

    assert calls == [0.35, 0.45]
    assert plan.coverage_complete
    assert plan.footprint_violation_count == 0
    assert plan.selection_mode == "baseline_min_swath_fallback"


def test_three_levels_of_rooms_are_not_merged_into_the_long_corridor():
    grid = np.full((230, 300), 100, dtype=np.int8)
    grid[10:220, 130:170] = 0
    for room_r0, room_r1, door_r0, door_r1 in (
            (20, 75, 42, 57),
            (90, 145, 108, 123),
            (160, 215, 178, 193)):
        grid[room_r0:room_r1, 20:122] = 0
        grid[room_r0:room_r1, 178:280] = 0
        grid[door_r0:door_r1, 122:178] = 0

    plan = _plan(grid, _world(205, 150))

    assert plan.coverage_complete
    assert plan.footprint_valid and plan.path_continuous
    assert plan.serviceable_coverage_ratio >= 0.95
    assert plan.actual_brush_coverage_ratio >= 0.95
    # Six physical rooms plus the central cleanable corridor.  A dominant-axis
    # guillotine alone incorrectly merged all three rooms on one side into the
    # corridor and then failed its mandatory perimeter pass.
    assert len(plan.regions) == 7
    assert len(plan.visit_order) == 7
    assert all(any(
        segment.kind == "fill" and segment.region_id == region_id
        for segment in plan.segments)
        for region_id in plan.visit_order)
    perimeter_indices = [
        index for index, segment in enumerate(plan.segments)
        if segment.kind == "perimeter"]
    assert len(perimeter_indices) == 1
    assert plan.segments[perimeter_indices[0]].region_id == plan.visit_order[-1]
