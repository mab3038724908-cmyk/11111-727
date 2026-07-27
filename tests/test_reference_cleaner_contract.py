import math

import numpy as np
import pytest

from td25a_robot_ui.algorithms.cleaning_mode import (
    CleaningPhase,
    LEFT_EDGE_ULTRASONICS,
    RIGHT_EDGE_ULTRASONICS,
    VerticalIntent,
    edge_extension_for_wall_distance,
    edge_extension_from_fixed_ranges,
    intent_for_cleaning_phase,
    wall_alignment_error_rad,
)
from td25a_robot_ui.algorithms.grid_coverage import plan_partitioned_coverage
from td25a_robot_ui.store.cleaned_area import paint_brush


def test_right_edge_geometry_and_safety_channels_follow_reference_contract():
    assert RIGHT_EDGE_ULTRASONICS.poll_channels == (4, 6, 7, 8, 10)
    assert RIGHT_EDGE_ULTRASONICS.poll_mask == 0x2E8
    assert RIGHT_EDGE_ULTRASONICS.moving_guard_channel == 9
    assert LEFT_EDGE_ULTRASONICS.poll_channels == (2, 4, 5, 6, 7)
    assert LEFT_EDGE_ULTRASONICS.poll_mask == 0x07A
    assert wall_alignment_error_rad(0.62, 0.60) == pytest.approx(
        math.atan2(0.02, 0.429))

    reachable = edge_extension_for_wall_distance(0.8348)
    assert reachable.extension_m == pytest.approx(0.2500)
    assert reachable.expected_cleaning_gap_m == pytest.approx(0.0300)
    assert reachable.reachable
    clamped = edge_extension_from_fixed_ranges(0.600, 0.600)
    assert clamped.extension_m == pytest.approx(0.250)
    assert not clamped.reachable


def test_fill_never_extends_and_transit_is_not_cleaning():
    fill = intent_for_cleaning_phase(CleaningPhase.FILL_DOWN, 0.2)
    assert (fill.vertical, fill.x_extension_m, fill.cleaning_enabled) == (
        VerticalIntent.DOWN, 0.0, True)
    transit = intent_for_cleaning_phase(CleaningPhase.TRANSIT_UP, 0.2)
    assert (transit.vertical, transit.x_extension_m, transit.cleaning_enabled) == (
        VerticalIntent.UP, 0.0, False)


def test_lateral_extension_moves_the_actual_brush_right():
    centered = np.zeros((120, 120), dtype=np.uint16)
    extended = np.zeros_like(centered)
    paint_brush(centered, 0.02, -1.2, -1.2, 0.0, 0.0, 0.0)
    paint_brush(extended, 0.02, -1.2, -1.2, 0.0, 0.0, 0.0,
                lateral_offset_m=-0.25)
    assert (np.nonzero(centered)[0].mean() - np.nonzero(extended)[0].mean()
            ) * 0.02 == pytest.approx(0.25, abs=0.02)


def test_selected_zone_keeps_leadin_from_robot_outside_mission_halo():
    resolution = 0.10
    grid = np.zeros((100, 200), dtype=np.int8)
    grid[[0, -1], :] = 100
    grid[:, [0, -1]] = 100
    robot = (1.50, 5.00)
    zone = [(8.0, 2.0), (16.0, 2.0), (16.0, 8.0), (8.0, 8.0)]
    plan = plan_partitioned_coverage(
        data=grid.tobytes(), width=grid.shape[1], height=grid.shape[0],
        resolution=resolution, origin_x=0.0, origin_y=0.0,
        robot_world=robot, robot_yaw=0.0, swath_spacing_m=0.525,
        clip_polygon=zone, selection_boundary_polygon=zone, path_step_m=0.10,
        min_swath_m=1.30, min_region_area_m2=3.0,
        min_useful_region_area_m2=0.0, min_useful_region_lane_m=0.0,
        max_regions=4, clean_width_m=0.70,
        enable_cleaner_semantics=False, selection_policy="baseline")
    assert plan.path, plan.failure_reason
    assert math.dist(plan.path[0], robot) <= 0.20
    assert plan.path[0][0] < 7.0
    assert plan.footprint_valid and plan.path_continuous
