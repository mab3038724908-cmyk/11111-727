from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from td25a_robot_ui.algorithms.grid_coverage import (
    BoundaryType,
    CleanerCommand,
    CleanerMode,
    CoverageFootprint,
    CoverageRegion,
    CoverageSegment,
    PartitionedCoveragePlan,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_plan_contract", ROOT / "tools/export_plan_contract.py")
EXPORTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXPORTER)


def _sample_plan() -> PartitionedCoveragePlan:
    path = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]
    commands = [
        CleanerCommand(),
        CleanerCommand(
            mode=CleanerMode.EDGE_LEFT,
            offset_m=0.20,
            boundary_type=BoundaryType.PHYSICAL_WALL,
        ),
        CleanerCommand(
            mode=CleanerMode.EDGE_RETRACT,
            cleaning_enabled=False,
        ),
    ]
    segment = CoverageSegment(
        kind="fill",
        region_id=7,
        path=path,
        path_start_idx=0,
        path_end_idx=2,
        cleaner_profile=commands,
    )
    region = CoverageRegion(
        region_id=7,
        mask=np.ones((2, 2), dtype=bool),
        bbox_cells=(0, 0, 1, 1),
        centroid=(0.5, 0.0),
        area_m2=2.0,
        axis="x",
    )
    return PartitionedCoveragePlan(
        path=path,
        segments=[segment],
        regions=[region],
        visit_order=[7],
        free_mask=np.ones((2, 2), dtype=bool),
        snapped_start=path[0],
        hard_stop_indices=[2],
        footprint_valid=True,
        coverage_complete=True,
        path_continuous=True,
        serviceable_coverage_ratio=0.97,
        actual_brush_coverage_ratio=0.98,
        centered_brush_coverage_ratio=0.95,
        cleaner_semantics_valid=True,
        arrival_yaws=[0.0, 0.0, 0.0],
        departure_yaws=[0.0, 0.0, 0.0],
        cleaner_profile=commands,
        cleaner_center_path=[(0.0, 0.0), (0.5, 0.2), (1.0, 0.0)],
        cleaner_max_offset_m=0.25,
        cleaner_mode_point_counts={
            "EDGE_CENTER": 1, "EDGE_LEFT": 1, "EDGE_RETRACT": 1},
        footprint=CoverageFootprint(),
    )


def test_contract_keeps_base_and_cleaner_state_point_aligned():
    contract = EXPORTER.plan_to_contract(
        _sample_plan(),
        map_name="synthetic",
        map_sha256="0" * 64,
        resolution_m=0.05,
        origin_xy=(-1.0, -1.0),
        seed_xy=(0.0, 0.0),
        planning_seconds=0.01,
    )
    assert contract["contract_version"] == "td25a.coverage_plan.v1"
    assert len(contract["trajectory"]) == 3
    left = contract["trajectory"][1]
    assert left["cleaner"]["mode"] == "EDGE_LEFT"
    assert left["cleaner"]["lateral_offset_m"] == pytest.approx(0.20)
    assert left["cleaner_center_xy_m"] == pytest.approx([0.5, 0.2])
    assert contract["trajectory"][2]["hard_stop"] is True
    assert contract["units"]["cleaner_offset_sign"] == (
        "positive_is_body_left")


def test_summary_contract_omits_heavy_trajectory_only():
    contract = EXPORTER.plan_to_contract(
        _sample_plan(),
        map_name="synthetic",
        map_sha256="0" * 64,
        resolution_m=0.05,
        origin_xy=(0.0, 0.0),
        seed_xy=(0.0, 0.0),
        planning_seconds=0.01,
        summary_only=True,
    )
    assert "trajectory" not in contract
    assert contract["metrics"]["point_count"] == 3
    assert contract["segments"][0]["path_start_index"] == 0
    assert contract["regions"][0]["region_id"] == 7


def test_alignment_validation_fails_closed():
    plan = _sample_plan()
    plan.cleaner_profile.pop()
    with pytest.raises(ValueError, match="cleaner_profile"):
        EXPORTER.validate_plan_alignment(plan)


def test_delivery_has_no_embedded_machine_path_or_remote_client():
    # Split the sentinels so the contract test does not flag its own source.
    forbidden_text = (
        "/" + "Users/", "/home/" + "wheeltec", "100." + "107.")
    forbidden_imports = (
        "import " + "paramiko", "from " + "paramiko",
        "import " + "rclpy", "from " + "rclpy")
    for path in ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden_text), path
        assert not any(value in text for value in forbidden_imports), path
