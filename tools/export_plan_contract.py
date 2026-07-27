#!/usr/bin/env python3
"""Export an offline coverage plan as a versioned, point-aligned JSON contract.

The program reads a saved map and executes only the deterministic geometry
planner bundled in this delivery.  It contains no network, ROS, or hardware
control code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/td25a_robot_ui"))
sys.path.insert(0, str(ROOT / "tools"))

import render_partitioned_planner as preview  # noqa: E402
from td25a_robot_ui.algorithms.free_space import (  # noqa: E402
    extract_clean_field_result,
)
from td25a_robot_ui.algorithms.grid_coverage import (  # noqa: E402
    PartitionedCoveragePlan,
    coverage_swath_spacing,
    plan_partitioned_coverage,
)


CONTRACT_VERSION = "td25a.coverage_plan.v1"
DEFAULT_PROFILE: Dict[str, Any] = {
    "clean_width_m": 0.70,
    "swath_spacing_factor": 0.75,
    "effective_overlap_fraction": 0.25,
    "swath_spacing_m": 0.525,
    "path_step_m": 0.10,
    "min_swath_m": 1.30,
    "min_region_area_m2": 5.0,
    "min_useful_region_area_m2": 15.0,
    "min_useful_region_lane_m": 3.0,
    "max_regions": 12,
    "selection_policy": "sparse_graph",
}


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _point(point: Sequence[float]) -> list[float]:
    return [_round(point[0]), _round(point[1])]


def _path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(points, points[1:])
    )


def validate_plan_alignment(plan: PartitionedCoveragePlan) -> None:
    """Fail closed when any point-aligned execution field is missing."""
    point_count = len(plan.path)
    aligned = {
        "arrival_yaws": len(plan.arrival_yaws),
        "departure_yaws": len(plan.departure_yaws),
        "cleaner_profile": len(plan.cleaner_profile),
        "cleaner_center_path": len(plan.cleaner_center_path),
    }
    mismatched = {
        name: size for name, size in aligned.items() if size != point_count
    }
    if mismatched:
        raise ValueError(
            f"point-aligned plan fields mismatch path={point_count}: "
            f"{mismatched}"
        )
    bad_stops = [
        index for index in plan.hard_stop_indices
        if index < 0 or index >= point_count
    ]
    if bad_stops:
        raise ValueError(f"hard_stop_indices out of range: {bad_stops[:10]}")
    finite_values: Iterable[float] = (
        coordinate for point in plan.path for coordinate in point
    )
    if not all(math.isfinite(float(value)) for value in finite_values):
        raise ValueError("path contains non-finite coordinates")


def plan_to_contract(
    plan: PartitionedCoveragePlan,
    *,
    map_name: str,
    map_sha256: str,
    resolution_m: float,
    origin_xy: Sequence[float],
    seed_xy: Sequence[float],
    planning_seconds: float,
    summary_only: bool = False,
) -> Dict[str, Any]:
    """Convert a plan to a stable JSON-compatible structure."""
    validate_plan_alignment(plan)
    hard_stops = set(plan.hard_stop_indices)
    metrics = {
        "path_length_m": _round(_path_length(plan.path), 4),
        "planning_seconds": _round(planning_seconds, 4),
        "region_count": len(plan.regions),
        "visited_region_count": len(plan.visit_order),
        "segment_count": len(plan.segments),
        "swath_count": len(plan.swaths),
        "point_count": len(plan.path),
        "hard_stop_count": len(plan.hard_stop_indices),
        "footprint_valid": bool(plan.footprint_valid),
        "footprint_violation_count": int(plan.footprint_violation_count),
        "path_continuous": bool(plan.path_continuous),
        "max_segment_gap_m": _round(plan.max_segment_gap_m),
        "coverage_complete": bool(plan.coverage_complete),
        "turn_safe_coverage_ratio": _round(plan.turn_safe_coverage_ratio),
        "serviceable_coverage_ratio": _round(
            plan.serviceable_coverage_ratio),
        "actual_brush_coverage_ratio": _round(
            plan.actual_brush_coverage_ratio),
        "centered_brush_coverage_ratio": _round(
            plan.centered_brush_coverage_ratio),
        "cleaner_extension_gain_area_m2": _round(
            plan.cleaner_extension_gain_area_m2),
        "cleaner_semantics_valid": bool(plan.cleaner_semantics_valid),
        "selection_mode": plan.selection_mode,
        "failure_reason": plan.failure_reason,
        "cleaner_semantics_failure_reason": (
            plan.cleaner_semantics_failure_reason),
    }
    regions = [
        {
            "region_id": int(region.region_id),
            "centroid_xy_m": _point(region.centroid),
            "area_m2": _round(region.area_m2, 4),
            "axis": region.axis,
            "bcd_cell_count": int(region.cell_count),
            "bbox_cells": [int(value) for value in region.bbox_cells],
            "serviceable_coverage_ratio": _round(
                plan.region_serviceable_coverage_ratios.get(
                    region.region_id, 0.0)),
            "actual_brush_coverage_ratio": _round(
                plan.region_actual_brush_coverage_ratios.get(
                    region.region_id, 0.0)),
        }
        for region in plan.regions
    ]
    segments = [
        {
            "segment_id": index,
            "kind": segment.kind,
            "region_id": int(segment.region_id),
            "from_region_id": int(segment.from_region_id),
            "to_region_id": int(segment.to_region_id),
            "path_start_index": int(segment.path_start_idx),
            "path_end_index": int(segment.path_end_idx),
            "point_count": len(segment.path),
            "swath_count": len(segment.swaths),
            "continuous_to_next": bool(segment.continuous_to_next),
            "default_cleaner_mode": segment.cleaner_mode.value,
            "default_cleaner_offset_m": _round(segment.cleaner_offset_m),
        }
        for index, segment in enumerate(plan.segments)
    ]
    contract: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "units": {
            "position": "metre",
            "angle": "radian",
            "cleaner_offset_sign": "positive_is_body_left",
            "body_axes": "+x_forward,+y_left",
        },
        "source": {
            "map_name": map_name,
            "map_sha256": map_sha256,
            "resolution_m": _round(resolution_m),
            "origin_xy_m": _point(origin_xy),
            "seed_xy_m": _point(seed_xy),
            "profile": dict(DEFAULT_PROFILE),
        },
        "robot_geometry_m": {
            "front": _round(plan.footprint.front_m),
            "rear": _round(plan.footprint.rear_m),
            "half_width": _round(plan.footprint.half_width_m),
            "turn_margin": _round(plan.footprint.turn_margin_m),
        },
        "cleaner_contract": {
            "status": "simulated_semantics_not_hardware_calibrated",
            "simulated_max_lateral_offset_m": _round(
                plan.cleaner_max_offset_m),
            "mode_point_counts": dict(plan.cleaner_mode_point_counts),
            "boundary_type_point_counts": dict(
                plan.boundary_type_point_counts),
        },
        "metrics": metrics,
        "visit_order": [int(value) for value in plan.visit_order],
        "hard_stop_indices": [int(value) for value in plan.hard_stop_indices],
        "regions": regions,
        "segments": segments,
    }
    if not summary_only:
        contract["trajectory"] = [
            {
                "index": index,
                "base_xy_m": _point(base),
                "arrival_yaw_rad": _round(plan.arrival_yaws[index]),
                "departure_yaw_rad": _round(plan.departure_yaws[index]),
                "hard_stop": index in hard_stops,
                "cleaner_center_xy_m": _point(
                    plan.cleaner_center_path[index]),
                "cleaner": {
                    "mode": plan.cleaner_profile[index].mode.value,
                    "lateral_offset_m": _round(
                        plan.cleaner_profile[index].offset_m),
                    "boundary_type": (
                        plan.cleaner_profile[index].boundary_type.value),
                    "cleaning_enabled": bool(
                        plan.cleaner_profile[index].cleaning_enabled),
                },
            }
            for index, base in enumerate(plan.path)
        ]
    return contract


def build_plan(
    yaml_path: Path,
    seed_override: Optional[Tuple[float, float]] = None,
) -> tuple[PartitionedCoveragePlan, Dict[str, Any]]:
    config, _, grid, known_free, recovered = preview.load_map(yaml_path)
    resolution = float(config["resolution"])
    origin_x, origin_y = map(float, config["origin"][:2])
    height, width = grid.shape
    if seed_override is None:
        seed, seed_note = preview.choose_seed(
            grid, resolution, origin_x, origin_y,
            yaml_path.stem, known_free)
    else:
        seed = seed_override
        seed_note = "command-line seed"
    extraction = extract_clean_field_result(
        grid.tobytes(), width, height, resolution, origin_x, origin_y,
        robot_radius_m=0.34,
        seed_world=seed,
        known_free_mask=known_free,
        max_seed_snap_m=2.0,
        recovered_unknown_cells=recovered,
    )
    if not extraction.polygon:
        raise RuntimeError(
            f"clean-field extraction failed: {extraction.failure_reason}")
    started = time.perf_counter()
    plan = plan_partitioned_coverage(
        grid.tobytes(), width, height, resolution, origin_x, origin_y,
        robot_world=seed,
        swath_spacing_m=coverage_swath_spacing(
            DEFAULT_PROFILE["clean_width_m"],
            DEFAULT_PROFILE["swath_spacing_factor"]),
        clip_polygon=extraction.polygon,
        selection_boundary_polygon=extraction.polygon,
        known_free_mask=known_free,
        path_step_m=DEFAULT_PROFILE["path_step_m"],
        min_swath_m=DEFAULT_PROFILE["min_swath_m"],
        min_region_area_m2=DEFAULT_PROFILE["min_region_area_m2"],
        min_useful_region_area_m2=(
            DEFAULT_PROFILE["min_useful_region_area_m2"]),
        min_useful_region_lane_m=(
            DEFAULT_PROFILE["min_useful_region_lane_m"]),
        max_regions=DEFAULT_PROFILE["max_regions"],
        clean_width_m=DEFAULT_PROFILE["clean_width_m"],
        selection_policy=DEFAULT_PROFILE["selection_policy"],
    )
    context = {
        "resolution_m": resolution,
        "origin_xy": (origin_x, origin_y),
        "seed_xy": seed,
        "seed_note": seed_note,
        "planning_seconds": time.perf_counter() - started,
        "reachable_area_m2": extraction.reachable_area_m2,
    }
    return plan, context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map_yaml", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--seed", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    yaml_path = args.map_yaml.resolve()
    plan, context = build_plan(
        yaml_path, tuple(args.seed) if args.seed else None)
    contract = plan_to_contract(
        plan,
        map_name=yaml_path.stem,
        map_sha256=hashlib.sha256(yaml_path.read_bytes()).hexdigest(),
        resolution_m=context["resolution_m"],
        origin_xy=context["origin_xy"],
        seed_xy=context["seed_xy"],
        planning_seconds=context["planning_seconds"],
        summary_only=args.summary_only,
    )
    contract["source"]["seed_note"] = context["seed_note"]
    contract["source"]["reachable_area_m2"] = _round(
        context["reachable_area_m2"], 4)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output_json),
        "contract_version": CONTRACT_VERSION,
        "summary_only": args.summary_only,
        "metrics": contract["metrics"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
