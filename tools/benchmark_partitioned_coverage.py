#!/usr/bin/env python3
"""Deterministic offline regression for the TD25A partitioned planner.

This utility reads saved maps only.  It never imports ROS, publishes a goal, or
touches ``cmd_vel``.  The output is intentionally machine-readable so planner
iterations can be compared without moving the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("TD25A_BENCH_SOURCE", str(ROOT)))
DEFAULT_RENDER_DIR = Path(os.environ.get(
    "TD25A_BENCH_RENDER_DIR",
    str(Path(__file__).resolve().parent),
))
sys.path.insert(0, str(SOURCE_ROOT / "src/td25a_robot_ui"))
sys.path.insert(0, str(DEFAULT_RENDER_DIR))

import render_partitioned_planner as preview  # noqa: E402
from td25a_robot_ui.algorithms.free_space import extract_clean_field_result  # noqa: E402
from td25a_robot_ui.algorithms.grid_coverage import (  # noqa: E402
    coverage_swath_spacing,
    plan_partitioned_coverage,
)


def path_length(points) -> float:
    return sum(math.hypot(
        second[0] - first[0], second[1] - first[1])
        for first, second in zip(points, points[1:]))


def yaml_paths(directory: Path, names: Iterable[str]) -> List[Path]:
    requested = list(names)
    if not requested:
        return sorted(directory.glob("*.yaml"))
    paths = []
    for name in requested:
        candidate = directory / (name if name.endswith(".yaml") else f"{name}.yaml")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        paths.append(candidate)
    return paths


def run_map(yaml_path: Path) -> Dict[str, object]:
    config, _, grid, known_free, recovered = preview.load_map(yaml_path)
    resolution = config["resolution"]
    origin_x, origin_y = config["origin"][:2]
    height, width = grid.shape
    seed, seed_note = preview.choose_seed(
        grid, resolution, origin_x, origin_y,
        yaml_path.stem, known_free)
    extraction = extract_clean_field_result(
        grid.tobytes(), width, height, resolution, origin_x, origin_y,
        robot_radius_m=0.34, seed_world=seed,
        known_free_mask=known_free, max_seed_snap_m=2.0,
        recovered_unknown_cells=recovered,
    )
    if not extraction.polygon:
        return {
            "name": yaml_path.stem,
            "seed": seed,
            "seed_note": seed_note,
            "extraction_failure": extraction.failure_reason,
        }
    started = time.perf_counter()
    plan = plan_partitioned_coverage(
        grid.tobytes(), width, height, resolution, origin_x, origin_y,
        robot_world=seed,
        swath_spacing_m=coverage_swath_spacing(0.70, 0.75),
        clip_polygon=extraction.polygon,
        selection_boundary_polygon=extraction.polygon,
        known_free_mask=known_free,
        path_step_m=0.10,
        min_swath_m=1.30,
        min_region_area_m2=5.0,
        min_useful_region_area_m2=15.0,
        min_useful_region_lane_m=3.0,
        max_regions=12,
        clean_width_m=0.70,
        selection_policy="sparse_graph",
    )
    elapsed = time.perf_counter() - started
    return {
        "name": yaml_path.stem,
        "seed": [round(seed[0], 3), round(seed[1], 3)],
        "seed_note": seed_note,
        "reachable_area_m2": round(extraction.reachable_area_m2, 4),
        "regions": len(plan.regions),
        "bcd_cells": sum(region.cell_count for region in plan.regions),
        "visited_regions": len(plan.visit_order),
        "swaths": len(plan.swaths),
        "points": len(plan.path),
        "hard_stops": len(plan.hard_stop_indices),
        "length_m": round(path_length(plan.path), 4),
        "planning_seconds": round(elapsed, 4),
        "footprint_valid": plan.footprint_valid,
        "footprint_violations": plan.footprint_violation_count,
        "path_continuous": plan.path_continuous,
        "max_gap_m": round(plan.max_segment_gap_m, 6),
        "turn_safe_coverage": round(plan.turn_safe_coverage_ratio, 6),
        "serviceable_coverage": round(
            plan.serviceable_coverage_ratio, 6),
        "reachable_coverage": round(plan.reachable_coverage_ratio, 6),
        "actual_brush_coverage": round(
            plan.actual_brush_coverage_ratio, 6),
        "centered_brush_coverage": round(
            plan.centered_brush_coverage_ratio, 6),
        "cleaner_extension_gain_area_m2": round(
            plan.cleaner_extension_gain_area_m2, 6),
        "cleaner_semantics_valid": plan.cleaner_semantics_valid,
        "cleaner_semantics_failure_reason": (
            plan.cleaner_semantics_failure_reason),
        "cleaner_mode_point_counts": plan.cleaner_mode_point_counts,
        "selection_mode": plan.selection_mode,
        "region_serviceable_coverages": {
            str(region_id): round(value, 6)
            for region_id, value in
            plan.region_serviceable_coverage_ratios.items()
        },
        "coverage_complete": plan.coverage_complete,
        "failure_reason": plan.failure_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument("--map", action="append", default=[])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    records = []
    for yaml_path in yaml_paths(args.maps_dir, args.map):
        try:
            record = run_map(yaml_path)
        except Exception as error:  # keep the batch diagnostic fail-closed
            record = {
                "name": yaml_path.stem,
                "benchmark_error": f"{type(error).__name__}: {error}",
            }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    summary = {
        "maps": len(records),
        "valid": sum(bool(record.get("footprint_valid")) for record in records),
        "complete": sum(bool(record.get("coverage_complete")) for record in records),
        "minimum_serviceable_coverage": min(
            (float(record["serviceable_coverage"]) for record in records
             if "serviceable_coverage" in record),
            default=0.0,
        ),
        "minimum_strict_reachable_coverage": min(
            (float(record["reachable_coverage"]) for record in records
             if "reachable_coverage" in record), default=0.0),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"records": records, "summary": summary},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
