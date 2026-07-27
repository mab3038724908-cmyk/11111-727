#!/usr/bin/env python3
"""Offline square/L/corridor/random-obstacle TD25A coverage regression."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/td25a_robot_ui"))

from td25a_robot_ui.algorithms.grid_coverage import (  # noqa: E402
    _simplify_collinear,
    coverage_swath_spacing,
    plan_partitioned_coverage,
)


RESOLUTION = 0.10


def world(row: int, col: int):
    return ((col + 0.5) * RESOLUTION, (row + 0.5) * RESOLUTION)


def path_length(path) -> float:
    return sum(math.hypot(
        second[0] - first[0], second[1] - first[1])
        for first, second in zip(path, path[1:]))


def repeat_ratio(path) -> float:
    seen = set()
    repeated = 0
    samples = 0
    last_cell = None
    for first, second in zip(path, path[1:]):
        distance = math.hypot(second[0] - first[0], second[1] - first[1])
        count = max(1, int(math.ceil(distance / 0.05)))
        for index in range(count):
            ratio = index / count
            point = (
                first[0] + (second[0] - first[0]) * ratio,
                first[1] + (second[1] - first[1]) * ratio,
            )
            cell = (
                int(math.floor(point[1] / RESOLUTION)),
                int(math.floor(point[0] / RESOLUTION)),
            )
            if cell == last_cell:
                continue
            repeated += int(cell in seen)
            seen.add(cell)
            samples += 1
            last_cell = cell
    return repeated / samples if samples else 0.0


def segment_repeat_metrics(plan):
    """Separate routing retrace from intentional fill/perimeter contact.

    A red perimeter necessarily meets every yellow lane endpoint in the same
    room.  Counting those shared raster cells as route retrace makes a clean
    rectangle look worse than a genuinely tangled connector.  Keep the raw
    ratio for transparency, and report the remaining upper bound after
    excluding only ``same-region fill -> perimeter`` contact.  That remainder
    still includes physically unavoidable reuse, such as entering and leaving
    a dead-end room through its only doorway, so it must not be presented as a
    proof that every counted sample can actually be eliminated.
    """
    seen = {}
    samples = repeated = intended = 0
    last_cell = None
    breakdown = {}
    for segment in plan.segments:
        current = (segment.kind, int(segment.region_id))
        for first, second in zip(segment.path, segment.path[1:]):
            distance = math.hypot(
                second[0] - first[0], second[1] - first[1])
            count = max(1, int(math.ceil(distance / 0.05)))
            for index in range(count):
                ratio = index / count
                point = (
                    first[0] + (second[0] - first[0]) * ratio,
                    first[1] + (second[1] - first[1]) * ratio,
                )
                cell = (
                    int(math.floor(point[1] / RESOLUTION)),
                    int(math.floor(point[0] / RESOLUTION)),
                )
                if cell == last_cell:
                    continue
                previous = seen.get(cell)
                if previous is not None:
                    repeated += 1
                    key = f"{previous[0]}->{current[0]}"
                    breakdown[key] = breakdown.get(key, 0) + 1
                    if (previous[0] == "fill"
                            and current[0] == "perimeter"
                            and previous[1] == current[1]):
                        intended += 1
                else:
                    seen[cell] = current
                samples += 1
                last_cell = cell
    avoidable = max(0, repeated - intended)
    denominator = max(1, samples)
    return {
        "raw": repeated / denominator,
        "avoidable": avoidable / denominator,
        "intended_fill_perimeter": intended / denominator,
        "breakdown": breakdown,
    }


def proper_crossings(path) -> int:
    points = _simplify_collinear(path)
    segments = list(zip(points, points[1:]))

    def orientation(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    crossings = 0
    for first_index, (a, b) in enumerate(segments):
        min_ax, max_ax = sorted((a[0], b[0]))
        min_ay, max_ay = sorted((a[1], b[1]))
        for second_index in range(first_index + 2, len(segments)):
            if second_index == first_index + 1:
                continue
            c, d = segments[second_index]
            if (max_ax <= min(c[0], d[0]) + 1e-8
                    or max(c[0], d[0]) <= min_ax + 1e-8
                    or max_ay <= min(c[1], d[1]) + 1e-8
                    or max(c[1], d[1]) <= min_ay + 1e-8):
                continue
            o1 = orientation(a, b, c)
            o2 = orientation(a, b, d)
            o3 = orientation(c, d, a)
            o4 = orientation(c, d, b)
            if o1 * o2 < -1e-10 and o3 * o4 < -1e-10:
                crossings += 1
    return crossings


def crossing_metrics(plan):
    labelled = []
    for segment_index, segment in enumerate(plan.segments):
        points = _simplify_collinear(segment.path)
        for local_index, (first, second) in enumerate(zip(points, points[1:])):
            labelled.append((
                first, second, segment.kind, int(segment.region_id),
                segment_index, local_index))

    def orientation(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    counts = {}
    intended = 0
    avoidable = 0
    for first_index, (
            a, b, first_kind, first_region,
            first_segment, first_local) in enumerate(labelled):
        min_ax, max_ax = sorted((a[0], b[0]))
        min_ay, max_ay = sorted((a[1], b[1]))
        for (c, d, second_kind, second_region,
             second_segment, second_local) in labelled[first_index + 1:]:
            if (first_segment == second_segment
                    and abs(first_local - second_local) <= 1):
                continue
            if (max_ax <= min(c[0], d[0]) + 1e-8
                    or max(c[0], d[0]) <= min_ax + 1e-8
                    or max_ay <= min(c[1], d[1]) + 1e-8
                    or max(c[1], d[1]) <= min_ay + 1e-8):
                continue
            if (orientation(a, b, c) * orientation(a, b, d) < -1e-10
                    and orientation(c, d, a) * orientation(c, d, b) < -1e-10):
                key = "+".join(sorted((first_kind, second_kind)))
                counts[key] = counts.get(key, 0) + 1
                kinds = {first_kind, second_kind}
                if (kinds == {"fill", "perimeter"}
                        and first_region == second_region):
                    intended += 1
                else:
                    avoidable += 1
    return {
        "categories": counts,
        "total": intended + avoidable,
        "avoidable": avoidable,
        "intended_fill_perimeter": intended,
    }


def crossing_categories(plan):
    return crossing_metrics(plan)["categories"]


def cases(
    random_count: int,
    random_manual_count: int = 0,
    random_office_count: int = 0,
):
    square = np.full((110, 140), 100, dtype=np.int8)
    square[8:102, 8:132] = 0
    yield "square", square, world(25, 25), None

    large_l = np.full((160, 160), 100, dtype=np.int8)
    large_l[10:150, 10:72] = 0
    large_l[88:150, 10:150] = 0
    yield "large_l", large_l, world(120, 25), None

    office = np.full((220, 260), 100, dtype=np.int8)
    office[15:205, 108:152] = 0
    office[25:92, 20:101] = 0
    office[115:192, 20:101] = 0
    office[25:92, 159:240] = 0
    office[115:192, 159:240] = 0
    # Door throats from the central corridor into all four rooms.
    office[55:68, 101:159] = 0
    office[148:161, 101:159] = 0
    yield "four_rooms_corridor", office, world(180, 130), None

    open_grid = np.full((180, 220), 100, dtype=np.int8)
    open_grid[8:172, 8:212] = 0
    square_clip = [(3.0, 3.0), (13.0, 3.0), (13.0, 13.0), (3.0, 13.0)]
    yield "manual_square", open_grid, (4.0, 4.0), square_clip
    l_clip = [
        (2.0, 2.0), (8.0, 2.0), (8.0, 8.0),
        (16.0, 8.0), (16.0, 15.0), (2.0, 15.0),
    ]
    yield "manual_l", open_grid, (4.0, 4.0), l_clip

    for seed in range(random_count):
        rng = np.random.default_rng(seed)
        grid = np.full((180, 240), 100, dtype=np.int8)
        grid[8:172, 8:232] = 0
        for _ in range(12):
            obstacle_height = int(rng.integers(4, 13))
            obstacle_width = int(rng.integers(4, 13))
            row = int(rng.integers(20, 160 - obstacle_height))
            col = int(rng.integers(35, 220 - obstacle_width))
            grid[row:row + obstacle_height, col:col + obstacle_width] = 100
        yield f"random_obstacles_{seed:02d}", grid, world(25, 20), None

    # Random user selections exercise the exact UI use case rather than only
    # changing obstacles inside one fixed full-map rectangle.  Each seed emits
    # one rectangle and one large L with a few static islands inside the clip.
    for seed in range(random_manual_count):
        rng = np.random.default_rng(10_000 + seed)
        grid = np.full((220, 280), 100, dtype=np.int8)
        grid[5:215, 5:275] = 0

        x0 = float(rng.uniform(1.5, 4.0))
        y0 = float(rng.uniform(1.5, 4.0))
        rect_width = float(rng.uniform(8.0, 17.0))
        rect_height = float(rng.uniform(7.0, 15.0))
        rectangle = [
            (x0, y0), (x0 + rect_width, y0),
            (x0 + rect_width, y0 + rect_height),
            (x0, y0 + rect_height),
        ]
        yield (
            f"random_manual_rectangle_{seed:02d}", grid.copy(),
            (x0 + 1.4, y0 + 1.4), rectangle)

        lx0 = float(rng.uniform(1.5, 3.0))
        ly0 = float(rng.uniform(1.5, 3.0))
        stem_width = float(rng.uniform(4.5, 8.0))
        total_width = float(rng.uniform(stem_width + 5.0, 20.0))
        total_height = float(rng.uniform(11.0, 18.0))
        elbow_height = float(rng.uniform(4.5, total_height - 4.0))
        l_polygon = [
            (lx0, ly0),
            (lx0 + stem_width, ly0),
            (lx0 + stem_width, ly0 + elbow_height),
            (lx0 + total_width, ly0 + elbow_height),
            (lx0 + total_width, ly0 + total_height),
            (lx0, ly0 + total_height),
        ]
        l_grid = grid.copy()
        # Add body-relevant, but comfortably passable, furniture islands.
        for _ in range(int(rng.integers(1, 4))):
            obstacle_height = int(rng.integers(4, 9))
            obstacle_width = int(rng.integers(4, 9))
            row = int(rng.integers(
                int((ly0 + 2.5) / RESOLUTION),
                int((ly0 + total_height - 2.0) / RESOLUTION)))
            col = int(rng.integers(
                int((lx0 + 2.5) / RESOLUTION),
                int((lx0 + stem_width - 1.5) / RESOLUTION)))
            l_grid[
                row:row + obstacle_height,
                col:col + obstacle_width,
            ] = 100
        yield (
            f"random_manual_l_{seed:02d}", l_grid,
            (lx0 + 1.2, ly0 + 1.2), l_polygon)

    # Randomised office graphs vary room count, door placement, corridor width
    # and furniture.  They expose ordering/retrace failures that an open field
    # with isolated boxes cannot reveal.
    for seed in range(random_office_count):
        rng = np.random.default_rng(20_000 + seed)
        height, width = 230, 300
        grid = np.full((height, width), 100, dtype=np.int8)
        corridor_width = int(rng.integers(34, 51))
        corridor_c0 = width // 2 - corridor_width // 2
        corridor_c1 = corridor_c0 + corridor_width
        grid[10:height - 10, corridor_c0:corridor_c1] = 0
        levels = int(rng.integers(2, 4))
        usable_top, usable_bottom = 16, height - 16
        level_span = (usable_bottom - usable_top) / levels
        for level in range(levels):
            row0 = int(round(usable_top + level * level_span + 3))
            row1 = int(round(usable_top + (level + 1) * level_span - 3))
            if row1 - row0 < 28:
                continue
            left_c0 = int(rng.integers(10, 23))
            right_c1 = width - int(rng.integers(10, 23))
            wall_depth = int(rng.integers(6, 11))
            grid[row0:row1, left_c0:corridor_c0 - wall_depth] = 0
            grid[row0:row1, corridor_c1 + wall_depth:right_c1] = 0
            door_height = int(rng.integers(12, 19))
            door_row = int(rng.integers(
                row0 + door_height // 2 + 5,
                row1 - door_height // 2 - 5))
            door_r0 = door_row - door_height // 2
            door_r1 = door_r0 + door_height
            grid[door_r0:door_r1,
                 corridor_c0 - wall_depth:corridor_c0 + 1] = 0
            grid[door_r0:door_r1,
                 corridor_c1 - 1:corridor_c1 + wall_depth + 1] = 0
            for room_side in ("left", "right"):
                for _ in range(int(rng.integers(0, 3))):
                    obstacle_height = int(rng.integers(4, 10))
                    obstacle_width = int(rng.integers(4, 10))
                    obstacle_row = int(rng.integers(
                        row0 + 8, row1 - obstacle_height - 8))
                    if room_side == "left":
                        obstacle_col = int(rng.integers(
                            left_c0 + 8,
                            corridor_c0 - wall_depth
                            - obstacle_width - 8))
                    else:
                        obstacle_col = int(rng.integers(
                            corridor_c1 + wall_depth + 8,
                            right_c1 - obstacle_width - 8))
                    grid[
                        obstacle_row:obstacle_row + obstacle_height,
                        obstacle_col:obstacle_col + obstacle_width,
                    ] = 100
        robot = world(height - 25, (corridor_c0 + corridor_c1) // 2)
        yield f"random_office_{seed:02d}", grid, robot, None


def run_case(name, grid, robot, clip_polygon):
    started = time.perf_counter()
    plan = plan_partitioned_coverage(
        data=grid.tobytes(),
        width=grid.shape[1],
        height=grid.shape[0],
        resolution=RESOLUTION,
        origin_x=0.0,
        origin_y=0.0,
        robot_world=robot,
        robot_yaw=0.0,
        swath_spacing_m=coverage_swath_spacing(0.70, 0.75),
        clip_polygon=clip_polygon,
        path_step_m=0.10,
        min_swath_m=0.45,
        min_region_area_m2=3.0,
        max_regions=12,
        clean_width_m=0.70,
    )
    elapsed = time.perf_counter() - started
    repeat_metrics = segment_repeat_metrics(plan)
    crossing_result = crossing_metrics(plan)
    return {
        "name": name,
        "regions": len(plan.regions),
        "bcd_cells": sum(region.cell_count for region in plan.regions),
        "visited_regions": len(plan.visit_order),
        "swaths": len(plan.swaths),
        "points": len(plan.path),
        "length_m": round(path_length(plan.path), 3),
        "seconds": round(elapsed, 3),
        "coverage": round(plan.serviceable_coverage_ratio, 6),
        "strict_reachable_coverage": round(
            plan.reachable_coverage_ratio, 6),
        "turn_safe_coverage": round(plan.turn_safe_coverage_ratio, 6),
        "region_coverages": {
            str(region_id): round(value, 6)
            for region_id, value in plan.region_coverage_ratios.items()
        },
        "region_serviceable_coverages": {
            str(region_id): round(value, 6)
            for region_id, value in
            plan.region_serviceable_coverage_ratios.items()
        },
        "footprint_valid": plan.footprint_valid,
        "violations": plan.footprint_violation_count,
        "continuous": plan.path_continuous,
        "crossings": crossing_result["total"],
        "avoidable_crossings": crossing_result["avoidable"],
        "intentional_fill_perimeter_crossings": crossing_result[
            "intended_fill_perimeter"],
        "crossing_categories": crossing_result["categories"],
        "repeat_ratio": round(repeat_ratio(plan.path), 6),
        "avoidable_repeat_ratio": round(
            repeat_metrics["avoidable"], 6),
        "intentional_fill_perimeter_overlap_ratio": round(
            repeat_metrics["intended_fill_perimeter"], 6),
        "repeat_breakdown": repeat_metrics["breakdown"],
        "complete": plan.coverage_complete,
        "failure": plan.failure_reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random-count", type=int, default=10,
        help="number of random-obstacle cases")
    parser.add_argument(
        "--random-manual-count", type=int, default=0,
        help=("number of manual-selection seeds; each emits one rectangle "
              "and one L case"))
    parser.add_argument(
        "--random-office-count", type=int, default=0,
        help="number of random multi-room office cases")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    records = []
    for case in cases(
            max(0, args.random_count),
            max(0, args.random_manual_count),
            max(0, args.random_office_count)):
        if args.case and case[0] not in set(args.case):
            continue
        record = run_case(*case)
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    summary = {
        "cases": len(records),
        "valid": sum(record["footprint_valid"] for record in records),
        "continuous": sum(record["continuous"] for record in records),
        "above_95": sum(record["coverage"] >= 0.95 for record in records),
        "minimum_coverage": min(record["coverage"] for record in records),
        "total_crossings": sum(record["crossings"] for record in records),
        "total_avoidable_crossings": sum(
            record["avoidable_crossings"] for record in records),
        "maximum_avoidable_crossings": max(
            record["avoidable_crossings"] for record in records),
        "maximum_repeat_ratio": max(record["repeat_ratio"] for record in records),
        "maximum_avoidable_repeat_ratio": max(
            record["avoidable_repeat_ratio"] for record in records),
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"records": records, "summary": summary}, indent=2) + "\n")


if __name__ == "__main__":
    main()
