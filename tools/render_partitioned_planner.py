#!/usr/bin/env python3
"""Render the offline TD25A room-by-room planner; never starts ROS actions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from td25a_robot_ui.algorithms.free_space import (
    extract_clean_field_result,
    largest_component,
    polygon_to_mask,
)
from td25a_robot_ui.algorithms.grid_coverage import (
    CoverageFootprint,
    _build_radial_clearance_mask,
    build_coverage_free_mask,
    coverage_swath_spacing,
    plan_partitioned_coverage,
)


def parse_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    image = re.search(r"^image:\s*(.+)$", text, re.M).group(1).strip()
    resolution = float(re.search(r"^resolution:\s*([0-9.]+)$", text, re.M).group(1))
    origin_text = re.search(r"^origin:\s*\[([^]]+)\]", text, re.M).group(1)
    origin = [float(value.strip()) for value in origin_text.split(",")]
    return {"image": image, "resolution": resolution, "origin": origin}


def load_map(path: Path):
    config = parse_yaml(path)
    raw = Image.open(path.parent / config["image"]).convert("L")
    pixels = np.flipud(np.asarray(raw, dtype=np.uint8))
    gray_unknown = (pixels >= 203) & (pixels <= 207)
    recover_legacy = int(gray_unknown.sum()) >= max(20, int(pixels.size * 0.01))
    grid = np.full(pixels.shape, -1, dtype=np.int8)
    grid[pixels <= 89] = 100
    if recover_legacy:
        # Emulate legacy map_server first; the planner receives the saved-PNG
        # known-free mask and repairs the gray false-free cells itself.
        grid[pixels >= 203] = 0
        known_free = pixels >= 250
    else:
        grid[pixels >= 250] = 0
        known_free = None
    return config, raw, grid, known_free, int(gray_unknown.sum())


def choose_seed(grid, resolution, ox, oy, name, known_free):
    if name == "9":
        return (-1.73, -3.34), "real robot seed"
    data_grid = grid.copy()
    if known_free is not None:
        data_grid[(data_grid >= 0) & ~known_free] = -1
    turn_safe = _build_radial_clearance_mask(
        data_grid.tobytes(), grid.shape[1], grid.shape[0], resolution,
        CoverageFootprint().turn_clearance_m, origin_x=ox, origin_y=oy)
    component = largest_component(turn_safe)
    rows, cols = np.nonzero(component)
    if not cols.size:
        raise RuntimeError("no travel-safe component")
    target_col = float(np.percentile(cols, 12))
    target_row = float(np.percentile(rows, 12))
    score = (
        ((cols - target_col) / max(1.0, float(cols.max() - cols.min()))) ** 2
        + ((rows - target_row) / max(1.0, float(rows.max() - rows.min()))) ** 2
    )
    index = int(np.argmin(score))
    return (
        ox + (int(cols[index]) + 0.5) * resolution,
        oy + (int(rows[index]) + 0.5) * resolution,
    ), "offline lower-left seed"


def length(points):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points, points[1:]))


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc" if bold else
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_arrowed(draw, points, pixel, color, width, interval_m,
                 draw_line=True):
    if len(points) < 2:
        return
    if draw_line:
        draw.line([pixel(point) for point in points], fill=color, width=width,
                  joint="curve")
    next_arrow = interval_m
    walked = 0.0
    arrow_size = max(3, int(width * 1.4))
    for start, end in zip(points, points[1:]):
        segment = math.hypot(end[0] - start[0], end[1] - start[1])
        if segment < 1e-9:
            continue
        while walked + segment >= next_arrow:
            ratio = (next_arrow - walked) / segment
            point = (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
            cx, cy = pixel(point)
            sx, sy = pixel(start)
            ex, ey = pixel(end)
            dx, dy = ex - sx, ey - sy
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                ux, uy = dx / norm, dy / norm
                nx, ny = -uy, ux
                tip = (cx + ux * arrow_size, cy + uy * arrow_size)
                left = (cx - ux * arrow_size * 0.72 + nx * arrow_size * 0.58,
                        cy - uy * arrow_size * 0.72 + ny * arrow_size * 0.58)
                right = (cx - ux * arrow_size * 0.72 - nx * arrow_size * 0.58,
                         cy - uy * arrow_size * 0.72 - ny * arrow_size * 0.58)
                draw.polygon([tip, left, right], fill=color)
            next_arrow += interval_m
        walked += segment


def draw_dashed_polygon(draw, points, pixel, color="#a855f7", width=4,
                        dash_px=15.0, gap_px=9.0):
    """Draw a review-only manual-selection boundary behind the route."""
    if len(points) < 3:
        return
    pixels = [pixel(point) for point in points]
    pixels.append(pixels[0])
    for start, end in zip(pixels, pixels[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        segment = math.hypot(dx, dy)
        if segment < 1e-9:
            continue
        offset = 0.0
        while offset < segment:
            finish = min(segment, offset + dash_px)
            draw.line((
                (start[0] + dx * offset / segment,
                 start[1] + dy * offset / segment),
                (start[0] + dx * finish / segment,
                 start[1] + dy * finish / segment),
            ), fill=color, width=width)
            offset += dash_px + gap_px


def find_simple_lane_transition(segments):
    """Pick one unobstructed adjacent-lane change for the turn callout."""
    best = None
    best_score = float("-inf")
    for segment in segments:
        if segment.kind != "fill":
            continue
        for current, following in zip(segment.swaths, segment.swaths[1:]):
            start, finish = current
            next_start, next_finish = following
            current_length = length((start, finish))
            next_length = length((next_start, next_finish))
            gap = length((finish, next_start))
            if min(current_length, next_length) < 2.0 or not 0.25 <= gap <= 0.90:
                continue
            ux = (finish[0] - start[0]) / current_length
            uy = (finish[1] - start[1]) / current_length
            vx = (next_finish[0] - next_start[0]) / next_length
            vy = (next_finish[1] - next_start[1]) / next_length
            gx = (next_start[0] - finish[0]) / gap
            gy = (next_start[1] - finish[1]) / gap
            # A real boustrophedon change: next lane reverses direction and the
            # short connector is approximately perpendicular to both lanes.
            if ux * vx + uy * vy > -0.80 or abs(ux * gx + uy * gy) > 0.25:
                continue
            score = min(current_length, next_length) - abs(gap - 0.525) * 4.0
            if score > best_score:
                best_score = score
                best = (finish, next_start, gap)
    return best


def split_fill_connectors(segment):
    """Return the executed paths between consecutive ordered swaths."""
    if segment.kind != "fill" or not segment.path or not segment.swaths:
        return []
    connectors = []
    cursor = 0

    def find_point(target, start_index):
        for index in range(start_index, len(segment.path)):
            if length((segment.path[index], target)) <= 1e-5:
                return index
        return None

    for start, finish in segment.swaths:
        start_index = find_point(start, cursor)
        if start_index is None:
            continue
        if start_index > cursor:
            connector = segment.path[cursor:start_index + 1]
            if length(connector) > 1e-6:
                connectors.append(connector)
        finish_index = find_point(finish, start_index)
        if finish_index is None:
            continue
        cursor = finish_index
    return connectors


def draw_number_marker(draw, point, pixel, label, scale,
                       outline_color="#facc15"):
    x, y = pixel(point)
    radius = max(9, int(4.0 * scale))
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 fill="white", outline=outline_color,
                 width=max(2, int(scale)))
    label_font = font(max(12, int(5.0 * scale)), True)
    if hasattr(draw, "textbbox"):
        box = draw.textbbox((0, 0), label, font=label_font)
        label_w, label_h = box[2] - box[0], box[3] - box[1]
    else:
        label_w, label_h = draw.textsize(label, font=label_font)
    draw.text((x - label_w / 2, y - label_h / 2 - 1), label,
              font=label_font, fill="#07111f")


def render(yaml_path: Path, output: Path, clip_polygon=None,
           seed_override=None, selection_label=None, turn_detail=False,
           focus_selection=False, yaw_override=None) -> dict:
    name = yaml_path.stem
    config, raw, grid, known_free, recovered = load_map(yaml_path)
    resolution = config["resolution"]
    ox, oy = config["origin"][:2]
    height, width = grid.shape
    if seed_override is None:
        seed, seed_note = choose_seed(
            grid, resolution, ox, oy, name, known_free)
    else:
        seed = tuple(seed_override)
        seed_note = "manual-selection seed"
    extraction = extract_clean_field_result(
        grid.tobytes(), width, height, resolution, ox, oy,
        robot_radius_m=0.34, seed_world=seed,
        known_free_mask=known_free, max_seed_snap_m=2.0,
        recovered_unknown_cells=recovered,
    )
    if not extraction.polygon:
        raise RuntimeError(extraction.failure_reason)
    planning_polygon = clip_polygon or extraction.polygon
    started = time.perf_counter()
    plan = plan_partitioned_coverage(
        grid.tobytes(), width, height, resolution, ox, oy,
        robot_world=seed,
        robot_yaw=yaw_override,
        swath_spacing_m=coverage_swath_spacing(0.70, 0.75),
        clip_polygon=planning_polygon,
        selection_boundary_polygon=planning_polygon,
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

    view_c0, view_c1 = 0, width - 1
    view_r0, view_r1 = 0, height - 1
    if focus_selection and clip_polygon:
        polygon_cols = [(point[0] - ox) / resolution for point in clip_polygon]
        polygon_rows = [(point[1] - oy) / resolution for point in clip_polygon]
        padding = max(4, int(math.ceil(1.20 / resolution)))
        view_c0 = max(0, int(math.floor(min(polygon_cols))) - padding)
        view_c1 = min(width - 1, int(math.ceil(max(polygon_cols))) + padding)
        view_r0 = max(0, int(math.floor(min(polygon_rows))) - padding)
        view_r1 = min(height - 1, int(math.ceil(max(polygon_rows))) + padding)
    view_width = view_c1 - view_c0 + 1
    view_height = view_r1 - view_r0 + 1
    scale = min(1220 / view_width, 990 / view_height)
    map_w, map_h = int(view_width * scale), int(view_height * scale)
    margin_x, top, panel = 40, (154 if turn_detail else 116), 248
    canvas = Image.new("RGB", (map_w + margin_x * 2, map_h + top + panel), "#08111f")
    resampling = getattr(Image, "Resampling", Image)
    raw_crop = raw.crop((
        view_c0,
        height - 1 - view_r1,
        view_c1 + 1,
        height - view_r0,
    ))
    map_image = raw_crop.convert("RGB").resize(
        (map_w, map_h), resampling.NEAREST)
    canvas.paste(map_image, (margin_x, top))
    draw = ImageDraw.Draw(canvas)

    def pixel(point):
        col = (point[0] - ox) / resolution
        row_from_bottom = (point[1] - oy) / resolution
        return (
            margin_x + int(round((col - view_c0) * scale)),
            top + int(round((view_r1 - row_from_bottom) * scale)),
        )

    robot_blue = "#0ea5e9"
    red = "#ef233c"
    yellow = "#facc15"
    if clip_polygon:
        draw_dashed_polygon(draw, clip_polygon, pixel)
    # Operator-requested review palette: every fill/turn/transfer movement is
    # one clearly visible yellow route; red is only the edge-clean phase.
    lane_width = max(2 if turn_detail else 1,
                     int((1.2 if turn_detail else 0.85) * scale))
    connector_width = max(1, int((0.8 if turn_detail else 0.65) * scale))
    for segment in plan.segments:
        if segment.kind == "fill":
            # Paint the execution path once so its connectors remain visible.
            # Connector arrows are added separately below, one per executed
            # connector, so a long shared corridor stays readable.
            draw.line([pixel(point) for point in segment.path], fill=yellow,
                      width=connector_width, joint="curve")
            for lane in segment.swaths:
                # Each sweep is rendered exactly once and receives one arrow
                # at 60% of its length, preserving its execution direction
                # without stacking arrowheads on a shared connector.
                draw_arrowed(
                    draw, lane, pixel, yellow, lane_width,
                    max(0.01, length(lane) * 0.60),
                )
            for connector in split_fill_connectors(segment):
                draw_arrowed(
                    draw, connector, pixel, yellow, connector_width,
                    max(0.01, length(connector) * 0.60),
                    draw_line=False,
                )
    for segment in plan.segments:
        if segment.kind == "perimeter":
            draw_arrowed(
                draw, segment.path, pixel, red, max(3, int(2.8 * scale)),
                max(1.2, length(segment.path) / 6.0),
            )
    for segment in plan.segments:
        if segment.kind == "transfer":
            draw_arrowed(
                draw, segment.path, pixel, yellow, connector_width,
                max(0.28, length(segment.path) / 2.0),
            )

    transition_example = find_simple_lane_transition(plan.segments) if turn_detail else None
    if transition_example is not None:
        finish, next_start, _ = transition_example
        midpoint = (
            0.5 * (finish[0] + next_start[0]),
            0.5 * (finish[1] + next_start[1]),
        )
        draw_number_marker(draw, finish, pixel, "1", scale, yellow)
        draw_number_marker(draw, midpoint, pixel, "2", scale, yellow)
        draw_number_marker(draw, next_start, pixel, "3", scale, yellow)

    order_index = {region_id: index + 1 for index, region_id in enumerate(plan.visit_order)}
    for region in plan.regions:
        if region.region_id not in order_index:
            continue
        x, y = pixel(region.centroid)
        radius = max(8, int(6 * scale))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill="#ffffff", outline="#07111f", width=2)
        label = str(order_index[region.region_id])
        label_font = font(max(12, int(9 * scale)), True)
        if hasattr(draw, "textbbox"):
            box = draw.textbbox((0, 0), label, font=label_font)
            label_w, label_h = box[2] - box[0], box[3] - box[1]
        else:  # Pillow 7.x on the remote benchmark server.
            label_w, label_h = draw.textsize(label, font=label_font)
        draw.text((x - label_w / 2, y - label_h / 2 - 1),
                  label, font=label_font, fill="#07111f")

    sx, sy = pixel(seed)
    radius = max(8, int(6 * scale))
    draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius),
                 fill=robot_blue, outline="white", width=3)

    preview_name = selection_label or name
    draw.text((40, 20), f"{preview_name}｜分区规划预览｜executing=false",
              font=font(29, True), fill="white")
    legend = "黄色＝填充/换行/区域连接   红色＝沿边清扫   蓝点＝机器人起点   箭头＝前进方向"
    if clip_polygon:
        legend += "   紫色虚线＝手动框选边界"
    draw.text((40, 66), legend,
              font=font(17), fill="#cbd5e1")
    if turn_detail:
        draw.text(
            (40, 100),
            "换行示例：①线尾停车原地转90°  →  ②沿黄线前进0.525米  →  ③原地转90°进入下一条",
            font=font(16), fill="#e2e8f0")
    panel_y = top + map_h
    draw.rectangle((0, panel_y, canvas.width, canvas.height), fill="#111b2d")
    status = "PASS" if plan.footprint_valid else "FAIL"
    status_color = (
        "#4ade80" if plan.footprint_valid and plan.coverage_complete
        else "#fbbf24" if plan.footprint_valid else "#fb7185")
    completion = "COMPLETE" if plan.coverage_complete else "INCOMPLETE"
    selected_area_m2 = extraction.reachable_area_m2
    if clip_polygon:
        selected_mask = polygon_to_mask(
            clip_polygon, width, height, resolution, ox, oy)
        selected_area_m2 = float((plan.free_mask & selected_mask).sum()) * resolution ** 2
    area_label = "selected reachable" if clip_polygon else "reachable"
    lines = [
        f"{area_label} {selected_area_m2:.1f}m² | {len(plan.regions)} cleaning regions | "
        f"visit order {' → '.join(map(str, range(1, len(plan.visit_order) + 1)))}",
        f"{len(plan.swaths)} uniform lanes | {len(plan.path)} points | {length(plan.path):.1f}m | "
        f"offline planning {elapsed:.2f}s | seed {seed[0]:.2f},{seed[1]:.2f} ({seed_note})",
        f"REAL 1.05m x 0.68m BODY SWEEP: {status} ({plan.footprint_violation_count} violations) | "
        f"COVERAGE: {completion} ({len(plan.visit_order)}/{len(plan.regions)} regions)",
        f"PATH CONTINUITY: {'PASS' if plan.path_continuous else 'FAIL'} "
        f"(max gap {plan.max_segment_gap_m:.3f}m) | TURN-SAFE CORE {plan.turn_safe_coverage_ratio:.1%} | "
        f"SERVICEABLE FLOOR {plan.serviceable_coverage_ratio:.1%} | "
        f"STRICT CENTER-REACHABLE {plan.reachable_coverage_ratio:.1%}",
        f"diagnostic: {plan.failure_reason or 'none'}",
    ]
    for index, line in enumerate(lines):
        draw.text((40, panel_y + 20 + index * 39), line,
                  font=font(17, index == 2),
                  fill=status_color if index == 2 else "#e2e8f0")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {
        "name": name,
        "regions": len(plan.regions),
        "swaths": len(plan.swaths),
        "points": len(plan.path),
        "length_m": length(plan.path),
        "seconds": elapsed,
        "footprint_valid": plan.footprint_valid,
        "coverage_complete": plan.coverage_complete,
        "path_continuous": plan.path_continuous,
        "max_segment_gap_m": plan.max_segment_gap_m,
        "turn_safe_coverage_ratio": plan.turn_safe_coverage_ratio,
        "serviceable_coverage_ratio": plan.serviceable_coverage_ratio,
        "reachable_coverage_ratio": plan.reachable_coverage_ratio,
        "violations": plan.footprint_violation_count,
        "failure_reason": plan.failure_reason,
        "selection_label": selection_label,
        "turn_detail": turn_detail,
        "turn_example_gap_m": (
            transition_example[2] if transition_example is not None else None),
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--clip-json",
        help="manual selection polygon as JSON [[x,y], ...]",
    )
    parser.add_argument("--seed", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument(
        "--yaw-deg", type=float,
        help="initial robot heading in degrees for offline preview",
    )
    parser.add_argument("--label")
    parser.add_argument("--turn-detail", action="store_true")
    parser.add_argument("--focus-selection", action="store_true")
    args = parser.parse_args()
    clip_polygon = json.loads(args.clip_json) if args.clip_json else None
    print(render(
        args.yaml, args.output,
        clip_polygon=clip_polygon,
        seed_override=args.seed,
        selection_label=args.label,
        turn_detail=args.turn_detail,
        focus_selection=args.focus_selection,
        yaw_override=(math.radians(args.yaw_deg)
                      if args.yaw_deg is not None else None),
    ))


if __name__ == "__main__":
    main()
