#!/usr/bin/env python3
"""Render yellow fill/connector and red edge-clean previews for synthetic cases."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/td25a_robot_ui"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_synthetic_coverage import cases  # noqa: E402
from td25a_robot_ui.algorithms.grid_coverage import (  # noqa: E402
    coverage_swath_spacing,
    plan_partitioned_coverage,
)


RESOLUTION = 0.10


def font(size, bold=False):
    paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in paths:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def length(path):
    return sum(math.hypot(
        second[0] - first[0], second[1] - first[1])
        for first, second in zip(path, path[1:]))


def draw_arrowed(draw, path, pixel, color, width, interval_m=2.0):
    if len(path) < 2:
        return
    draw.line([pixel(point) for point in path], fill=color, width=width,
              joint="curve")
    walked = 0.0
    next_arrow = interval_m
    size = max(6, width * 2)
    for first, second in zip(path, path[1:]):
        segment = math.hypot(
            second[0] - first[0], second[1] - first[1])
        if segment <= 1e-8:
            continue
        while walked + segment >= next_arrow:
            ratio = (next_arrow - walked) / segment
            point = (
                first[0] + (second[0] - first[0]) * ratio,
                first[1] + (second[1] - first[1]) * ratio,
            )
            x, y = pixel(point)
            x0, y0 = pixel(first)
            x1, y1 = pixel(second)
            dx, dy = x1 - x0, y1 - y0
            norm = math.hypot(dx, dy)
            if norm:
                ux, uy = dx / norm, dy / norm
                nx, ny = -uy, ux
                draw.polygon([
                    (x + ux * size, y + uy * size),
                    (x - ux * size * 0.7 + nx * size * 0.55,
                     y - uy * size * 0.7 + ny * size * 0.55),
                    (x - ux * size * 0.7 - nx * size * 0.55,
                     y - uy * size * 0.7 - ny * size * 0.55),
                ], fill=color)
            next_arrow += interval_m
        walked += segment


def render(case_name, output):
    random_count = 10
    random_manual_count = 0
    random_office_count = 0
    try:
        seed_count = int(case_name.rsplit("_", 1)[1]) + 1
    except (IndexError, ValueError):
        seed_count = 0
    if case_name.startswith("random_obstacles_"):
        random_count = max(random_count, seed_count)
    elif case_name.startswith("random_manual_"):
        random_manual_count = seed_count
    elif case_name.startswith("random_office_"):
        random_office_count = seed_count
    selected = None
    for case in cases(
            random_count, random_manual_count, random_office_count):
        if case[0] == case_name:
            selected = case
            break
    if selected is None:
        raise ValueError(f"unknown case: {case_name}")
    name, grid, robot, clip_polygon = selected
    plan = plan_partitioned_coverage(
        data=grid.tobytes(), width=grid.shape[1], height=grid.shape[0],
        resolution=RESOLUTION, origin_x=0.0, origin_y=0.0,
        robot_world=robot, robot_yaw=0.0,
        swath_spacing_m=coverage_swath_spacing(0.70, 0.75),
        clip_polygon=clip_polygon,
        path_step_m=0.10, min_swath_m=0.45,
        min_region_area_m2=3.0, max_regions=12, clean_width_m=0.70)

    rows, cols = np.nonzero(grid <= 50)
    row_min, row_max = max(0, int(rows.min()) - 8), min(
        grid.shape[0] - 1, int(rows.max()) + 8)
    col_min, col_max = max(0, int(cols.min()) - 8), min(
        grid.shape[1] - 1, int(cols.max()) + 8)
    scale = 5
    top = 150
    bottom = 90
    map_width = (col_max - col_min + 1) * scale
    map_height = (row_max - row_min + 1) * scale
    image = Image.new("RGB", (map_width, top + map_height + bottom), "#07111f")
    crop = grid[row_min:row_max + 1, col_min:col_max + 1]
    pixels = np.zeros((*crop.shape, 3), dtype=np.uint8)
    pixels[crop > 50] = (32, 37, 45)
    pixels[crop <= 50] = (235, 238, 240)
    map_image = Image.fromarray(np.flipud(pixels), "RGB").resize(
        (map_width, map_height), Image.Resampling.NEAREST)
    image.paste(map_image, (0, top))
    draw = ImageDraw.Draw(image)

    def pixel(point):
        col = point[0] / RESOLUTION - col_min
        row = point[1] / RESOLUTION - row_min
        return (col * scale, top + (row_max - row_min + 1 - row) * scale)

    for segment in plan.segments:
        color = "#ef2b45" if segment.kind == "perimeter" else "#facc15"
        draw_arrowed(draw, segment.path, pixel, color,
                     4 if segment.kind == "perimeter" else 5)
    start_x, start_y = pixel(robot)
    draw.ellipse((start_x - 8, start_y - 8, start_x + 8, start_y + 8),
                 fill="#12b8e5", outline="white", width=2)
    for order, region_id in enumerate(plan.visit_order, start=1):
        region = next(region for region in plan.regions
                      if region.region_id == region_id)
        x, y = pixel(region.centroid)
        draw.ellipse((x - 13, y - 13, x + 13, y + 13),
                     fill="white", outline="#facc15", width=2)
        draw.text((x - 5, y - 10), str(order), font=font(15, True), fill="#07111f")

    draw.text((28, 20), f"{name} · BCD分区覆盖预览",
              font=font(30, True), fill="white")
    draw.text((28, 70), "黄线=填充/换行/区间连接  红线=沿边清扫  箭头=前进方向",
              font=font(20), fill="#d7dde8")
    draw.text((28, top + map_height + 22),
              f"serviceable coverage {plan.serviceable_coverage_ratio * 100:.2f}%  |  "
              f"body collision {plan.footprint_violation_count}  |  "
              f"regions {len(plan.regions)} / cells "
              f"{sum(region.cell_count for region in plan.regions)}  |  "
              f"length {length(plan.path):.1f}m",
              font=font(20, True), fill="#facc15")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.case, args.output)


if __name__ == "__main__":
    main()
