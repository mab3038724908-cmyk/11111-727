"""闭环补扫: 从已扫/未扫掩码找出"机器人可达但未清扫"的连通区域 -> 世界多边形,
供 LawnMower 以 clip_polygon 重新规划补扫。纯 numpy, 可离线测试。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np

from td25a_robot_ui.algorithms.free_space import (
    dilate_binary, _flood_from, largest_component, trace_outer_contour,
    simplify_polygon, pixel_to_world,
)

Point = Tuple[float, float]


@dataclass
class UncoveredRegion:
    polygon: List[Point]
    area_m2: float
    centroid: Point


def find_uncovered_regions(counts, target_mask, reachable_mask,
                            resolution, origin_x, origin_y,
                            min_area_m2=0.5, simplify_eps_m=0.10):
    """可达且未清扫的连通区域 -> 世界多边形(外扩 1 格, 抵消格心描边收缩)。

    单遍连通域: 直接 flood 每个分量并按 min_cells 过滤, 不再 filter_clusters_by_size
    先 flood 一遍再 flood 一遍(避免大图上双重 flood)。
    """
    uncovered = target_mask & (counts == 0) & reachable_mask
    if not uncovered.any():
        return []
    cell_area = resolution * resolution
    min_cells = max(1, int(round(min_area_m2 / cell_area)))
    height = counts.shape[0]
    # eps 不能超过 ~半格: 膨胀只外扩 1 格, 若 simplify_eps_m 比格距还大,
    # Douglas-Peucker 会把小/窄区域的轮廓塌成 <3 点而被静默丢弃(见 len(poly)<3)。
    # 把 eps 夹到 resolution*0.5 以内, 保证膨胀后的轮廓在任何分辨率下都活下来。
    eff_eps = min(simplify_eps_m, resolution * 0.5)
    regions: List[UncoveredRegion] = []
    remaining = uncovered.copy()
    while remaining.any():
        ys, xs = np.where(remaining)
        cc = _flood_from(remaining, int(ys[0]), int(xs[0]))
        remaining &= ~cc
        n = int(cc.sum())
        if n < min_cells:
            continue
        # 外扩 1 格再描轮廓: trace_outer_contour 走格心, 重新按格心 clip 会丢边缘行/列;
        # 先膨胀 1 格让 clip 多边形把原始未扫格全包进去, 否则小/窄区域补不全。
        # 注: 膨胀的 1 格 halo 可能越过 reachable 边界, 让 clip 多边形名义上圈进几个
        # 不可达(墙/膨胀障碍)格。这是**良性**的: plan_lawnmower_coverage 内
        # build_coverage_free_mask 把 clip_polygon 与膨胀后的自由域做 AND
        # (free &= polygon_to_mask(clip_polygon)), clip 只能**缩小**可走域、不能把
        # 不可达格加进规划; 连接段走的是 travel_reachable(自由域连通分量)而非 clip 多边形。
        # 故不在此裁回 reachable —— 裁回反而会丢边界那 1 格的覆盖(违背膨胀本意)。
        contour = trace_outer_contour(dilate_binary(cc, 1))
        if len(contour) < 3:
            continue
        poly = simplify_polygon(
            pixel_to_world(contour, origin_x, origin_y, resolution, height),
            eff_eps)
        if len(poly) < 3:
            continue
        cys, cxs = np.where(cc)
        cen = (origin_x + (float(cxs.mean()) + 0.5) * resolution,
               origin_y + (float(cys.mean()) + 0.5) * resolution)
        regions.append(UncoveredRegion(polygon=poly, area_m2=n * cell_area,
                                       centroid=cen))
    return regions


def reachable_from_robot(reachable_mask, robot_yx):
    """只保留机器人当前所在的连通自由域, 排除被障碍隔开、其实到不了的"孤岛"。

    否则补扫会对孤岛里的未扫区规划一条穿障直线连接段(planner 的 _connect_points
    在 A*/snap 失败时会退化成densified直线)。robot_yx 不在自由域内时 largest_component
    退回到最大连通分量(安全兜底)。

    连通性说明: flood-fill 用 4-连通, 看似比 8-连通的 A* 保守(可能漏掉只能斜向到的格)。
    但 grid_coverage._astar 的对角移动有"不切角"约束(line 659-661: 对角邻格的两个
    正交邻格必须都自由), 所以纯斜向通道(staircase)A* 同样过不去 —— 4-连通 flood 与
    A* 实际可达域一致。这里**故意**保持 4-连通: 若改 8-连通, 会把 A* 其实到不了的
    斜向孤格标成可达, 补扫会规划一条穿障连接段。不要改成 8-连通。
    """
    return largest_component(reachable_mask, seed=robot_yx)


def reachable_coverage_ratio(counts, target_mask, reachable_mask) -> float:
    """覆盖率只按"机器人可达的目标格"算(排除墙边够不着那圈), 返回 0..1。"""
    tgt = target_mask & reachable_mask
    total = int(tgt.sum())
    if total == 0:
        return 1.0
    cleaned = int((tgt & (counts >= 1)).sum())
    return cleaned / total


def order_regions_nearest(regions, start_xy):
    remaining = list(regions)
    ordered = []
    cur = start_xy
    while remaining:
        i = min(range(len(remaining)),
                key=lambda k: math.hypot(remaining[k].centroid[0] - cur[0],
                                          remaining[k].centroid[1] - cur[1]))
        r = remaining.pop(i)
        ordered.append(r)
        cur = r.centroid
    return ordered


def reclean_should_stop(ratio, last_ratio, pass_count, n_regions,
                         target_ratio=0.95, max_passes=3, min_gain=0.01):
    if n_regions == 0:
        return True, "no_uncovered"
    if ratio >= target_ratio:
        return True, f"ratio_ok:{ratio:.3f}"
    if pass_count >= max_passes:
        return True, f"max_passes:{max_passes}"
    if pass_count >= 1 and 0 <= (ratio - last_ratio) < min_gain:
        # [2315实测] 负增长不算收敛: 覆盖率46.6%→26.9%是行人站在未扫区上把live可达区
        # 临时挤小的假回退, 按"无改善"停止会26.9%就收官; 负增长继续补(max_passes兜底)。
        return True, f"low_gain:{ratio - last_ratio:.3f}"
    return False, ""
