"""从 OccupancyGrid 提取"可清扫区域"多边形。

标准 vacuum-robot 做法(Galceran 2013 / Choset BCD / Roborock):
  1. 二值化 occupancy → free 像素 mask
  2. 按机器人半径做形态学腐蚀,留安全距离
  3. flood-fill 从种子点(机器人位置或最大 bbox 中心)算连通域
  4. Moore-neighbor 边界跟踪得到外轮廓(像素 polyline)
  5. Douglas-Peucker 简化到 ~30-100 个顶点
  6. 像素坐标 → 世界坐标(米)

实现纯 numpy(机器人上 cv2/scipy 被 numpy 2.x 打坏了,不依赖)。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os
from typing import List, Optional, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass
class CleanFieldResult:
    """Result of robot-seeded clean-field extraction.

    ``polygon`` is deliberately empty on any ambiguity that could otherwise
    select a disconnected large region.  The remaining fields are lightweight
    diagnostics suitable for the UI log / annotation metadata.
    """
    polygon: List[Point] = field(default_factory=list)
    reachable_area_m2: float = 0.0
    reachable_cells: int = 0
    safe_cells: int = 0
    seed_cell: Optional[Tuple[int, int]] = None
    snapped_seed_cell: Optional[Tuple[int, int]] = None
    seed_snap_m: float = 0.0
    used_open_guard: bool = False
    recovered_unknown_cells: int = 0
    failure_reason: str = ""


# ---------- 步骤 1-2:mask + 腐蚀 ----------

def free_mask_from_grid(data: bytes, width: int, height: int,
                         free_thresh: int = 50,
                         unknown_as_obstacle: bool = True) -> np.ndarray:
    """OccupancyGrid.data → (H, W) bool mask。

    -1 unknown, 0 free, 100 obstacle (Nav2 惯例)。
    保守起见把 unknown 当障碍(机器人不进未探区)。
    """
    g = np.frombuffer(data, dtype=np.int8).reshape(height, width)
    if unknown_as_obstacle:
        return (g >= 0) & (g <= free_thresh)
    return g <= free_thresh


def erode_binary(mask: np.ndarray, iterations: int) -> np.ndarray:
    """4-连通二值腐蚀 iterations 次。地图边界视为障碍。"""
    out = mask
    for _ in range(max(0, iterations)):
        new = np.zeros_like(out)
        new[1:-1, 1:-1] = (out[1:-1, 1:-1]
                            & out[:-2, 1:-1]
                            & out[2:, 1:-1]
                            & out[1:-1, :-2]
                            & out[1:-1, 2:])
        out = new
    return out


def dilate_binary(mask: np.ndarray, iterations: int) -> np.ndarray:
    """4-连通二值膨胀。配合 erode 做开/闭运算去噪。"""
    out = mask
    for _ in range(max(0, iterations)):
        new = out.copy()
        new[1:, :] |= out[:-1, :]
        new[:-1, :] |= out[1:, :]
        new[:, 1:] |= out[:, :-1]
        new[:, :-1] |= out[:, 1:]
        out = new
    return out


def open_binary(mask: np.ndarray, n: int) -> np.ndarray:
    """形态学开运算 = 先腐蚀再膨胀;去掉 ≤ (2n+1)² 的孤立块,保留大结构形状。"""
    return dilate_binary(erode_binary(mask, n), n)


def extract_obstacle_cutouts(data: bytes, width: int, height: int,
                              resolution: float, origin_x: float, origin_y: float,
                              robot_radius_m: float = 0.40,
                              denoise_open_cells: int = 1,
                              min_cluster_cells: int = 30,
                              merge_cells: int = 4,
                              max_cutouts: int = 30,
                              ) -> List[List[Point]]:
    """每个障碍连通簇 → 凸包 + pad(robot_radius)→ cutout 多边形。

    流程:
      1. 同一套去噪(open + min_cluster) — 必须和 extract_clean_field 一致,
         否则 cutouts 会落在 outer 外面被 F2C silent reject。
      2. dilate(merge_cells)合并相邻 ≤ 2*merge_cells 距离的簇,降低 cutout 总数。
      3. flood-fill 取连通簇 → 凸包 → 按 robot_radius pad → 一个 cutout。
      4. 数量超 max_cutouts(F2C ~40+ polygons 会崩)时,按面积排序留最大的。
    """
    grid = np.frombuffer(data, dtype=np.int8).reshape(height, width)
    obstacle = grid > 50
    obstacle = open_binary(obstacle, denoise_open_cells)
    obstacle = filter_clusters_by_size(obstacle, min_cluster_cells)
    # 合并邻近障碍:dilate 再回到 obstacle 上界 — 簇虽合并但凸包仍基于原始像素
    merged = dilate_binary(obstacle, merge_cells)
    raw_cutouts: List[List[Point]] = []
    remaining = merged.copy()
    while remaining.any():
        ys, xs = np.where(remaining)
        sy, sx = int(ys[0]), int(xs[0])
        cluster = _flood_from(remaining, sy, sx)
        # 凸包基于原始障碍(没膨胀)的像素,保证不"长大"超 outer
        orig_cluster = cluster & obstacle
        cys, cxs = np.where(orig_cluster if orig_cluster.any() else cluster)
        xy = [
            (origin_x + (int(x) + 0.5) * resolution,
             origin_y + (int(y) + 0.5) * resolution)
            for y, x in zip(cys, cxs)
        ]
        hull = _convex_hull_xy(xy)
        if len(hull) >= 3:
            raw_cutouts.append(_pad_polygon_outward(hull, robot_radius_m))
        remaining &= ~cluster
    if len(raw_cutouts) <= max_cutouts:
        return raw_cutouts
    # 太多了:留 max_cutouts 个面积最大的
    def area(poly):
        n = len(poly)
        return 0.5 * abs(sum(poly[i][0] * poly[(i + 1) % n][1]
                              - poly[(i + 1) % n][0] * poly[i][1]
                              for i in range(n)))
    return sorted(raw_cutouts, key=area, reverse=True)[:max_cutouts]


def generate_free_swaths(data: bytes, width: int, height: int,
                          resolution: float, origin_x: float, origin_y: float,
                          robot_radius_m: float = 0.40,
                          swath_spacing_m: float = 0.70,
                          axis: str = "x",
                          min_swath_m: float = 0.50,
                          denoise_open_cells: int = 0,
                          clip_polygon: Optional[List[Point]] = None,
                          ) -> List[Tuple[Point, Point]]:
    """直接在自由空间像素上生成 boustrophedon swaths。

    每条 swath 物理上保证在 free 像素上 → 不可能穿越障碍(连 1-pixel 噪点都
    被膨胀进 not_walkable)。逐行扫描遇障碍自然切段。

    - denoise_open_cells=0:不开运算,每个 obstacle pixel 都算
    - 边界腐蚀用 ceiling 保证 ≥ robot_radius 物理距离
    - clip_polygon:可选,只保留多边形内的 swath 段
    """
    import math as _math
    grid = np.frombuffer(data, dtype=np.int8).reshape(height, width)
    obstacle = grid > 50
    if denoise_open_cells > 0:
        obstacle = open_binary(obstacle, denoise_open_cells)
    pad_cells = max(1, _math.ceil(robot_radius_m / resolution))
    not_walkable = dilate_binary(obstacle, pad_cells)
    unknown = grid < 0
    not_walkable = not_walkable | unknown
    free = ~not_walkable
    free[:pad_cells, :] = False; free[-pad_cells:, :] = False
    free[:, :pad_cells] = False; free[:, -pad_cells:] = False

    spacing_cells = max(1, int(round(swath_spacing_m / resolution)))
    min_run_cells = max(2, int(round(min_swath_m / resolution)))

    swaths_pix: List[Tuple[int, int, int]] = []  # (row, col_start, col_end) 或 (col, row_start, row_end)
    if axis == "x":
        # 水平 swath,沿 x 扫,每隔 spacing 一行
        for y in range(spacing_cells // 2, height, spacing_cells):
            row = free[y, :]
            in_run = False; rs = 0
            for x in range(width):
                if row[x] and not in_run:
                    in_run = True; rs = x
                elif (not row[x]) and in_run:
                    in_run = False
                    if x - rs >= min_run_cells:
                        swaths_pix.append((y, rs, x - 1))
            if in_run and width - rs >= min_run_cells:
                swaths_pix.append((y, rs, width - 1))
    else:
        # 竖直 swath
        for x in range(spacing_cells // 2, width, spacing_cells):
            col = free[:, x]
            in_run = False; rs = 0
            for y in range(height):
                if col[y] and not in_run:
                    in_run = True; rs = y
                elif (not col[y]) and in_run:
                    in_run = False
                    if y - rs >= min_run_cells:
                        swaths_pix.append((x, rs, y - 1))
            if in_run and height - rs >= min_run_cells:
                swaths_pix.append((x, rs, height - 1))

    # 真 S 形:按 row 分组,每行内的 seg 按"接续上一行末端最近"排,
    # 行内每个 seg 方向也跟着翻,实现 →→→ U→ ←←← U→ →→→ pattern
    from collections import defaultdict
    rows = defaultdict(list)
    for s in swaths_pix:
        rows[s[0]].append(s)
    sorted_avs = sorted(rows.keys())

    def _in_clip(wx, wy):
        if clip_polygon is None: return True
        # ray-cast point-in-polygon
        inside = False; j = len(clip_polygon) - 1
        for i in range(len(clip_polygon)):
            xi, yi = clip_polygon[i]; xj, yj = clip_polygon[j]
            if ((yi > wy) != (yj > wy)) and (wx < (xj - xi) * (wy - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    def _clip_segment_to_poly(x0, x1, y, poly):
        """把水平段 [x0..x1] @ y 与 polygon 求交,返回 polygon 内的子段列表。
        采用沿 x 方向采样 + 跨界检测,简单但准确。"""
        if poly is None:
            return [(x0, x1)]
        # 沿 x 方向以原始 grid resolution 采样,标记每点是否在 polygon 内
        step_x = max(resolution * 0.5, (x1 - x0) / 200.0)
        n = max(2, int(abs(x1 - x0) / step_x) + 1)
        out_segs = []
        cur_start = None
        for k in range(n):
            t = k / (n - 1) if n > 1 else 0.0
            x = x0 + t * (x1 - x0)
            inside = _in_clip(x, y)
            if inside and cur_start is None:
                cur_start = x
            elif (not inside) and cur_start is not None:
                if abs(x - cur_start) >= resolution:  # 太短的丢
                    out_segs.append((cur_start, x))
                cur_start = None
        if cur_start is not None and abs(x1 - cur_start) >= resolution:
            out_segs.append((cur_start, x1))
        return out_segs

    def _clip_segment_to_poly_v(y0, y1, x, poly):
        if poly is None:
            return [(y0, y1)]
        step_y = max(resolution * 0.5, (y1 - y0) / 200.0)
        n = max(2, int(abs(y1 - y0) / step_y) + 1)
        out_segs = []; cur_start = None
        for k in range(n):
            t = k / (n - 1) if n > 1 else 0.0
            y = y0 + t * (y1 - y0)
            inside = _in_clip(x, y)
            if inside and cur_start is None:
                cur_start = y
            elif (not inside) and cur_start is not None:
                if abs(y - cur_start) >= resolution:
                    out_segs.append((cur_start, y))
                cur_start = None
        if cur_start is not None and abs(y1 - cur_start) >= resolution:
            out_segs.append((cur_start, y1))
        return out_segs

    out: List[Tuple[Point, Point]] = []
    flip = False        # 整行翻转标志
    for av in sorted_avs:
        row_swaths = rows[av]
        # row 内 seg 按 lo 排;flip 决定遍历方向
        row_swaths.sort(key=lambda s: s[1])
        if flip:
            row_swaths = list(reversed(row_swaths))
        if axis == "x":
            y_w = origin_y + (av + 0.5) * resolution
            for (av_, lo, hi) in row_swaths:
                xs_w = origin_x + (lo + 0.5) * resolution
                xe_w = origin_x + (hi + 0.5) * resolution
                subs = _clip_segment_to_poly(xs_w, xe_w, y_w, clip_polygon)
                if flip:
                    # 反向行 — sub-seg 也反向遍历且每条端点翻
                    subs = list(reversed(subs))
                    for (a, b) in subs:
                        out.append(((b, y_w), (a, y_w)))
                else:
                    for (a, b) in subs:
                        out.append(((a, y_w), (b, y_w)))
        else:
            x_w = origin_x + (av + 0.5) * resolution
            for (av_, lo, hi) in row_swaths:
                ys_w = origin_y + (lo + 0.5) * resolution
                ye_w = origin_y + (hi + 0.5) * resolution
                subs = _clip_segment_to_poly_v(ys_w, ye_w, x_w, clip_polygon)
                if flip:
                    subs = list(reversed(subs))
                    for (a, b) in subs:
                        out.append(((x_w, b), (x_w, a)))
                else:
                    for (a, b) in subs:
                        out.append(((x_w, a), (x_w, b)))
        flip = not flip
    return out


def filter_clusters_by_size(mask: np.ndarray, min_cells: int) -> np.ndarray:
    """删除所有 < min_cells 的连通簇。比 open_binary 温和:
    不会因为薄墙(1 px)被腐蚀掉,只要簇足够大就保留。
    """
    if not mask.any() or min_cells <= 1:
        return mask
    out = np.zeros_like(mask)
    remaining = mask.copy()
    while remaining.any():
        ys, xs = np.where(remaining)
        sy, sx = int(ys[0]), int(xs[0])
        cc = _flood_from(remaining, sy, sx)
        if int(cc.sum()) >= min_cells:
            out |= cc
        remaining &= ~cc
    return out


# ---------- 步骤 3:连通域 ----------

def _flood_from(mask: np.ndarray, sy: int, sx: int) -> np.ndarray:
    """4-连通 flood-fill,返回 (sy,sx) 所在连通簇的 bool mask。"""
    cur = np.zeros_like(mask)
    cur[sy, sx] = True
    while True:
        nxt = cur.copy()
        nxt[1:, :] |= cur[:-1, :]
        nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        nxt &= mask
        if np.array_equal(nxt, cur):
            return cur
        cur = nxt


def largest_component(mask: np.ndarray,
                       seed: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """返回 4-连通最大分量。

    - seed 给了且落在 mask 内 → flood 包含 seed 的那个分量
    - 否则枚举所有分量,返回最大那个(真"最大",不是任意一个)
    """
    if not mask.any():
        return np.zeros_like(mask)

    if seed is not None:
        sy, sx = seed
        if 0 <= sy < mask.shape[0] and 0 <= sx < mask.shape[1] and mask[sy, sx]:
            return _flood_from(mask, sy, sx)

    # 枚举所有分量,留最大者。形态学开运算后,噪点已死,分量数通常 ≤ 几十,够快。
    remaining = mask.copy()
    best = np.zeros_like(mask)
    best_size = 0
    while remaining.any():
        ys, xs = np.where(remaining)
        sy, sx = int(ys[0]), int(xs[0])
        cc = _flood_from(remaining, sy, sx)
        sz = int(cc.sum())
        if sz > best_size:
            best_size = sz
            best = cc
        remaining &= ~cc
    return best


# ---------- 步骤 4:Moore-neighbor 边界跟踪 ----------

# 8-邻居顺时针:E, SE, S, SW, W, NW, N, NE
_DY = [0, 1, 1, 1, 0, -1, -1, -1]
_DX = [1, 1, 0, -1, -1, -1, 0, 1]


def _find_start(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """从左上扫描第一个 True 像素 — 该像素一定在外轮廓上。"""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(ys[0]), int(xs[0])


def trace_outer_contour(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Moore-neighbor (Jacob's stop) 跟踪外轮廓,返回像素 (y, x) 列表。

    约定 direction = 上一步的移动方向(0=E,顺时针 0..7)。
    新起点的搜索方向 = (backtrack + 1) % 8 = (direction + 5) % 8,顺时针。
    """
    start = _find_start(mask)
    if start is None:
        return []
    h, w = mask.shape
    contour = [start]
    cur_y, cur_x = start
    # 从左到右扫描发现 start,上一格在西 → 我们是"向东"移动到 start。direction=0(E)。
    direction = 0
    max_steps = int(mask.sum() * 8) + 32
    for _ in range(max_steps):
        start_dir = (direction + 5) % 8   # backtrack 之后一格,CW 扫
        found = False
        for k in range(8):
            d = (start_dir + k) % 8
            ny, nx = cur_y + _DY[d], cur_x + _DX[d]
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                cur_y, cur_x = ny, nx
                direction = d
                contour.append((cur_y, cur_x))
                found = True
                break
        if not found:
            break
        if (cur_y, cur_x) == start and len(contour) > 3:
            break
    return contour


# ---------- 步骤 5:Douglas-Peucker 简化 ----------

def _dp(points: List[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return points
    # 找距 P0-Pn 直线最远的点
    p0 = np.array(points[0]); pn = np.array(points[-1])
    arr = np.array(points)
    if np.allclose(p0, pn):
        d = np.linalg.norm(arr - p0, axis=1)
    else:
        v = pn - p0
        L = np.linalg.norm(v)
        d = np.abs((arr[:, 0] - p0[0]) * v[1] - (arr[:, 1] - p0[1]) * v[0]) / L
    idx = int(np.argmax(d))
    if d[idx] > eps:
        left = _dp(points[:idx + 1], eps)
        right = _dp(points[idx:], eps)
        return left[:-1] + right
    return [points[0], points[-1]]


def simplify_polygon(points: List[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return points
    closed = points[0] == points[-1]
    pts = points if closed else points + [points[0]]
    simp = _dp(pts, eps)
    return simp if closed else simp[:-1]


# ---------- 步骤 6:像素 → 世界 ----------

def pixel_to_world(yx_list: List[Tuple[int, int]],
                    origin_x: float, origin_y: float, resolution: float,
                    height_cells: int) -> List[Point]:
    """OccupancyGrid 像素 (y, x) → 世界 (x, y) 米。

    OccupancyGrid 约定:data[0] 是地图原点(origin_x, origin_y)那个 cell,
    row 沿 +y 增长,col 沿 +x 增长。所以 world_x = origin_x + (x + 0.5) * res,
    world_y = origin_y + (y + 0.5) * res。
    """
    out = []
    for y, x in yx_list:
        wx = origin_x + (x + 0.5) * resolution
        wy = origin_y + (y + 0.5) * resolution
        out.append((wx, wy))
    return out


# ---------- 主入口 ----------

def polygon_to_mask(polygon, width: int, height: int, resolution: float,
                     origin_x: float, origin_y: float) -> np.ndarray:
    """polygon → 像素 mask(True=在内)。numpy 向量化 ray-cast,几 ms。"""
    if not polygon or len(polygon) < 3:
        return np.ones((height, width), dtype=bool)
    poly = np.array(polygon, dtype=np.float64)
    xs = origin_x + (np.arange(width) + 0.5) * resolution
    ys = origin_y + (np.arange(height) + 0.5) * resolution
    X, Y = np.meshgrid(xs, ys)
    mask = np.zeros((height, width), dtype=bool)
    n = len(poly); j = n - 1
    for i in range(n):
        yi = poly[i, 1]; yj = poly[j, 1]
        xi = poly[i, 0]; xj = poly[j, 0]
        cond1 = (yi > Y) != (yj > Y)
        cond2 = X < (xj - xi) * (Y - yi) / (yj - yi + 1e-12) + xi
        mask ^= (cond1 & cond2)
        j = i
    return mask


def near_walls_clip_polygon(data: bytes, width: int, height: int,
                              resolution: float, origin_x: float, origin_y: float,
                              within_m: float = 3.0,
                              min_cluster_cells: int = 30,
                              ) -> List[Tuple[float, float]]:
    """生成"距墙 ≤ within_m"的紧贴区域多边形(用于 warehouse 等开放型地图)。

    思路:
      - 把 obstacle 像素膨胀 within_m → "靠墙带"
      - 跟 free mask 求交 → 仓库内部 + 走廊(不含建筑外大片空地)
      - 取最大连通分量 → 外轮廓(凸包近似)
    """
    import math as _math
    grid = np.frombuffer(data, dtype=np.int8).reshape(height, width)
    obstacle = grid > 50
    obstacle = filter_clusters_by_size(open_binary(obstacle, 1), min_cluster_cells)
    if not obstacle.any():
        return []
    pad = max(1, _math.ceil(within_m / resolution))
    near = dilate_binary(obstacle, pad)
    free = ~obstacle & (grid >= 0)
    region = near & free
    if not region.any():
        return []
    cc = largest_component(region)
    ys, xs = np.where(cc)
    if len(ys) == 0:
        return []
    xy = [(origin_x + (int(x) + 0.5) * resolution,
            origin_y + (int(y) + 0.5) * resolution)
           for y, x in zip(ys, xs)]
    hull = _convex_hull_xy(xy)
    return hull


def _convex_hull_xy(points_xy: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Andrew's monotone chain,返回 CCW 凸包(首点 != 尾点)。纯 Python,O(N log N)。"""
    pts = sorted(set((float(x), float(y)) for x, y in points_xy))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _pad_polygon_outward(poly: List[Tuple[float, float]], pad_m: float
                          ) -> List[Tuple[float, float]]:
    """凸包每个顶点沿质心→顶点方向外推 pad_m。简单 + 对凸包正确。"""
    if len(poly) < 3 or pad_m <= 0:
        return poly
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    out = []
    for x, y in poly:
        dx, dy = x - cx, y - cy
        L = (dx * dx + dy * dy) ** 0.5
        if L < 1e-6:
            out.append((x, y))
        else:
            s = (L + pad_m) / L
            out.append((cx + dx * s, cy + dy * s))
    return out


def _wall_bbox_polygon(obstacle_mask: np.ndarray, resolution: float,
                        origin_x: float, origin_y: float,
                        pad_m: float = 1.0,
                        use_largest_cluster_only: bool = False) -> List[Point]:
    """对所有(去噪后)障碍像素做凸包,得"仓库轮廓" — 比 axis-aligned bbox
    紧得多,能自动处理倾斜布局(走廊/货架斜着排)。

    use_largest_cluster_only=False(默认):用所有障碍像素 → 凸包包住整个 facility
    True:只用最大障碍簇 → 兜底单一房间/单一结构
    """
    if not obstacle_mask.any():
        return []
    if use_largest_cluster_only:
        cluster = largest_component(obstacle_mask)
    else:
        cluster = obstacle_mask
    ys, xs = np.where(cluster)
    if len(ys) == 0:
        return []
    # 像素 → 世界坐标
    h = obstacle_mask.shape[0]
    xy_world = [
        (origin_x + (int(x) + 0.5) * resolution,
         origin_y + (int(y) + 0.5) * resolution)
        for y, x in zip(ys, xs)
    ]
    hull = _convex_hull_xy(xy_world)
    if len(hull) < 3:
        return []
    return _pad_polygon_outward(hull, pad_m)


def recover_saved_map_known_free_mask(
    yaml_path: Optional[str],
    width: int,
    height: int,
) -> Tuple[Optional[np.ndarray], int]:
    """Recover map-saver's gray unknown cells before `/map` loses them.

    Some legacy TD25A yaml files use ``free_thresh: 0.25``.  A standard
    map-saver gray pixel (205, occupancy ~= 0.196) is consequently loaded by
    map_server as FREE, making the unobserved exterior indistinguishable from
    measured white floor in OccupancyGrid.  When the active saved image has the
    unmistakable 0/205/254 palette, return only its white cells as known-free.

    This is intentionally a narrow compatibility guard: continuous-tone or
    differently-sized images return ``None`` and the normal OccupancyGrid path
    remains authoritative.
    """
    if not yaml_path or not os.path.isfile(yaml_path):
        return None, 0
    try:
        import yaml
        from PIL import Image
        with open(yaml_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        image_path = str(doc.get("image") or "")
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(yaml_path), image_path)
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    except Exception:
        return None, 0
    if image.shape != (height, width):
        return None, 0

    gray_unknown = (image >= 203) & (image <= 207)
    measured_free = image >= 250
    unknown_count = int(gray_unknown.sum())
    # Require a meaningful gray population and measured white floor.  A few
    # anti-aliased 205 pixels in an arbitrary image must not change semantics.
    if unknown_count < max(20, int(image.size * 0.01)) or not measured_free.any():
        return None, 0
    # Saved images are top-left origin; OccupancyGrid rows grow upward from the
    # yaml origin, hence the vertical flip.
    return np.flipud(measured_free).copy(), unknown_count


def _nearest_true_cell(mask: np.ndarray, seed: Tuple[int, int],
                       max_radius: int) -> Optional[Tuple[int, int]]:
    """Euclidean nearest True cell inside a small bounded window."""
    sy, sx = seed
    h, w = mask.shape
    y0 = max(0, sy - max_radius)
    y1 = min(h, sy + max_radius + 1)
    x0 = max(0, sx - max_radius)
    x1 = min(w, sx + max_radius + 1)
    yy, xx = np.nonzero(mask[y0:y1, x0:x1])
    if yy.size == 0:
        return None
    yy = yy + y0
    xx = xx + x0
    d2 = (yy - sy) ** 2 + (xx - sx) ** 2
    inside = d2 <= max_radius * max_radius
    if not inside.any():
        return None
    candidates = np.flatnonzero(inside)
    k = int(candidates[int(np.argmin(d2[candidates]))])
    return int(yy[k]), int(xx[k])


def _snap_to_safe_through_component(
    safe_mask: np.ndarray,
    allowed_component: np.ndarray,
    seed: Tuple[int, int],
    max_radius: int,
) -> Tuple[Optional[Tuple[int, int]], int]:
    """4-connected bounded snap that cannot cross a wall into another room."""
    from collections import deque
    sy, sx = seed
    if safe_mask[sy, sx]:
        return seed, 0
    q = deque([(sy, sx, 0)])
    seen = {seed}
    h, w = safe_mask.shape
    while q:
        y, x, d = q.popleft()
        if d >= max_radius:
            continue
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            ny, nx = y + dy, x + dx
            cell = (ny, nx)
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if cell in seen or not allowed_component[ny, nx]:
                continue
            if safe_mask[ny, nx]:
                return cell, d + 1
            seen.add(cell)
            q.append((ny, nx, d + 1))
    return None, 0


def _component_reaches_inner_map_edge(component: np.ndarray, margin: int) -> bool:
    """Erosion clears the literal border; inspect its inner safety frame."""
    h, w = component.shape
    m = max(1, min(int(margin), max(1, min(h, w) // 3)))
    return bool(
        component[:m + 1, :].any()
        or component[h - m - 1:, :].any()
        or component[:, :m + 1].any()
        or component[:, w - m - 1:].any()
    )


def extract_clean_field_result(
    data: bytes,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    robot_radius_m: float = 0.40,
    simplify_eps_m: float = 0.10,
    seed_world: Optional[Point] = None,
    open_map_threshold: float = 0.80,
    denoise_open_cells: int = 1,
    min_cluster_cells: int = 30,
    known_free_mask: Optional[np.ndarray] = None,
    max_seed_snap_m: float = 0.80,
    recovered_unknown_cells: int = 0,
) -> CleanFieldResult:
    """Extract the maximum *robot-reachable* safe component around ``seed``.

    Safety invariants:

    - Raw obstacle pixels are never denoised away for reachability.
    - A supplied robot seed never silently falls back to the global largest
      component.  If erosion removes the seed, snapping is bounded and follows
      only its original 4-connected free component.
    - A component leaking to an open map edge is clipped by an obstacle-support
      envelope and flood-filled again.  With no defensible structure it fails
      closed instead of returning the whole map rectangle.
    """
    result = CleanFieldResult(recovered_unknown_cells=int(recovered_unknown_cells))
    if width <= 0 or height <= 0 or resolution <= 0.0:
        result.failure_reason = "invalid_map_geometry"
        return result
    grid = np.frombuffer(data, dtype=np.int8).reshape(height, width)
    obstacle_raw = grid > 50
    free = (grid >= 0) & (grid <= 50)
    if known_free_mask is not None:
        if known_free_mask.shape != free.shape:
            result.failure_reason = "known_free_shape_mismatch"
            return result
        free &= known_free_mask.astype(bool, copy=False)
    if not free.any():
        result.failure_reason = "no_known_free_space"
        return result

    seed_cell = None
    if seed_world is not None:
        sx = int(np.floor((seed_world[0] - origin_x) / resolution))
        sy = int(np.floor((seed_world[1] - origin_y) / resolution))
        if not (0 <= sy < height and 0 <= sx < width):
            result.failure_reason = "seed_outside_map"
            return result
        seed_cell = (sy, sx)
        result.seed_cell = seed_cell

    # Occupied pixels represent square cells, not zero-width points.  After k
    # erosions the nearest surviving centre is (k+1) cells away, i.e. its
    # clearance to the occupied cell boundary is (k+0.5)*resolution.  This
    # conversion keeps a real 0.68m doorway passable for the 0.34m half-width
    # footprint on an 0.08m map (k=4), while remaining geometrically safe.
    erode_n = max(1, int(np.ceil(robot_radius_m / resolution - 0.5)))
    safe = erode_binary(free, erode_n)
    result.safe_cells = int(safe.sum())
    if not safe.any():
        result.failure_reason = "no_clearance_safe_space"
        return result

    if seed_cell is None:
        component = largest_component(safe)
        yy, xx = np.nonzero(component)
        if yy.size == 0:
            result.failure_reason = "no_safe_component"
            return result
        snapped = (int(yy[0]), int(xx[0]))
        snap_steps = 0
        raw_component = largest_component(free, seed=snapped)
    else:
        # If localization lands on a noisy occupied pixel, only a very local
        # raw-free snap is allowed.  It must never jump across the room.
        raw_seed = seed_cell
        if not free[raw_seed]:
            raw_seed = _nearest_true_cell(
                free, raw_seed,
                max_radius=max(1, int(np.ceil(0.25 / resolution))))
            if raw_seed is None:
                result.failure_reason = "seed_not_in_known_free_space"
                return result
        raw_component = _flood_from(free, raw_seed[0], raw_seed[1])
        safe_in_room = safe & raw_component
        snapped, snap_steps = _snap_to_safe_through_component(
            safe_in_room, raw_component, raw_seed,
            max_radius=max(1, int(np.ceil(max_seed_snap_m / resolution))))
        if snapped is None:
            result.failure_reason = "seed_component_has_no_safe_clearance"
            return result
        component = _flood_from(safe_in_room, snapped[0], snapped[1])

    result.snapped_seed_cell = snapped
    result.seed_snap_m = float(snap_steps) * resolution

    component_size = int(component.sum())
    map_ratio = component_size / max(1, width * height)
    leaks_open_edge = _component_reaches_inner_map_edge(component, erode_n + 1)
    if leaks_open_edge or map_ratio >= float(open_map_threshold):
        # The envelope is only an open-map upper bound.  Raw occupied pixels
        # remain collision barriers, and we re-flood from the robot afterwards.
        envelope_obstacles = filter_clusters_by_size(
            obstacle_raw, max(1, int(min_cluster_cells)))
        envelope_poly = _wall_bbox_polygon(
            envelope_obstacles, resolution, origin_x, origin_y,
            pad_m=max(1.0, robot_radius_m * 2.0),
            use_largest_cluster_only=False)
        if len(envelope_poly) < 3:
            result.failure_reason = "open_map_without_structure_envelope"
            return result
        envelope = polygon_to_mask(
            envelope_poly, width, height, resolution, origin_x, origin_y)
        guarded_safe = safe & raw_component & envelope
        guarded_allowed = raw_component & envelope
        guarded_seed, guarded_steps = _snap_to_safe_through_component(
            guarded_safe, guarded_allowed, snapped,
            max_radius=max(1, int(np.ceil(max_seed_snap_m / resolution))))
        if guarded_seed is None:
            result.failure_reason = "seed_outside_open_map_envelope"
            return result
        component = _flood_from(guarded_safe, guarded_seed[0], guarded_seed[1])
        if _component_reaches_inner_map_edge(component, erode_n + 1):
            result.failure_reason = "open_map_envelope_still_unbounded"
            return result
        result.used_open_guard = True
        result.snapped_seed_cell = guarded_seed
        result.seed_snap_m += float(guarded_steps) * resolution

    result.reachable_cells = int(component.sum())
    result.reachable_area_m2 = result.reachable_cells * resolution * resolution
    if result.reachable_cells < 3:
        result.failure_reason = "reachable_component_too_small"
        return result

    contour_pix = trace_outer_contour(component)
    if len(contour_pix) < 3:
        result.failure_reason = "contour_trace_failed"
        return result
    contour_world = pixel_to_world(
        contour_pix, origin_x, origin_y, resolution, height)
    polygon = simplify_polygon(contour_world, simplify_eps_m)
    if len(polygon) > 1 and polygon[0] == polygon[-1]:
        polygon = polygon[:-1]
    if len(polygon) < 3:
        result.failure_reason = "polygon_simplification_failed"
        return result
    result.polygon = polygon
    return result


def extract_clean_field(data: bytes, width: int, height: int,
                         resolution: float, origin_x: float, origin_y: float,
                         robot_radius_m: float = 0.40,
                         simplify_eps_m: float = 0.10,
                         seed_world: Optional[Point] = None,
                         open_map_threshold: float = 0.80,
                         denoise_open_cells: int = 1,
                         min_cluster_cells: int = 30,
                         known_free_mask: Optional[np.ndarray] = None,
                         max_seed_snap_m: float = 0.80,
                         recovered_unknown_cells: int = 0,
                         ) -> List[Point]:
    """Compatibility wrapper returning only the reachable outer polygon."""
    return extract_clean_field_result(
        data=data, width=width, height=height, resolution=resolution,
        origin_x=origin_x, origin_y=origin_y,
        robot_radius_m=robot_radius_m, simplify_eps_m=simplify_eps_m,
        seed_world=seed_world, open_map_threshold=open_map_threshold,
        denoise_open_cells=denoise_open_cells,
        min_cluster_cells=min_cluster_cells,
        known_free_mask=known_free_mask,
        max_seed_snap_m=max_seed_snap_m,
        recovered_unknown_cells=recovered_unknown_cells,
    ).polygon
