"""从 turn_on_td25a_robot/config/nav2_params.yaml 读取代价地图参数。

提取清扫区域、显示机器人足迹、计算膨胀缓冲等都要跟 Nav2 保持一致 — 改一处 yaml
就同步,避免 UI 和 planner 用不同的 robot_radius 导致路径与渲染不匹配。
"""
from __future__ import annotations

import ast
import math
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class CostmapParams:
    robot_radius: float = 0.40             # 机器人物理半径 (m)
    footprint_inscribed_radius: float = 0.40  # 原点到 footprint 最近边,用于通行连通性
    inflation_radius: float = 0.80         # 障碍膨胀半径 (m)
    cost_scaling_factor: float = 2.0       # 膨胀代价衰减系数
    source: str = "default"                # 实际加载源,便于调试


def _nav2_yaml_paths() -> list[str]:
    """候选位置(按优先级)。"""
    paths = []
    # 1) ament install share
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("turn_on_td25a_robot")
        paths.append(os.path.join(share, "config", "nav2_params.yaml"))
    except Exception:
        pass
    # 2) workspace src(开发环境直接拿源,免每次 colcon build)
    home = os.path.expanduser("~")
    paths.append(os.path.join(home, "td25a_robot", "src", "turn_on_td25a_robot",
                                "config", "nav2_params.yaml"))
    return paths


def _footprint_inscribed_radius(value) -> Optional[float]:
    """Return origin-to-nearest-edge distance for a polygon footprint."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        pts = [(float(p[0]), float(p[1])) for p in value]
    except (TypeError, ValueError, IndexError):
        return None
    best = float("inf")
    for a, b in zip(pts, pts[1:] + pts[:1]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            dist = math.hypot(a[0], a[1])
        else:
            t = max(0.0, min(1.0, -(a[0] * dx + a[1] * dy) / denom))
            dist = math.hypot(a[0] + t * dx, a[1] + t * dy)
        best = min(best, dist)
    return best if math.isfinite(best) and best > 0.0 else None


def load_costmap_params(prefer: str = "local_costmap") -> CostmapParams:
    """读取 yaml。``prefer`` 选 local_costmap / global_costmap;读不到全用默认。

    yaml 嵌套形如 ``{prefer}: {prefer: {ros__parameters: {robot_radius, ...}}}``
    (Nav2 双层 namespace),用 yaml.safe_load 后按这路径下钻。
    """
    try:
        import yaml
    except ImportError:
        return CostmapParams(source="yaml-module-missing")

    for p in _nav2_yaml_paths():
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
        except Exception:
            continue
        node = doc.get(prefer, {}).get(prefer, {}).get("ros__parameters", {})
        if not node:
            continue
        inflation = node.get("inflation_layer", {}) or {}
        robot_radius = float(node.get("robot_radius", 0.40))
        footprint_radius = _footprint_inscribed_radius(node.get("footprint"))
        return CostmapParams(
            robot_radius=robot_radius,
            footprint_inscribed_radius=(footprint_radius
                                         if footprint_radius is not None
                                         else robot_radius),
            inflation_radius=float(inflation.get("inflation_radius", 0.80)),
            cost_scaling_factor=float(inflation.get("cost_scaling_factor", 2.0)),
            source=p,
        )
    return CostmapParams(source="no-yaml-found")
