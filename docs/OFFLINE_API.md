# 离线接口与坐标约定

## 输入地图

核心函数使用 OccupancyGrid 风格的行优先 `int8` 数组：

- `<0`：未知
- `0..50`：可通行候选
- `>50`：障碍
- 第 0 行对应地图原点一侧，世界坐标由 `origin_x/origin_y/resolution` 换算

保存的 PNG 使用左上角图像原点，因此加载时需要上下翻转。`render_partitioned_planner.load_map()` 已实现该约定，并能修复旧地图中灰色 205 未知格被误读为 free 的问题。

## 主入口

```python
from td25a_robot_ui.algorithms.grid_coverage import (
    CoverageFootprint,
    coverage_swath_spacing,
    plan_partitioned_coverage,
)
```

返回 `PartitionedCoveragePlan`。重要字段：

- `path`：完整底盘中心路径。
- `segments`：`fill/perimeter/transfer` 分段。
- `regions`：分区掩码、包围盒、中心和方向。
- `visit_order`：区域访问顺序。
- `swaths`：清扫直线段。
- `hard_stop_indices`：不能被执行层跨越的扫带/转向边界。
- `arrival_yaws/departure_yaws`：同一停止点的进入和离开朝向。
- `cleaner_profile`：与 `path` 等长的清扫头命令。
- `cleaner_center_path`：偏移后的清扫工具中心路径。
- `footprint_valid/coverage_complete/path_continuous`：硬验收结果。
- `serviceable_coverage_ratio/actual_brush_coverage_ratio`：覆盖指标。
- `failure_reason`：历史上混合承载失败和非致命诊断，不能只按是否为空判断；同时检查结构化布尔量、碰撞数和覆盖率。

## 当前生产等价 profile

见 `docs/CURRENT_ARCHITECTURE.md`。本包的真实地图 benchmark、renderer 和 contract exporter 已统一使用该 profile。

## 清扫头符号

- `offset_m > 0`：车体左侧。
- `offset_m < 0`：车体右侧。
- 车体 `+x` 为前方，`+y` 为左侧。
- yaw 单位为弧度。

## 计划导出

`tools/export_plan_contract.py` 会输出可供其他语言、优化器或训练管线读取的 JSON。默认包含完整路径；使用 `--summary-only` 只输出指标和结构摘要。

合同版本为 `td25a.coverage_plan.v1`。`trajectory` 中每个记录同时携带底盘点、进入/离开航向、硬停止、工具中心和清扫头命令；导出器会先做等长校验，字段错位时拒绝输出。
