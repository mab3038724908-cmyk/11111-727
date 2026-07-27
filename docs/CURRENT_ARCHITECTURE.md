# 当前算法架构

## 名义规划链

```text
OccupancyGrid / saved PNG
        ↓
恢复 known-free 语义、当前位姿可达域
        ↓
完整车体通行掩码与转向安全掩码
        ↓
门口/走廊窄颈检测 + BCD split/merge
        ↓
碎片价值筛选与区域邻接图
        ↓
从机器人位置开始的区域访问顺序
        ↓
每区完整 fill → perimeter
        ↓
完整车体安全的区域间 transfer
        ↓
硬停止点、到达/离开航向、清扫头语义
        ↓
PartitionedCoveragePlan
```

主入口在：

```text
src/td25a_robot_ui/td25a_robot_ui/algorithms/grid_coverage.py
```

调用：

```python
plan_partitioned_coverage(..., selection_policy="sparse_graph")
```

## 区域完成语义

`PartitionedCoveragePlan.segments` 中的段类型为：

- `fill`：本区域内部平行扫带和必要的区内连接。
- `perimeter`：本区域真实墙边清扫。
- `transfer`：区域间通行，不属于新区域的填充。

规划器按 `visit_order` 输出区域。一个区域的 `fill` 和 `perimeter` 应连续出现，完成后才出现去往下一区域的 `transfer`。任何改造都必须保持这一原子性。

## 安全模型

`CoverageFootprint` 默认：

- 前伸：`0.33m`
- 后伸：`0.72m`
- 半宽：`0.34m`
- 额外转向余量：`0.07m`

规划器区分：

- 机器人对正后可通过的门口宽度。
- 机器人长后悬进行原地转向或圆角时所需的更大空间。
- 底盘中心路径可通行与前部刷盘实际覆盖是两个不同问题。

## 当前 UI 等价参数

```python
plan_partitioned_coverage(
    ...,
    swath_spacing_m=coverage_swath_spacing(0.70, 0.75),
    clip_polygon=clean_field,
    selection_boundary_polygon=clean_field,
    path_step_m=0.10,
    min_swath_m=1.30,
    min_region_area_m2=5.0,
    min_useful_region_area_m2=15.0,
    min_useful_region_lane_m=3.0,
    max_regions=12,
    clean_width_m=0.70,
    selection_policy="sparse_graph",
)
```

这里 `coverage_swath_spacing(0.70, 0.75)` 得到 `0.525m` 间距。历史函数把第二个参数命名为 `overlap`，但它实际是“间距/清扫宽度系数”；按通常定义对应 25% 几何重叠，改造时不要误读成 75% 重叠。

## 当前不足

- 一些单门房间必然重复经过同一门口，不能简单以“线段重合”为错误。
- 中央走廊可能被多个房间共享，区域图排序仍有优化空间。
- 栅格毛刺会产生低价值小 BCD 单元，目前采用阈值剔除，不是完整语义地图。
- 清扫头横移只生成离线命令，没有真实执行时间和反馈闭环。
- 动态障碍不应重新排列名义区域顺序；它属于车端局部控制和有界恢复层，本包不包含该层。
