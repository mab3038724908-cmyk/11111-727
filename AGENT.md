# TD25A 离线覆盖规划项目指南

## 1. 项目定位与边界

这是从 TD25A 项目抽取的**纯离线、确定性几何覆盖规划器**。它读取
OccupancyGrid 风格地图或保存的 PNG/YAML 地图，生成底盘中心路径、分区信息、
执行停点和逐点清扫语义。

- 不包含 ROS 节点、SSH、Jetson 连接、底盘控制、速度指令、串口/CAN 或执行器驱动。
- 离线规划通过不等于真实车辆、清扫机构或现场安全已经验证。
- `maps/9.*` 是真实环境样例地图，分享前应确认授权；不能分享时移除它和对应
  `reference_outputs/map9*`，用合成场景继续开发。

## 2. 目录与核心入口

| 位置 | 职责 |
| --- | --- |
| `src/td25a_robot_ui/td25a_robot_ui/algorithms/grid_coverage.py` | 核心规划器；入口是 `plan_partitioned_coverage()`。 |
| `algorithms/free_space.py` | 已知自由空间恢复、形态学处理、可清扫区域提取。 |
| `algorithms/path_manager.py`、`path_chunks.py`、`path_progress.py` | 路径窗口、进度、分段与安全重接的纯算法实现。 |
| `algorithms/reclean.py` | 根据实际清扫栅格找漏扫区并组织补扫。 |
| `algorithms/cleaning_mode.py` | 生产清洁模式、右沿墙伸出公式、超声波通道选择和执行意图。 |
| `store/cleaned_area.py` | 清扫刷实际覆盖的栅格记账模型。 |
| `tools/` | 基准、渲染、JSON 合约导出工具。 |
| `src/td25a_robot_ui/test/`、`tests/` | 回归测试和交付/生产合约测试。 |

主要返回值是 `PartitionedCoveragePlan`，重点字段：

- `path`：完整底盘中心路径；
- `segments`：`fill`、`perimeter`、`transfer` 分段；
- `regions`、`visit_order`、`swaths`：分区与扫带结构；
- `hard_stop_indices`、`arrival_yaws`、`departure_yaws`：执行边界；
- `cleaner_profile` 与 `cleaner_center_path`：逐点清扫语义和工具中心轨迹；
- `footprint_valid`、`path_continuous`、`coverage_complete`、覆盖率字段：验收结果。

## 3. 规划链路与不变量

规划主链路：

```text
地图 -> known-free/可达域 -> 车体通行与转向掩码
     -> 门口/窄颈检测与 BCD 分区 -> Boustrophedon 平行扫带
     -> 直连、Theta*、A* 安全连接 -> 实际边界沿墙
     -> 分区计划 + 路径航向 + 清扫 profile
```

必须保持以下不变量：

1. 未知格、障碍格和用户禁区不可进入。
2. 使用完整非对称底盘，而不是圆形近似。默认车体参数：前伸 `0.33 m`、后伸
   `0.72 m`、半宽 `0.34 m`、转向余量 `0.07 m`。
3. 每个区域必须完整按 `fill -> perimeter -> transfer` 排列；不能在一个区域中途跳到
   另一个区域。
4. 连接器优先直连，受阻时使用 Theta*，再回退 A*；直线检查采用 supercover，不能从
   阻塞角点挤过。
5. `cleaner_profile` 必须与 `path` 等长；路径、到达/离开航向、硬停点和清扫命令的索引
   必须保持对齐。
6. `failure_reason` 可能包含非致命诊断，不能仅以其是否为空判断成败；应以结构化安全、
   连续性、覆盖和碰撞字段为准。

## 4. 当前生产清洁机构合约

以根目录 `production_contract.yaml` 为**当前权威来源**。它优先于较早的
`BASELINE.json`、README 和部分历史文档中的对称 `0.70 m` 刷盘假设。

- 坐标系：`base_link`，`+x` 向前，`+y` 向左。
- X 零位刷盘：`x=0.20450..0.56726 m`，左边缘 `+0.42700 m`，右边缘
  `-0.55480 m`，总宽 `0.98180 m`。
- X 台只允许向车体右侧伸出：可执行 `offset_m` 为 `[-0.250, 0.0] m`；正值代表左侧，
  不可输出为真实执行命令。
- 填充清扫：X 不摆动，`EDGE_CENTER`、`offset_m=0`；生产扫带间距为
  `0.98180 * 0.75 = 0.73635 m`。
- 右侧沿墙：`offset_m = -clamp(D - 0.5548 - 0.03, 0, 0.25)`；目标间隙为 `0.03 m`。
- 转场不清扫；伸出状态仅允许前进、角速度受限；倒车、原地转向前必须收回。
- `cleaning_mode.py` 是纯合约模块，不直接控制硬件；真实驱动、反馈和急停仍属于项目外部。

## 5. 验收与优化顺序

先过硬门槛，再比较软指标。以下任一失败，候选方案不可接受：

1. `footprint_violation_count == 0` 且 `footprint_valid == true`；
2. `path_continuous == true`、`coverage_complete == true`；
3. 可服务区域覆盖率不低于 `0.95`；
4. 路径不进入未知、障碍或禁区；
5. 完整刷盘也无碰撞，且 `cleaner_profile` 数值有限、长度对齐、偏移在行程范围内；
6. 所有相关回归测试通过。

硬门槛通过后，按以下顺序做字典序优化：可避免交叉、重复覆盖、路径长度、转向/停点、
规划耗时和工具动作次数。不得为了更小的重叠或更短路径而牺牲碰撞、连续性或覆盖门槛。

## 6. 开发与验证

推荐 Python 3.10 或 3.11。在项目根目录：

```bash
python -m pip install -r requirements.txt
./run_offline_checks.sh
```

`run_offline_checks.sh` 会编译核心模块、运行指定回归测试，并生成合成场景和 map9 的
基准 JSON。常用单独命令：

```bash
PYTHONPATH=src/td25a_robot_ui python tools/benchmark_partitioned_coverage.py \
  --maps-dir maps --map 9 --json reference_outputs/map9_metrics.json

PYTHONPATH=src/td25a_robot_ui python tools/export_plan_contract.py \
  maps/9.yaml reference_outputs/map9_plan_contract.json
```

每次修改规划或清扫语义后，至少运行受影响测试；修改 `grid_coverage.py` 时还应运行
`test_partitioned_coverage.py`、`test_theta_coverage_connectors.py`、
`tests/test_delivery_contract.py` 和 `tests/test_reference_cleaner_contract.py`。

## 7. 修改准则

- 只改与任务直接相关的代码；不要趁机重构无关模块。
- 每完成一个可独立验证的小功能点，先运行相应验证，再单独创建一次 Git commit。
  Commit message 必须使用概括性的中文，说明修改的实际作用，而非泛泛地写“更新代码”或
  “修复问题”。例如：`修复选区路径从机器人真实位置引入`。
- 改变安全掩码、连接器、扫带排序或清扫 offset 时，必须增加能复现该行为的测试。
- 学习/搜索模型只能提出候选；全部候选必须回到现有确定性验证器，不得绕过安全门槛。
- 真实机械参数未知时应 fail-closed，填写 `examples/cleaner_hardware_template.yaml`，不要猜测。
- 交付应包含：代码、测试结果、map9 与合成场景的前后 JSON、预览图及失败/回退说明。
- `MANIFEST.sha256` 是文件完整性清单。新增或修改受清单覆盖的文件后，必须同步更新其哈希。
