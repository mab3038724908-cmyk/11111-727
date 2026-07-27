# TD25A 离线分区覆盖规划算法交付包

这是从 TD25A 车端工程中抽出的纯离线算法包，用于在开发者自己的电脑上修改、训练、基准测试和渲染预览。

本包不包含 SSH 配置、IP、密钥、ROS 启动脚本、底盘控制、导航 action、串口协议或清扫执行接口；运行这里的程序不会连接机器人，也不会让机器人运动。

## 当前基线

- 来源提交：`2c7443e5b4337a6aae0787cddb220e5fbdcdf4a1`
- 核心文件 SHA-256：`0064b602424cf38602440b04143b3a0d70d4e037b2026918e3d565e743c03918`
- 主入口：`plan_partitioned_coverage()`
- 当前策略：机器人可达区 → 门/走廊与 BCD 分区 → 低价值碎片剔除 → 从机器人位置排序区域 → 每区完整平行填充 → 每区沿边 → 区域连接
- 顺序约束：一个区域的填充和沿边完成后，才访问下一个区域
- 车体模型：前 `0.33m`、后 `0.72m`、半宽 `0.34m`
- 当前 map9 基线：8 个区域、23 个 BCD 单元、44 条扫带、466.32m；完整车体碰撞为 0，可服务地面覆盖 98.43%，实际刷洗覆盖 96.99%

详细任务目标见 [DELIVERY_BRIEF.md](DELIVERY_BRIEF.md)，可移动前清扫装置改造见 [docs/MOVABLE_FRONT_CLEANER.md](docs/MOVABLE_FRONT_CLEANER.md)，训练边界与数据要求见 [docs/TRAINING_AND_OPTIMIZATION.md](docs/TRAINING_AND_OPTIMIZATION.md)。

## 快速开始

推荐 Python 3.10 或 3.11：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
./run_offline_checks.sh
```

生成当前 map9 预览和 JSON 基准：

```bash
./render_examples.sh
```

单独运行真实地图基准：

```bash
PYTHONPATH=src/td25a_robot_ui \
python tools/benchmark_partitioned_coverage.py \
  --maps-dir maps --map 9 --json reference_outputs/map9_metrics.json
```

导出完整的底盘路径、区域、段、航向以及清扫头命令：

```bash
PYTHONPATH=src/td25a_robot_ui \
python tools/export_plan_contract.py \
  maps/9.yaml reference_outputs/map9_plan_contract.json
```

## 目录

```text
src/td25a_robot_ui/td25a_robot_ui/algorithms/  覆盖、分区、连接和PathManager核心
src/td25a_robot_ui/td25a_robot_ui/store/       真实前部刷盘覆盖记账模型
src/td25a_robot_ui/test/                        从车端抽出的纯离线回归测试
tests/                                          本交付包新增合同测试
tools/                                          基准、渲染和计划导出工具
maps/9.yaml + 9.png                             当前真实样例地图
docs/                                           接口、目标、安全和验收说明
examples/cleaner_hardware_template.yaml          可移动前清扫装置待测参数模板
reference_outputs/                              基线图片与JSON结果
```

## 关于“训练”

当前核心是确定性几何规划器，不是神经网络模型。建议先在现有硬安全约束下优化分区、区域排序、连接和清扫头状态；如果引入强化学习、模仿学习或搜索参数训练，模型只能提出候选，最终结果仍必须经过现有碰撞、连续性、覆盖率和执行器约束校验。

## 明确边界

- 可以修改：离线算法、评分函数、合成数据、区域图排序、清扫头轨迹、基准和可视化。
- 不可以假设：开发者能 SSH 机器人、能直接发布速度、能直接调用清扫执行器。
- 返回成果：压缩包或 Git patch、全部测试结果、基准前后对比、预览图和改造说明。
- 上车集成和实车安全测试由机器人所有者完成。

## 地图隐私

`maps/9.*` 是真实环境布局，用于复现实车基线。转发前请确认接收方有权查看；如果不能共享，删除这两个文件，合成测试仍可独立运行。详情见 [docs/SECURITY_AND_PRIVACY.md](docs/SECURITY_AND_PRIVACY.md)。
