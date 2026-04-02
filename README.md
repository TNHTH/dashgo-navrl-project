# DashGo NavRL Project

创建时间: 2026-03-26

这是一个与现有 `dashgo_rl_project` 完全分离的实验仓库，用于在本地 `Isaac Lab 2.0.2 / Isaac Sim 4.5` 环境中实现 `DashGo x NavRL-style` 训练、评测和横向对比。

## 目标

- 保留 DashGo 实车参数、URDF 和差速底盘语义。
- 迁入 NavRL 风格的 `TorchRL + TensorDict + PPO + BetaActor` 训练栈。
- 用统一的 DashGo quick/main 评测协议，对比旧仓库模型与新仓库模型。
- 第一阶段只做训练与仿真评测，不接 ROS2/TorchScript 部署链路。

## 当前状态

- 2026-04-02 已完成第一阶段仿真闭环、正式 quick/main 评测和基线对比。
- 旧仓库基线固定为当前在线 manifest 指向的 GeoNav checkpoint：
  - `/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/models/policy_torchscript.manifest.json`
- 候选模型当前使用：
  - `/home/gwh/dashgo_navrl_project/artifacts/runs/pilot_20260327_134032/checkpoints/checkpoint_2765056.pt`
- 第一阶段结论：
  - `quick` 下候选没有形成成功 episode，但因为旧在线基线严重 timeout/绕圈，候选 `score` 略高于基线。
  - `main` 下候选整体劣于基线，主要问题是碰撞占主导。
  - 两边都未通过行为 gate，因此当前不是“NavRL-style 已经优于主线”，而是“仿真复现已收口，但候选尚未达到可替代基线的水平”。

## 仓库边界

- 上游只读参考: `/home/gwh/NavRL_upstream`
- 旧基线仓库: `/home/gwh/dashgo_rl_project`
- 当前独立实验仓库: `/home/gwh/dashgo_navrl_project`

## 参数来源

- DashGo 轮径/轮距与底盘参数来源:
  - `/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_driver_ros2/config/dashgo_driver.yaml`
- DashGo 环境底座来源:
  - `/home/gwh/dashgo_rl_project/src/dashgo_rl/dashgo_env_v2.py`
  - `/home/gwh/dashgo_rl_project/src/dashgo_rl/dashgo_assets.py`
  - `/home/gwh/dashgo_rl_project/configs/robot/dashgo.urdf`
- NavRL PPO 参考来源:
  - `/home/gwh/NavRL_upstream/isaac-training/training/scripts/ppo.py`
  - `/home/gwh/NavRL_upstream/isaac-training/training/scripts/utils.py`

## 目录

- `src/dashgo_rl/`: 从旧仓库迁入的 DashGo Isaac 环境底座
- `src/navrl_dashgo/`: NavRL 风格训练器、适配器、评测与对比逻辑
- `apps/isaac/train_navrl.py`: 新训练入口
- `tools/compare_models.py`: 统一评测结果对比脚本
- `configs/train/`: Hydra 风格训练配置
- `configs/robot/`: DashGo URDF 与实车参数快照
- `artifacts/`: checkpoints、评测 JSON、训练日志

## 运行环境

- Python 运行时统一使用 `/home/gwh/IsaacLab/_isaac_sim/python.sh`
- 需要安装:
  - `torchrl==0.6.0`
  - `tensordict==0.6.0`
  - `hydra-core`
  - `einops`
  - `wandb`

## 快速开始

```bash
/home/gwh/IsaacLab/_isaac_sim/python.sh -m pip install torchrl==0.6.0 tensordict==0.6.0 hydra-core einops wandb
/home/gwh/IsaacLab/_isaac_sim/python.sh apps/isaac/train_navrl.py --headless profiles=smoke
```

## 后台训练

```bash
python3 tools/background_train.py start --profile smoke
python3 tools/background_train.py status --profile smoke
python3 tools/background_train.py stop --profile smoke
```

- supervisor 状态文件: `artifacts/supervisor/<profile>/status.json`
- supervisor 日志: `artifacts/supervisor/<profile>/train.log`

## 评测与对比

```bash
python3 tools/eval_checkpoint.py --checkpoint artifacts/runs/<run>/checkpoints/checkpoint_final.pt --suite quick
python3 tools/compare_models.py --candidate-checkpoint /path/to/new.pt --suite quick
python3 tools/compare_models.py \
  --suite quick \
  --candidate-json artifacts/eval/pilot_20260327_134032_quick.json \
  --json-out artifacts/eval/compare_quick_online_vs_navrl.json \
  --report-out artifacts/eval/compare_quick_online_vs_navrl.md
```

- `tools/compare_models.py` 默认会从旧仓库在线 manifest 解析 GeoNav 基线 checkpoint。
- 若已经有现成评测 JSON，可直接用 `--baseline-json` 或 `--candidate-json` 跳过重复评测。
- 当前正式产物：
  - `artifacts/eval/baseline_model_883_quick.json`
  - `artifacts/eval/baseline_model_883_main.json`
  - `artifacts/eval/compare_quick_online_vs_navrl.json`
  - `artifacts/eval/compare_quick_online_vs_navrl.md`
  - `artifacts/eval/compare_main_online_vs_navrl.json`
  - `artifacts/eval/compare_main_online_vs_navrl.md`
  - `docs/phase1_formal_comparison_2026-04-02.md`
  - `docs/phase2_real_robot_boundary_2026-04-02.md`

## 说明

- 旧仓库完全不改，这里只复用其参数来源和环境设计。
- 当前版本优先保证闭环可跑和对比协议一致，再继续加深动态障碍密度与训练规模。
- 在 RTX 4060 8GB 上，不建议在训练进行中并发启动第二个 Isaac 实例做评测；`tools/eval_checkpoint.py` 默认会拦截这种并发。
- 第二阶段默认不直接移植上游 `onboard_detector + safe_action`，而是为 DashGo 设计激光雷达/差速友好的动态障碍表示与安全接口。
