# DashGo NavRL Project

创建时间: 2026-03-26

这是一个与现有 `dashgo_rl_project` 完全分离的实验仓库，用于在本地 `Isaac Lab 2.0.2 / Isaac Sim 4.5` 环境中实现 `DashGo x NavRL-style` 训练、评测和横向对比。

## 目标

- 保留 DashGo 实车参数、URDF 和差速底盘语义。
- 迁入 NavRL 风格的 `TorchRL + TensorDict + PPO + BetaActor` 训练栈。
- 用统一的 DashGo quick/main 评测协议，对比旧仓库模型与新仓库模型。
- 第一阶段只做训练与仿真评测，不接 ROS2/TorchScript 部署链路。

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
python3 tools/compare_models.py --baseline-checkpoint /path/to/old.pt --candidate-checkpoint /path/to/new.pt
```

## 说明

- 旧仓库完全不改，这里只复用其参数来源和环境设计。
- 当前版本优先保证闭环可跑和对比协议一致，再继续加深动态障碍密度与训练规模。
- 在 RTX 4060 8GB 上，不建议在训练进行中并发启动第二个 Isaac 实例做评测；`tools/eval_checkpoint.py` 默认会拦截这种并发。
