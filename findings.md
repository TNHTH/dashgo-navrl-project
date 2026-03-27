# Findings & Decisions

创建时间: 2026-03-26

## Requirements
- 新建独立仓库，不与现有 `dashgo_rl_project` 混在一起。
- 下载上游 `NavRL` 到本地作为只读参考。
- 保留 DashGo 的轮径、轮距、速度边界和差速动作语义。
- 训练核心切到 NavRL 风格 `TorchRL + TensorDict + PPO + BetaActor`。
- 统一 quick/main 评测协议，最终对比新旧模型。

## Research Findings
- `/home/gwh/NavRL_upstream` 已成功 clone，本地可直接查阅其 PPO/训练脚本。
- `/home/gwh/IsaacLab/_isaac_sim/python.sh` 当前 Python 版本为 `3.10.15`，Torch 版本为 `2.5.1+cu118`。
- 该 Isaac Python 初始未安装 `torchrl` 与 `tensordict`。
- `torchrl==0.6.0` 与 `tensordict==0.6.0` 可兼容当前 `torch==2.5.1+cu118`；`torchrl/tensordict 0.7+` 会向上拉 `torch>=2.6`，不适合当前环境。
- 旧 DashGo 环境 `reset()` 和 `observation_manager.compute()` 都返回 `{"policy": [N, 246]}`。
- 这 246 维观测可按旧环境定义切分为:
  - `lidar_history`: 前 216 维，对应 `72 x 3`
  - `waypoint_vector_history`: 9 维
  - `goal_vector_history`: 9 维
  - `lin_vel_history`: 3 维
  - `yaw_rate_history`: 3 维
  - `last_action_history`: 6 维
- 旧动作语义已经满足 DashGo 非对称速度边界:
  - 正向最大 `0.3 m/s`
  - 倒车最大 `0.15 m/s`
  - 角速度最大 `1.0 rad/s`
- 新 TensorDict 合同已通过探针验证:
  - `("agents","observation","state") -> [N, 1, 8]`
  - `("agents","observation","lidar") -> [N, 1, 72, 3]`
  - `("agents","observation","dynamic_obstacle") -> [N, 1, 5, 10]`
- 新 PPO 训练闭环已通过最小 collector 验证，且训练入口可成功跑完 `128` frames sanity run。
- 在 RTX 4060 Laptop 8GB 上，训练进行时并发启动第二个 Isaac 实例做评测不稳定，应串行执行训练和评测。
- Isaac Python 已内置 `tensorboard`，可直接通过 `torch.utils.tensorboard.SummaryWriter` 落盘 `.tfevents` 文件，无需额外安装。
- 当前官方路线环境已经切到真实 `RayCaster -> /World/ground` + terrain 静态障碍 + 动态障碍真值分支，不再依赖相机 fallback。
- 训练入口的静默秒退问题已经修复，根因是 `AppLauncher(args_cli)` 改写 `Namespace` 后再次读取 `enable_cameras`。
- `pilot / 8 env / 8192 frames` 的真实吞吐基线约为 `243.7 frames/s`。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 上游 `NavRL` 只放在 `/home/gwh/NavRL_upstream` 只读参考 | 避免直接在 upstream 上做 DashGo 定制 |
| 新训练环境对齐官方 NavRL 路线：`RayCaster -> /World/ground` + terrain 内静态障碍 + 动态障碍真值分支 | 满足“唯一差异是能接入 DashGo 小车”的目标 |
| 奖励主干保持 NavRL 风格 `+1 + goal_velocity + static_safety + dynamic_safety - 0.1 * smoothness` | 不再沿用旧 DashGo 项目的启发式训练 reward |
| 训练入口直接写 TensorBoard 到 `artifacts/runs/<run>/logs/tensorboard` | 便于网页直接查看训练曲线 |
| 长时间训练统一由 `tools/background_train.py` supervisor 值守 | 保证状态、日志、pid 三方可核验 |

## Resources
- `/home/gwh/NavRL_upstream/isaac-training/training/scripts/ppo.py`
- `/home/gwh/NavRL_upstream/isaac-training/training/scripts/utils.py`
- `/home/gwh/dashgo_navrl_project/apps/isaac/train_navrl.py`
- `/home/gwh/dashgo_navrl_project/tools/background_train.py`
- `/home/gwh/dashgo_navrl_project/src/dashgo_rl/dashgo_env_navrl_official.py`
- `/home/gwh/dashgo_navrl_project/configs/train/train.yaml`
