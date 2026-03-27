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
- 这轮 `pilot_20260327_004917` 训练的旧格式 checkpoint 只包含 `feature_extractor / actor / critic` 参数，不包含 `value_norm / gae` buffer；评测侧需要兼容 legacy checkpoint 才能正常读取。
- `pilot_20260327_004917` 的训练数值表现“稳定但早熟”：
  - `1050` 条训练采样日志里，后段 `entropy` 基本降到 `0`
  - late window `explained_var_mean≈0.8447`，`critic_mean≈0.0530`
  - 说明 value 学得不差，但 policy 基本已收缩到近确定性动作分布
- `pilot_20260327_004917` 的任务效果明显不达标：
  - quick: `success_rate=0.0`, `collision_rate=0.4167`, `timeout_rate=0.5833`, `score=-18.20`
  - main: `success_rate=0.0`, `collision_rate=0.5208`, `timeout_rate=0.4792`, `score=-20.35`
  - 大多数 episode 能把距离压到 `0.5m~0.9m`，但无法稳定收进 `0.5m` 终止半径，最终以超时或碰撞结束
- 针对用户要求“严格按 NavRL 论文策略训练”，当前仓库又进一步收紧为：
  - PPO 超参对齐上游：`num_minibatches=16`、`entropy_loss_coefficient=1e-3`、`actor/critic clip_ratio=0.1`
  - 关闭旧 DashGo 的自适应课程学习
  - 终止条件只保留 `time_out / reach_goal / object_collision`
  - `object_collision` 改为 NavRL 风格的 clearance-based 安全半径判定，不再沿用 contact-force 逻辑
  - 唯一保留的非论文差异是 DashGo 的动作物理边界；不照抄无人机 `2.0 m/s` 动作上限
- 新一轮论文对齐训练已于 `2026-03-27` 启动，run 为 `pilot_20260327_121705`，目标帧数 `3,329,024`，按上一轮真实吞吐估算约 `4` 小时。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 迁入 `src/dashgo_rl/dashgo_env_v2.py` 作为新仓库 Isaac 底座 | 旧环境已验证可在本机版本启动 |
| 新训练器用外层 TensorDict 适配器而不是直接重写环境管理器 | 降低侵入面，便于快速形成闭环 |
| 评测指标直接对齐旧仓库 `EvalMetrics` | 让最终对比表可直接横向比较 |
| DashGo 参数文件在新仓库保存为 `configs/robot/dashgo_driver.yaml` | 保持来源清晰且不依赖旧仓库路径 |
| `BetaActor` 直接输出单 agent 维 `[N,1,action_dim]` | 与 TorchRL `("agents","action")` spec 对齐，避免 rollout 时补维 |
| full smoke 训练通过 `tools/background_train.py` 后台值守 | 让长跑训练具备可观测状态文件、日志与 pid 管理 |
| 评测入口默认阻止与训练并发 | 避免同机双 Isaac 实例在 8GB 显存上触发底层初始化异常 |
| 奖励主干切换为 NavRL 风格 `+1 + goal_velocity + static_safety + dynamic_safety - 0.1 * smoothness` | 用户明确要求不再沿用旧 DashGo 项目的 stall/orbit/reverse_escape 等启发式训练策略 |
| `reach_goal` 终止条件改为纯距离阈值 `0.5m` | 更贴近上游 NavRL 的目标到达判据，不再要求“到点且基本停住” |
| 训练入口直接写 TensorBoard event 到 `artifacts/runs/<run>/logs/tensorboard` | 让用户可直接用网页查看训练曲线，无需二次解析 stdout |
| `pilot` 作为当前长跑档位，帧数设为 `5,376,000` | 在 8GB 4060 上已验证可启动，场景复杂度高于 `smoke`，同时风险明显低于 `main` |
| 评测入口兼容 legacy checkpoint 的缺失键 | 当前这轮长跑 checkpoint 已经生成，不能重训才能评估；必须允许缺少 `value_norm / gae` 的非推理键 |
| “严格按论文策略训练”在 DashGo 上解释为：对齐 PPO 主干、reward 主干、collision 终止和无课程学习；不照抄无人机动作上限 | 用户要求论文策略，但平台仍是 DashGo 差速底盘，必须保留实车动作语义 |
| 官方路线新环境单独落在 `src/dashgo_rl/dashgo_env_navrl_official.py` | 保留 `dashgo_env_v2.py` 作为旧基线实验线，避免继续混线 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Isaac 启动日志会遮蔽观测探针输出 | 让 Python 先写 JSON 文件，再由 shell 读取 |
| `SafeProbabilisticModule` 对自定义 Beta 分布发出 `deterministic_sample` 警告 | 在 `IndependentBeta` 中补齐 `deterministic_sample` 属性 |
| 并发评测时 `eval_worker.py` 抛 `AttributeError: 'Articulation' object has no attribute '_data'` | 不再并发评测；通过入口层并发保护给出明确阻塞信息 |
| 首轮 smoke 训练完成时仍在使用旧 reward 图谱 | 该轮 checkpoint 仅保留作调试记录；后续正式对比必须基于新 reward 重新启动 |
| `profiles=pilot/main` 最初不会覆盖根配置，实际总是回落到 `smoke` | 调整 `train.yaml` 的 defaults 顺序，并为 profile 文件声明 `# @package _global_` |
| `eval_worker.py` 以 `strict=True` 读取 legacy checkpoint 会失败 | 改为允许缺少 `value_norm / gae` 相关键，并对非关键缺失给出兼容说明 |
| 仅替换 reward 还不够“严格对齐论文”，课程学习与 contact-force collision 也会改变训练策略 | 因此进一步关闭 curriculum，并切回 NavRL 风格 collision 终止 |

## Resources
- `/home/gwh/NavRL_upstream/isaac-training/training/scripts/ppo.py`
- `/home/gwh/NavRL_upstream/isaac-training/training/scripts/utils.py`
- `/home/gwh/dashgo_rl_project/src/dashgo_rl/dashgo_env_v2.py`
- `/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_driver_ros2/config/dashgo_driver.yaml`
- `/home/gwh/dashgo_rl_project/tools/diagnostics/eval_checkpoint.py`
- `/home/gwh/dashgo_navrl_project/tools/background_train.py`
- `/home/gwh/dashgo_navrl_project/apps/isaac/train_navrl.py`
- `/home/gwh/dashgo_navrl_project/configs/train/train.yaml`

## 2026-03-27 Additional Findings: RayCaster 与低渲染
- `apps/isaac/train_navrl.py` 与 `apps/isaac/eval_worker.py` 已移除“无条件强制 `enable_cameras=True`”的硬编码，后续是否开相机由配置驱动。
- `src/dashgo_rl/dashgo_env_v2.py` 的渲染已降到低开销档：
  - `antialiasing_mode="Off"`
  - `enable_direct_lighting=False`
  - `enable_shadows=False`
  - `enable_ambient_occlusion=False`
  - `samples_per_pixel=1`
- 原生 `RayCaster` 在本机 Isaac Lab 2.0.2 上无法直接替代当前训练场景的相机伪 LiDAR。验证结果见 [raycast_validation_20260327.json](/home/gwh/dashgo_navrl_project/artifacts/diagnostics/raycast_validation_20260327.json)。
- 多障碍 mesh 验证直接失败：
  - `NotImplementedError('RayCaster currently only supports one mesh prim. Received: 2')`
- 单障碍根路径 `/World/envs/env_0/Obs_In_1` 也无法作为 raycast 目标：
  - `RuntimeError('Invalid mesh prim path: /World/envs/env_0/Obs_In_1')`
- 这说明当前场景里的障碍原语不是 `RayCaster` 当前实现认可的 `Mesh/Plane` 静态 mesh；同时它也无法覆盖多障碍和运动学障碍。
- 在同一验证中，相机 fallback 正常工作：
  - 手工把障碍摆到机器人正前方 `1.2m` 时，camera backend 读到 `initial_min_distance≈1.4623m`
- 对场景 prim 树做了进一步实证检查，结果保存在 `/tmp/inspect_dashgo_prim_tree.json`：
  - `/World/envs/env_0/Obs_In_1/geometry/mesh` 的类型是 `Cylinder`
  - `/World/envs/env_0/Obs_In_2/geometry/mesh` 的类型是 `Cube`
  - `/World/envs/env_0/Obs_Out_1/geometry/mesh` 的类型是 `Cube`
  - 只有 `/World/ground/terrain/mesh` 的类型是真正的 `Mesh`
- 这与本机 Isaac Lab 2.0.2 的 `RayCaster` 源码约束完全一致：
  - 只接受 `len(mesh_prim_paths) == 1`
  - 只会寻找 `Plane` 或 `Mesh` 类型的 prim
  - 内部只对 `self.cfg.mesh_prim_paths[0]` 对应的单个 warp mesh 做 `raycast_mesh(...)`
- 因此在当前场景上，`RayCaster` 最多只能可靠命中地面 mesh，不能把这些 `Cube/Cylinder` 障碍物直接纳入同一条 raycast 链路。
- 更关键的是，当前障碍不是静态摆设：
  - 静态/运动学障碍会在 reset 时通过 `_write_kinematic_obstacle_pose(...)` 直接改 root pose
  - Gen2 动态障碍会在 `configure_dynamic_obstacles(...)` 和 `animate_dynamic_obstacles(...)` 中持续改位姿
  - 而 `RayCaster` 在初始化时就把目标几何转换成 warp mesh，后续更新只跟踪传感器位姿，不会重建多障碍几何
- 结论：
  - 当前仓库训练主链必须保留深度相机伪 LiDAR
  - 若未来要切到真正 raycast，必须先重构场景几何，把静态障碍烘焙成单个可 raycast 的 mesh，并重做障碍进入 LiDAR 的方案

## 2026-03-27 Additional Findings: 为什么官方 NavRL 可以直接用 RayCaster
- 官方训练环境的 LiDAR 配置本身就只打单个地形 mesh：
  - `mesh_prim_paths=["/World/ground"]`
  - 对应代码在 `/home/gwh/NavRL_upstream/isaac-training/training/scripts/env.py`
- 官方静态障碍是通过 `TerrainImporterCfg + HfDiscreteObstaclesTerrainCfg` 生成到 `/World/ground` 这张地形里，而不是像当前 DashGo 场景这样单独生成一批 `obs_*` 刚体。
- 因此官方 LiDAR 的工作前提是：
  - 所有静态障碍已经被烘焙进一张静态地形 mesh
  - RayCaster 只需要对这一张 mesh 做 raycast
- 官方动态障碍并不依赖 LiDAR 感知：
  - 代码里动态障碍单独维护 `dyn_obs_state / dyn_obs_vel / dyn_obs_size`
  - 网络输入里的 `dynamic_obstacle` 来自这些真值状态，而不是 RayCaster 命中结果
  - 动态碰撞和动态安全 reward 也基于这些解析量直接计算
- 所以“官方代码可以直接用”的真实原因不是它的 RayCaster 更强，而是它的场景设计刚好满足 RayCaster 的限制：
  - 单静态 mesh 负责静态障碍
  - 动态障碍走单独真值分支
- 当前 DashGo 场景不满足这个前提：
  - `obs_*` 障碍是独立 `Cube/Cylinder` 原语，不在 `/World/ground` 这张 mesh 内
  - 其中一部分障碍还会在 reset/interval 中持续改位姿
  - 直接照搬官方 RayCaster 配置只会让它看到地面或部分静态地形，而看不到当前关键障碍

## 2026-03-27 Additional Findings: 官方路线 DashGo 环境已落地
- 新环境文件：`/home/gwh/dashgo_navrl_project/src/dashgo_rl/dashgo_env_navrl_official.py`
- 新环境与官方 NavRL 的结构对齐方式：
  - `RayCaster` 只打 `/World/ground`
  - 静态障碍通过 `HfDiscreteObstaclesTerrainCfg` 直接生成进 terrain mesh
  - 动态障碍通过 `RigidObjectCollectionCfg` 独立维护，并从真值状态分支输出 token
  - 奖励继续保持 `+1 + goal_velocity + static_safety + dynamic_safety - 0.1 * smoothness`
- 训练/评测入口已经切到新环境：
  - `src/navrl_dashgo/env_adapter.py`
  - `apps/isaac/eval_worker.py`
  - `configs/train/train.yaml` 默认 `enable_cameras=false`
- headless reset/step 探针已通过，结果见 [navrl_official_probe_20260327.json](/home/gwh/dashgo_navrl_project/artifacts/diagnostics/navrl_official_probe_20260327.json)：
  - `obs_shape=[2,246]`
  - `scan_shape=[2,72]`
  - `token_shape=[2,5,10]`
  - `scene_keys=["dynamic_obstacles","lidar","robot","sky_light","sun_light","terrain"]`
- 最小 collector 探针已通过，结果见 [navrl_official_collector_probe_20260327.json](/home/gwh/dashgo_navrl_project/artifacts/diagnostics/navrl_official_collector_probe_20260327.json)：
  - `SyncDataCollector -> PPO.update()` 可运行
  - `actor_loss≈0.00898`
  - `critic_loss≈0.44308`
  - `entropy≈1.60e-4`
  - `explained_var≈-0.4289`
- 结论：
  - 现在已经不再依赖相机 fallback 才能跑训练闭环
  - 后续正式训练应基于这条新环境线，而不是旧 `dashgo_env_v2.py`

## 2026-03-27 Additional Findings: 训练入口秒退的根因已修复
- 症状：
  - `background_train.py start ...` 会拉起进程，但几秒后变成 `blocked_exited`
  - `train.log` 只有 Isaac 启动噪音，没有 `run_root/batch/final_checkpoint`
- 根因不是 PPO 或环境本身，而是训练入口的启动链问题：
  - `AppLauncher(args_cli)` 会改写传入的 `argparse.Namespace`
  - 后续代码仍然从 `args_cli.enable_cameras` 取值，触发 `AttributeError`
  - 因为 `finally` 里会正常 `simulation_app.close()`，进程表现为“静默正常退出”，exit code 仍是 `0`
- 同时修了第二个启动链风险：
  - 在调用 `AppLauncher` 前显式从 `sys.argv` 里剥离 Hydra override，避免 Kit 误读非 argparse 参数
- 修复后实测：
  - 真实入口 `apps/isaac/train_navrl.py` 已能直接产出 `run_root/tensorboard_root/checkpoint/final_checkpoint`
  - 不再需要临时 probe 脚本才能完成 `collector -> PPO.update()` 闭环
- 4 小时训练的吞吐基线也已拿到：
  - `pilot` 档、`8 env`、`max_frame_num=8192`
  - 实测总耗时约 `33.62s`
  - 吞吐约 `243.7 frames/s`
  - 折算 4 小时训练预算约 `3.5M frames`
  - 正式后台训练采用 `3.6M frames` 留少量缓冲
