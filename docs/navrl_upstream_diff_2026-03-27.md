# DashGo-NavRL 相对上游 NavRL 的修改说明

创建时间: 2026-03-27

## 文档目的

本文档说明当前仓库 `/home/gwh/dashgo_navrl_project` 相对于上游 GitHub 项目 `/home/gwh/NavRL_upstream` 做了哪些结构性修改，以及这些修改为什么存在。

对照范围主要基于以下上游文件：

- `NavRL_upstream/isaac-training/training/scripts/train.py`
- `NavRL_upstream/isaac-training/training/scripts/ppo.py`
- `NavRL_upstream/isaac-training/training/scripts/env.py`
- `NavRL_upstream/isaac-training/training/cfg/train.yaml`

对照范围主要对应以下当前仓库文件：

- `apps/isaac/train_navrl.py`
- `apps/isaac/eval_worker.py`
- `src/navrl_dashgo/ppo.py`
- `src/navrl_dashgo/env_adapter.py`
- `src/dashgo_rl/dashgo_env_navrl_official.py`
- `configs/train/train.yaml`

## 一句话总结

当前仓库不是对上游 NavRL 的原地改版，而是把上游的 `TorchRL + TensorDict + PPO + BetaActor + NavRL 风格观测/奖励` 迁移到一个独立的新仓库中，并将平台从“无人机安全飞行”改成“DashGo 差速小车局部导航”。

## 总体改动概览

| 维度 | 上游 NavRL | 当前项目 |
|---|---|---|
| 仓库定位 | 一个同时包含训练、quick demos、ROS1、ROS2 部署的总仓库 | 一个只做 DashGo 训练、评测、对比的独立实验仓库 |
| 仿真栈 | Isaac Sim 2023.1.0-hotfix.1 + OmniDrones/Orbit | Isaac Lab 2.0.2 / Isaac Sim 4.5 |
| 机器人平台 | Hummingbird 无人机 | DashGo 差速底盘小车 |
| 动作语义 | 速度动作先经世界坐标/控制器转换，再交给无人机执行器 | 直接输出局部 `[v, w]`，再映射为左右轮角速度 |
| 静态场景 | 单地形 mesh + LiDAR raycast | 保持同一路线，但换成适配 DashGo 的 2D 地面导航场景 |
| 动态障碍 | 真实状态分支 + 空中/地面混合障碍 | 保持“真值 token 分支”，改成地面障碍版本 |
| 奖励目标 | 安全飞行 | 安全地面导航 |
| 评测协议 | 上游训练过程自带 evaluation，偏训练内部统计 | 单独做 `quick/main` 正式评测与旧 DashGo 模型对比 |
| 工程化 | Hydra + Wandb 为主 | 增加 TensorBoard、后台 supervisor、统一评测 JSON、模型对比工具 |

## 1. 仓库边界与工程组织

### 上游项目是什么

上游 `NavRL` 是一个总仓库，包含：

- Isaac 训练环境
- quick demos
- ROS1 部署
- ROS2 部署
- 无人机平台相关资源

### 当前项目改了什么

当前仓库把目标范围收缩成一个独立的 DashGo 实验仓库，只保留：

- 训练入口
- 评测入口
- 模型对比脚本
- DashGo 机器人资产和参数快照

没有继续迁入上游这些部分：

- quick demos
- ROS1/ROS2 部署链路
- 无人机控制器与无人机资产

### 对应文件

- 新训练入口：`apps/isaac/train_navrl.py`
- 新评测入口：`tools/eval_checkpoint.py`
- 新对比工具：`tools/compare_models.py`
- 新后台训练 supervisor：`tools/background_train.py`

## 2. 运行时与依赖栈修改

### 上游项目是什么

上游 README 明确要求：

- Isaac Sim `2023.1.0-hotfix.1`
- 通过 `setup.sh` 建立 `NavRL` conda 环境
- 训练脚本直接运行在上游那套旧版仿真/Orbit 接口上

### 当前项目改了什么

当前项目不再回退到上游的旧版 Isaac Sim，而是统一迁移到当前机器已有的：

- Isaac Lab `2.0.2`
- Isaac Sim `4.5`
- Python 运行时统一使用 `/home/gwh/IsaacLab/_isaac_sim/python.sh`

这意味着以下内容都被重写或适配过：

- 环境 API
- 传感器 API
- terrain 生成接口
- 训练入口启动顺序
- Hydra 与 Kit 参数兼容方式

### 对应文件

- 训练配置：`configs/train/train.yaml`
- 启动入口：`apps/isaac/train_navrl.py`
- 项目路径适配：`src/dashgo_rl/project_paths.py`

## 3. 机器人平台从无人机改成 DashGo 小车

### 上游项目是什么

上游 `env.py` 用的是：

- `Hummingbird` 无人机
- `LeePositionController`
- 3D 飞行动力学
- 高度约束和高度惩罚

### 当前项目改了什么

当前项目把平台完全换成 DashGo 差速小车：

- 使用 DashGo URDF 和底盘参数
- 保留速度控制思路，但去掉无人机专用控制器
- 保留局部导航问题，不再是 3D 飞行任务

DashGo 的实车参数来自旧仓库和 ROS driver 快照，包括：

- `wheel_diameter = 0.1264`
- `wheel_track = 0.342`
- `max_lin_vel = 0.3`
- `max_reverse_speed = 0.15`
- `max_ang_vel = 1.0`

### 对应文件

- 机器人资产：`configs/robot/dashgo.urdf`
- 参数快照：`configs/robot/dashgo_driver.yaml`
- 资产装载：`src/dashgo_rl/dashgo_assets.py`
- 参数封装：`src/dashgo_rl/dashgo_config.py`

## 4. 动作语义修改

### 上游项目是什么

上游 `ppo.py` 的 actor 输出会经过：

- `action_normalized`
- 缩放到动作范围
- `vec_to_world(...)`
- 再经 `LeePositionController` 与 `VelController`

也就是说，上游动作语义本质上仍然服务于无人机。

### 当前项目改了什么

当前项目改为 DashGo 的局部速度控制：

- actor 输出 2 维动作
- 语义固定为 `[v, w]`
- 直接映射为左右轮角速度
- 不再做世界坐标速度变换
- 不再使用 `LeePositionController`

这属于最关键的平台适配之一，因为如果继续保留上游世界系动作语义，动作不会自然匹配差速底盘。

### 对应文件

- DashGo 官方路线动作：`src/dashgo_rl/dashgo_env_navrl_official.py`
- PPO 动作输出：`src/navrl_dashgo/ppo.py`

## 5. 观测与输入结构修改

### 上游项目是什么

上游 NavRL 的网络输入由三部分组成：

- `state`
- `lidar`
- `dynamic_obstacle`

LiDAR 走 CNN，动态障碍走单独分支，然后再拼接进特征抽取器。

### 当前项目改了什么

当前项目保留这个总体结构，但把内容改成适配 DashGo：

- `state` 改成地面导航所需的 8 维状态
- `lidar` 改成前向 72 维 2D 地面扫描
- `dynamic_obstacle` 改成最多 5 个地面动态障碍 token，每个 10 维

此外，当前项目还加了一层 `env_adapter`，把 DashGo 环境观测转换成 TorchRL 需要的 TensorDict 合同。

### 对应文件

- 观测适配：`src/navrl_dashgo/env_adapter.py`
- TorchRL 工具：`src/navrl_dashgo/torchrl_utils.py`
- DashGo 官方路线环境：`src/dashgo_rl/dashgo_env_navrl_official.py`

## 6. LiDAR 与场景路线修改

### 上游项目是什么

上游官方 LiDAR 路线的关键假设是：

- `RayCaster` 只打 `/World/ground`
- 静态障碍被烘焙进 terrain mesh
- 动态障碍不依赖 LiDAR，而是单独从真值状态进入网络

### 当前项目改了什么

当前项目最终对齐到了这条“官方路线”，但做了地面机器人适配：

- 静态障碍改为 `HfDiscreteObstaclesTerrainCfg` 生成进地形 mesh
- LiDAR 使用 `RayCaster` 且只打 `/World/ground`
- 动态障碍使用 `RigidObjectCollectionCfg` 维护真值状态
- 不再依赖旧版“深度相机伪 LiDAR”作为主训练路线

同时，当前仓库保留了一个历史兼容环境：

- `src/dashgo_rl/dashgo_env_v2.py`

它主要用于保留旧 DashGo 环境底座和调试，不是当前推荐的主训练环境。

### 对应文件

- 官方路线环境：`src/dashgo_rl/dashgo_env_navrl_official.py`
- 历史兼容环境：`src/dashgo_rl/dashgo_env_v2.py`
- RayCaster 可行性验证：`apps/isaac/validate_raycast_backend.py`

## 7. 动态障碍实现修改

### 上游项目是什么

上游动态障碍是：

- 多种尺寸类别
- 一部分是“3D 障碍”
- 一部分是“2D 长柱障碍”
- 真实状态单独送入 `dynamic_obstacle` 分支

### 当前项目改了什么

当前项目保留“动态障碍真值分支”的方法，但全部改成地面导航语义：

- 障碍物是地面场景中的 cylinder / cuboid
- 以本地相对位置、速度、尺寸、类型等组装 token
- 控制在固定槽位和固定维度，方便和网络结构兼容

这一步属于“沿用 NavRL 思路，但改成 DashGo 场景”。

### 对应文件

- 动态障碍状态与 token：`src/dashgo_rl/dashgo_env_navrl_official.py`

## 8. 奖励函数与终止条件修改

### 上游项目是什么

从上游 `env.py` 可见，奖励主干大致是：

- `reward_vel`
- `+1` 生存偏置
- `reward_safety_static`
- `reward_safety_dynamic`
- `-0.1 * penalty_smooth`
- `-8.0 * penalty_height`

终止条件包含：

- 越界
- 碰撞
- 等任务相关条件

### 当前项目改了什么

当前项目在“官方路线”里尽量保留 NavRL 的主干结构，但把无人机专属部分去掉：

- 保留 `+1` 生存偏置
- 保留目标方向速度奖励
- 保留静态/动态障碍安全项
- 保留平滑惩罚
- 去掉高度惩罚
- 去掉飞行越界逻辑
- 终止条件收缩为更适合 DashGo 的：
  - `time_out`
  - `reach_goal`
  - `object_collision`

也就是说，当前不是照抄无人机 reward，而是做了“论文方向保留，平台语义替换”。

### 对应文件

- 奖励和终止：`src/dashgo_rl/dashgo_env_navrl_official.py`

## 9. PPO 与训练器修改

### 上游项目是什么

上游 PPO 的核心结构是：

- LiDAR CNN
- 动态障碍 MLP 分支
- BetaActor
- GAE
- ValueNorm
- SyncDataCollector

### 当前项目改了什么

当前项目保留了这些核心组件，但做了以下工程化适配：

- 改成适配 DashGo 的 observation spec / action spec
- 将上游训练入口拆成更清晰的本地模块：
  - `src/navrl_dashgo/ppo.py`
  - `src/navrl_dashgo/torchrl_utils.py`
- 训练入口直接写 TensorBoard
- checkpoint 增加 `inference_state_dict` 和配置快照
- 兼容当前 Isaac Lab/Kit 的启动顺序

### 对应文件

- PPO：`src/navrl_dashgo/ppo.py`
- TorchRL 工具：`src/navrl_dashgo/torchrl_utils.py`
- 训练入口：`apps/isaac/train_navrl.py`

## 10. 配置系统修改

### 上游项目是什么

上游使用 Hydra 配置，但训练配置更偏向无人机和大规模环境，例如：

- `max_frame_num: 12e8`
- `env.num_envs: 2` 示例，README 中也给出 `1024` 规模训练
- `env.num_obstacles: 350`
- `env_dyn.num_obstacles: 80`

### 当前项目改了什么

当前项目的配置改成适配本机 RTX 4060 8GB 和 DashGo：

- 训练 profile 分成 `smoke / pilot / main`
- 明确区分静态障碍数和动态障碍数
- 改用 `episode_length_s`
- 去掉无人机专用配置项
- 默认关闭 `enable_cameras`

### 对应文件

- 主配置：`configs/train/train.yaml`
- 档位配置：`configs/train/profiles/smoke.yaml`
- 档位配置：`configs/train/profiles/pilot.yaml`
- 档位配置：`configs/train/profiles/main.yaml`

## 11. 评测协议修改

### 上游项目是什么

上游训练入口里带有训练过程中的 `evaluate(...)`，但更偏向训练内循环统计，不是为了和另一个项目做统一横向对比。

### 当前项目改了什么

当前项目新增了“正式评测协议”：

- `quick/main` 两套 suite
- 固定 JSON schema
- 固定指标集合
- 可以和旧 DashGo 模型做同口径对比

指标包括：

- `success_rate`
- `collision_rate`
- `timeout_rate`
- `time_to_goal`
- `path_efficiency`
- `spin_proxy_rate`
- `progress_stall_rate`
- `score`

### 对应文件

- 评测 worker：`apps/isaac/eval_worker.py`
- 评测入口：`tools/eval_checkpoint.py`
- 指标定义：`src/navrl_dashgo/metrics.py`
- 模型对比：`tools/compare_models.py`

## 12. 可观测性与后台训练修改

### 上游项目是什么

上游以 Wandb 为主，缺少本地长期后台训练管理。

### 当前项目改了什么

当前项目新增了更适合本机值守和本地实验的工程能力：

- TensorBoard 事件文件
- 后台 supervisor
- `status.json`
- `train.log`
- `tensorboard_root`
- 后台启动/停止/状态查询脚本

这部分不是算法修改，但它是当前项目相对上游一个很实用的工程补充。

### 对应文件

- 训练日志与 TensorBoard：`apps/isaac/train_navrl.py`
- supervisor：`tools/background_train.py`
- 运行目录组织：`src/navrl_dashgo/runtime.py`

## 13. 删除或未迁入的上游能力

以下上游能力当前没有迁入本仓库：

- 无人机 quick demos
- ROS1 部署链路
- ROS2 部署链路
- LeePositionController / VelController 无人机控制链
- 上游的多种无人机资产与 OmniDrones 完整生态

这是有意裁剪，不是缺失。当前仓库目标是：

- 先把 DashGo 训练、评测、对比闭环做稳
- 不把 ROS 部署和无人机资产混进这个实验仓库

## 14. 当前项目的保留项

以下内容是当前项目故意从上游保留的核心思想：

- `TorchRL + TensorDict` 训练组织方式
- `PPO + BetaActor + GAE + ValueNorm`
- LiDAR CNN 分支
- 动态障碍单独 token 分支
- NavRL 风格安全奖励主干
- `RayCaster -> static mesh` 的官方场景路线

也就是说，当前项目并不是“另起炉灶重写一个 PPO”，而是保留 NavRL 的训练思想，再做 DashGo 平台化改造。

## 15. 当前项目独有的额外修改

除了平台适配，当前项目还额外做了这些上游没有的事情：

- 把旧 DashGo 项目的底盘参数、URDF 和局部导航语义迁入新仓库
- 为当前 Isaac Lab 版本修复 Hydra/Kit 启动兼容
- 处理 checkpoint 与评测加载的兼容问题
- 增加与旧 DashGo 模型的统一对比工具
- 增加正式评测 JSON 和行为 gate

## 16. 最终结论

相对于上游 GitHub 原始 NavRL，当前项目可以概括为：

1. 从“无人机安全飞行仓库”改造成“DashGo 地面机器人局部导航仓库”。
2. 从“老版本 Isaac Sim / OmniDrones 路线”迁移到“本机 Isaac Lab 2.0.2 / Isaac Sim 4.5”。
3. 保留 NavRL 的 PPO、TensorDict、LiDAR CNN、动态障碍 token 和安全奖励思想。
4. 去掉无人机控制器和 3D 飞行专属逻辑，替换成 DashGo 的差速底盘动作与地面导航语义。
5. 新增正式评测、模型对比、TensorBoard 和后台训练管理，使其更适合本地长期实验和模型横向比较。
