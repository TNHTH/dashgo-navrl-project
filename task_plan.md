# Task Plan: DashGo-NavRL 独立训练仓库

创建时间: 2026-03-26

## Goal
在 `/home/gwh/dashgo_navrl_project` 中实现一个独立于旧仓库的 DashGo x NavRL-style 训练、评测和模型对比闭环，并保持 DashGo 实车参数一致。

## Current Phase
Phase 18

## Phases

### Phase 1: 仓库边界与基础资料固化
- [x] 确认上游参考仓库、旧基线仓库和新实验仓库边界
- [x] 固化 DashGo 参数来源与评测协议来源
- [x] 初始化持久化文档
- **Status:** complete

### Phase 2: DashGo Isaac 环境底座迁移
- [x] 迁入 DashGo 环境、URDF 和参数文件
- [x] 修正新仓库路径解析与 YAML 参数读取
- [x] 验证新仓库可独立创建 Isaac 环境
- **Status:** complete

### Phase 3: NavRL 风格训练栈实现
- [x] 实现 TensorDict 观测合同
- [x] 实现 BetaActor + PPO + GAE + ValueNorm
- [x] 实现训练入口与 profile 配置
- **Status:** complete

### Phase 4: 统一评测与对比
- [x] 实现 quick/main 评测 worker
- [x] 统一 JSON schema
- [x] 实现 compare_models.py
- **Status:** complete

### Phase 5: 自检与交付
- [x] 安装依赖并做 import 自检
- [x] 完成 reset/step 形状自检和动作运动学自检
- [x] 尝试跑 smoke 训练并产出 checkpoint
- **Status:** complete

### Phase 10: 官方 NavRL 路线重构
- [x] 将训练环境切到 `RayCaster -> /World/ground` 单 mesh 路线
- [x] 将静态障碍改为地形 mesh 内生成，不再依赖 `obs_*` 独立刚体进入 LiDAR
- [x] 将动态障碍改为单独真值状态分支，并保持 DashGo 差速动作语义
- [x] 完成 headless reset/step 探针与最小 collector 验证
- **Status:** complete

### Phase 11: 训练入口与 Supervisor 修复
- [x] 修复 Hydra override 与 Kit 参数冲突
- [x] 修复 `AppLauncher` 改写 `Namespace` 后的静默秒退
- [x] 校准 `pilot` 档真实吞吐
- **Status:** complete

### Phase 12: 四小时正式训练
- [x] 按校准后的帧预算启动后台训练
- [x] 核对 `status.json / pid / run_root / tensorboard_root` 一致
- [x] 记录本轮正式训练入口与恢复方式
- **Status:** complete

### Phase 13: 暂停与正式评测
- [x] 暂停当前 `pilot_20260327_134032` 训练
- [x] 修复正式评测的 checkpoint 兼容问题
- [x] 基于最新已保存 checkpoint 跑完 `quick/main`
- **Status:** complete

### Phase 14: 正式基线对比与阶段封板
- [x] 固定旧仓库在线 GeoNav manifest 作为唯一基线来源
- [x] 产出旧基线 `quick/main` 正式评测 JSON
- [x] 产出 NavRL-style 与在线 GeoNav 的正式对比 JSON/Markdown
- [x] 固化第一阶段结论与第二阶段实机边界文档
- **Status:** complete

### Phase 15: 本机模式摸索与 17 小时自治训练
- [x] 为训练环境增加 `env.map_source`，支持 `dashgo_official / navrl_upstream`
- [x] 修正 official-route reward/termination 的最小改动集
- [x] 跑完本机吞吐摸索，确认 `enable_cameras=false` 且高 `num_envs` 更优
- [x] 跑通“训练完成 -> quick/main -> compare”自治短周期回归
- [x] 启动正式 `17` 小时自治训练周期
- [ ] 等待自治周期完成并收取 quick/main/compare 最终产物
- **Status:** blocked

### Phase 16: 长跑复盘驱动的 correctness blocker 修复
- [x] 交叉核对静态审查 findings、live `train.log` 与当前 status
- [x] 固化长跑失效边界与 checkpoint 处置策略
- [x] 编写正式复盘文档并回写项目台账
- [x] 停止当前 NaN 长跑并归档失败样本
- [x] 修复 `truncated -> done` 在 PPO/GAE 中的处理
- [x] 修复自治链对 failed eval payload 的假阳性完成
- [x] 为训练增加 NaN fail-fast 与首个异常快照
- [x] 禁用或修正 `navrl_upstream` 的 API 不兼容配置
- [x] 修复 benchmark 默认候选集、统计口径与并发保护
- [ ] 执行 `formal 96 env >= 8,000,000 frames` 稳定性验证
- [ ] 按用户最新要求，验证结束后不自动启动正式长训
- **Status:** in_progress

### Phase 17: TorchRL 与评测生命周期修复、记录与复盘
- [x] 修复 `TorchRLDashgoEnv._reset()` 对 TorchRL partial reset 的整批误 reset
- [x] 修复评测 worker 在 Isaac Lab auto-reset 后读取终态指标的问题
- [x] 修复 recycle env 重新布场景后未刷新观测的问题
- [x] 统一训练/评测脚本的 Isaac Python 入口解析
- [x] 补齐生命周期回归单测、真实 smoke train 与 quick eval smoke
- [x] 回写项目台账、正式修复记录与复盘
- **Status:** complete

### Phase 18: 12 小时 formal 连续训练
- [x] 核对 `formal` supervisor 空闲且无活动 `train_navrl.py`
- [x] 按 `96 env * training_frame_num=32` 对齐 12 小时预算到 `max_frame_num=28932096`
- [x] 在 Obsidian 建立正式执行记录并记录训练后需上传 GitHub 的后续动作
- [x] 启动 `formal` 后台训练并确认 `batch=0`、首个 checkpoint、`run_root/tensorboard_root`
- [ ] 持续值守并在训练结束后收取最终 checkpoint 与状态摘要
- [ ] 训练完成后整理仓库变更并上传 GitHub
- **Status:** in_progress

## Key Questions
1. 旧 DashGo 246 维观测如何稳定切分为 `state/lidar/dynamic_obstacle` 三路输入？
2. 新仓库如何在不改旧仓库的前提下复用 DashGo Isaac 环境并切换到 TorchRL PPO？
3. 评测 JSON 如何与旧仓库 `EvalMetrics` 字段完全对齐？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 上游 `NavRL` 只放在 `/home/gwh/NavRL_upstream` 只读参考 | 避免直接在 upstream 上做 DashGo 定制 |
| 新仓库 vendoring DashGo 环境底座 | 降低 Isaac 启动链路重写风险 |
| 旧 246 维观测先做无侵入切分，再叠加动态障碍 token | 先建立可跑闭环，再逐步增强环境细节 |
| 统一使用 `/home/gwh/IsaacLab/_isaac_sim/python.sh` | 与本机 Isaac Lab 2.0.2 / Isaac Sim 4.5 一致 |
| 官方路线解释为“传感器/静态场景/动态障碍分支/奖励主干对齐 NavRL，动作与底盘保留 DashGo 语义” | 满足用户“只让模型可接入 DashGo 小车”的约束 |
| 旧仓库基线固定为在线 TorchScript manifest 指向的 checkpoint | 避免把历史最佳、训练样本和线上基线混成多个比较口径 |
| 第一阶段以仿真 quick/main 正式对比为完成定义 | 保证实验先收口，再进入实机感知/安全适配 |
| 第二阶段不直接移植 upstream `onboard_detector + safe_action` | 上游依赖深度图/YOLO/service 链，与 DashGo 当前 `LaserScan` 合同不兼容 |
| `TorchRLDashgoEnv._reset()` 在 Isaac Lab 已 auto-reset 的 env 上优先复用当前观测 | 保持 TorchRL partial reset 合同，避免单个 done env 触发整批 rollout 二次清空 |
| 评测循环显式控制 done env reset，并在复位后立即重算观测 | 保证 episode 终态指标来自真正结束瞬间，下一局首拍不再吃旧观测 |
| 训练/评测工具统一通过 `src/dashgo_rl/project_paths.py::ISAAC_PYTHON` 解析入口 | 避免 `IsaacLab/_isaac_sim/python.sh` 与 `IsaacSim/python.sh` 各脚本漂移 |
| 本轮 12 小时长跑使用 `formal + dashgo_official + 96 env + cameras off`，并只启动训练不自动串评测 | 先收集连续训练稳定性证据，避免把评测故障与训练故障混线 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Isaac 探针日志过大淹没结果 | 1 | 改为把观测摘要写入临时 JSON 文件 |
| 当前正式长跑在 `frames=6147072` 后持续 NaN | 1 | 尚未修复；当前 run 已降级为 debug 样本，下一阶段先修 correctness blocker |
| 自治链可能把 failed eval JSON 当成成功产物 | 1 | 尚未修复；已升级为下一轮正式训练前的阻塞项 |
| `PPO/GAE` 忽略 `truncated` 终止语义 | 1 | 尚未修复；已升级为训练正确性阻塞项 |
| `navrl_upstream` 地图模式与当前 Isaac Lab API 高概率不兼容 | 1 | 尚未修复；已要求至少显式报错或先禁用 |
| TorchRL partial reset 与 Isaac Lab auto-reset 叠加导致 rollout 被整批误 reset | 1 | 已修复；训练侧按 `_reset` 掩码只处理目标 env，并在已 auto-reset 时复用当前观测 |
| 评测 worker 在 `env.step()` 之后统计终态，读到的是 reset 后状态 | 1 | 已修复；评测侧改为不 auto-reset 的 step 流程，完成统计后再显式 reset done env |
