# Task Plan: DashGo-NavRL 独立训练仓库

创建时间: 2026-03-26

## Goal
在 `/home/gwh/dashgo_navrl_project` 中实现一个独立于旧仓库的 DashGo x NavRL-style 训练、评测和模型对比闭环，并保持 DashGo 实车参数一致。

## Current Phase
Phase 20

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
- [x] 因用户改为要求“先测出更优训练参数”而停止并废弃首个 `96 env` formal run
- **Status:** complete

### Phase 19: GPU 利用率驱动的训练参数搜索与优化重启
- [x] 汇总现有 `48/64/96/128 env` probe 的吞吐、显存与 GPU 利用率
- [x] 追加 `128/160/192/224/256 env` 的 `131072 frames` 长探针
- [x] 选出当前已测最佳配置 `256 env`
- [x] 按 `2026-04-03 13:10:22 +0800` 结束目标重新计算预算并重启 `formal`
- [ ] 持续值守并在训练结束后收取最终 checkpoint 与状态摘要
- **Status:** in_progress

### Phase 20: GitHub 整理、提交与上传
- [x] 确认远端仓库 `TNHTH/dashgo-navrl-project` 存在且 SSH 可用
- [x] 使用远端 `origin/main` 临时 clone 规避对当前运行中工作树的扰动
- [x] 在临时 clone 中创建 `codex/bench-and-upload-20260403` 分支
- [x] 提交 `3bf74a2 Add lifecycle safeguards and benchmark tooling`
- [x] 推送分支到 GitHub
- [ ] 视需要再创建 PR 或继续补充后续提交
- **Status:** in_progress

### Phase 21: 接续训练能力改造与实机续训
- [x] 为训练入口新增 `resume_from` 与 `resume_optimizer_state` 配置
- [x] 抽离 checkpoint 读写/恢复逻辑，并兼容旧 checkpoint 的 `frame_count + inference_state_dict`
- [x] 让新 checkpoint 额外保存优化器状态，供后续受控试验使用
- [x] 补齐 resume 回归单测并通过全量 `32` 项测试
- [x] 用已完成的 `formal_20260403_003355/checkpoint_final.pt` 真实续训 `1` 个 batch 验证兼容路径
- [x] 发现“恢复优化器状态”会在当前 Isaac/PhysX/CUDA 栈上触发 illegal instruction，并切回默认禁用
- [x] 基于兼容续训模式启动新的正式长窗口续训
- [x] 挂起独立 watchdog，到 `2026-04-03 13:10:00 +0800` 自动停机
- [ ] 持续值守并在训练结束后收取最终 checkpoint 与状态摘要
- **Status:** in_progress

### Phase 22: 暂停当前续训并评估当前仿真效果
- [x] 停止当前 `formal_20260403_100334` 续训进程
- [x] 锁定当前最新 checkpoint `checkpoint_242491392.pt`
- [x] 基于当前 checkpoint 跑完 `quick` 仿真评测
- [x] 基于当前 checkpoint 跑完 `main` 仿真评测
- [x] 与现有 baseline JSON 做并排对比
- [x] 回写“当前仿真效果”结论到项目台账
- **Status:** complete

### Phase 23: 根因检查与旧 DashGo 对比
- [x] 核对当前正式训练主链实际使用的是 `dashgo_env_navrl_official.py`
- [x] 对比当前 `official` 奖励主干与旧 `dashgo_env_v2.py` 的 shaping 差异
- [x] 对比 `env_adapter` 压缩后的观测与旧 `246` 维合同差异
- [x] 核对 reverse case 训练分布是否覆盖旧 DashGo 的 recovery 语义
- [x] 核对旧 DashGo ROS2 控制链中 `heading_guard/recovery/safety_filter` 的额外守护
- [x] 回写“为什么当前效果更差”的结构化结论到项目台账
- **Status:** complete

### Phase 24: 保留 NavRL 主体思路并修正 DashGo 适配偏差
- [x] 对照 upstream `NavRL` 核对当前 DashGo `official` 线的关键偏差点
- [x] 把 `env_adapter` 的 state 分支恢复为完整 `30` 维非 LiDAR 切片
- [x] 把 `official` 环境 LiDAR 改回全向 `360°` 感知
- [x] 在不回退旧 DashGo 厚 shaping 的前提下，为当前成功终止语义补充对齐的 goal bonus
- [x] 增补回归测试，锁住 `30` 维 state / `360°` LiDAR / goal bonus 三个约束
- [x] 跑通定向 `11` 项测试
- [x] 跑通全量 `34` 项测试
- **Status:** complete

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
| 在用户明确指出 GPU 利用率偏低后，优先改为“先测参再长跑” | 长跑前先把吞吐和 GPU 占用压到更优，比盲目按旧配置跑满更有效 |
| 当前已测最佳长探针配置升级为 `formal + dashgo_official + cameras off + env.num_envs=256` | `131072 frames` 探针下测得 `3885.6 fps / 99% GPU / 62.4% memory_ratio`，优于 `96/128/160/192/224 env` |
| 上传 GitHub 不直接改写当前正在训练的工作树，而是通过 `origin/main` 临时 clone 提交推送新分支 | 避免训练过程中文件系统切换扰动 live run |
| 接续训练默认采用“恢复模型参数 + value_norm + frame_count，但默认不恢复优化器状态” | 旧 checkpoint 兼容续训已实机通过；恢复优化器状态在当前 Isaac/PhysX/CUDA 栈上触发 illegal instruction，稳定性优先 |

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
