# Task Plan: DashGo-NavRL 独立训练仓库

创建时间: 2026-03-26

## Goal
在 `/home/gwh/dashgo_navrl_project` 中实现一个独立于旧仓库的 DashGo x NavRL-style 训练、评测和模型对比闭环，并保持 DashGo 实车参数一致。

## Current Phase
Phase 12

## Phases

### Phase 1: 仓库边界与基础资料固化
- [x] 确认上游参考仓库、旧基线仓库和新实验仓库边界
- [x] 固化 DashGo 参数来源与评测协议来源
- [x] 初始化持久化文档
- **Status:** complete

### Phase 2: DashGo Isaac 环境底座迁移
- [ ] 迁入 DashGo 环境、URDF 和参数文件
- [ ] 修正新仓库路径解析与 YAML 参数读取
- [ ] 验证新仓库可独立创建 Isaac 环境
- **Status:** in_progress

### Phase 3: NavRL 风格训练栈实现
- [ ] 实现 TensorDict 观测合同
- [ ] 实现 BetaActor + PPO + GAE + ValueNorm
- [ ] 实现训练入口与 profile 配置
- **Status:** pending

### Phase 4: 统一评测与对比
- [ ] 实现 quick/main 评测 worker
- [ ] 统一 JSON schema
- [ ] 实现 compare_models.py
- **Status:** pending

### Phase 5: 自检与交付
- [ ] 安装依赖并做 import 自检
- [ ] 完成 reset/step 形状自检和动作运动学自检
- [ ] 尝试跑 smoke 训练并产出 checkpoint
- **Status:** pending

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
- [ ] 按校准后的帧预算启动后台训练
- [ ] 核对 `status.json / pid / run_root / tensorboard_root` 一致
- [ ] 记录本轮正式训练入口与恢复方式
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

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Isaac 探针日志过大淹没结果 | 1 | 改为把观测摘要写入临时 JSON 文件 |
