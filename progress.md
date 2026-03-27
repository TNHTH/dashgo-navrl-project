# Progress Log

创建时间: 2026-03-26

## 2026-03-26
- 初始化独立仓库 `/home/gwh/dashgo_navrl_project`
- 迁入 DashGo Isaac 环境底座、URDF 和参数快照
- 实现 TorchRL + PPO + BetaActor + TensorDict 训练闭环
- 新增 `tools/background_train.py` 和统一评测入口

## 2026-03-27
- 将奖励与训练主干收紧到 NavRL 论文方向
- 验证 `RayCaster` 无法直接覆盖旧场景后，重构为官方路线环境：`RayCaster -> /World/ground` + terrain 静态障碍 + 动态障碍真值分支
- 修复 `train_navrl.py` 的静默秒退，修复后真实入口可直接产出 `run_root / checkpoint / final_checkpoint`
- 用 `pilot / 8 env / 8192 frames` 完成吞吐基准，实测约 `243.7 frames/s`
- 启动 4 小时级正式训练：`max_frame_num=3600000`
- 当前正式 run: `pilot_20260327_134032`
- 当前 supervisor 状态: `running`
