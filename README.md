# DashGo NavRL Project

创建时间: 2026-03-26

这是一个与现有 `dashgo_rl_project` 完全分离的实验仓库，用于在本地 `Isaac Lab 2.0.2 / Isaac Sim 4.5` 环境中实现 `DashGo x NavRL-style` 训练、评测和横向对比。

## 目标

- 保留 DashGo 实车参数、URDF 和差速底盘语义。
- 迁入 NavRL 风格的 `TorchRL + TensorDict + PPO + BetaActor` 训练栈。
- 用统一的 DashGo quick/main 评测协议，对比旧仓库模型与新仓库模型。
- 第一阶段只做训练与仿真评测，不接 ROS2/TorchScript 部署链路。
