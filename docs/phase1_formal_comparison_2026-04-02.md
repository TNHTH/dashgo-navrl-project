# Phase 1 正式对比结论

创建时间: 2026-04-02

## 结论

- 第一阶段已经完成“DashGo x NavRL-style 仿真闭环 + 统一 quick/main 正式对比”。
- 对比口径固定为：
  - 基线：旧仓库当前在线 GeoNav manifest 指向的 checkpoint
  - 候选：`pilot_20260327_134032/checkpoint_2765056.pt`
  - 平台：同一 DashGo 物理语义、同一 quick/main 协议、不同算法实现
- 当前结论不是“NavRL-style 已超过 GeoNav”，而是：
  - `quick`：候选未形成成功 episode，但旧在线基线本身严重超时/绕圈，因此候选 `score` 略高于基线
  - `main`：候选整体劣于基线，主要因为碰撞占主导
  - 两边都未通过行为 gate，说明第一阶段已经完成复现与对比收口，但候选还不能替代当前基线

## 基线与候选来源

- 旧在线基线 manifest：
  - `/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/models/policy_torchscript.manifest.json`
- 基线 checkpoint：
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260319_113548_wave50_gen2_model704_escapecurriculum05_softgeometry_seed44/checkpoints/model_883.pt`
- 候选 checkpoint：
  - `/home/gwh/dashgo_navrl_project/artifacts/runs/pilot_20260327_134032/checkpoints/checkpoint_2765056.pt`

## 正式结果

### quick

- 基线：
  - `success_rate=0.0833`
  - `collision_rate=0.0000`
  - `timeout_rate=0.9167`
  - `progress_stall_rate=0.8333`
  - `score=-65.3396`
- 候选：
  - `success_rate=0.0000`
  - `collision_rate=0.9167`
  - `timeout_rate=0.0833`
  - `progress_stall_rate=0.9167`
  - `score=-60.7047`
- 解读：
  - 基线主导失败模式是 `time_out + orbit`
  - 候选主导失败模式是 `object_collision`
  - 候选没有形成成功 episode，但平均终点距离略好于基线，且 `score` 高 `4.6349`
  - 这个结果更像“失败模式变了”，不是“候选已经赢了”

### main

- 基线：
  - `success_rate=0.1042`
  - `collision_rate=0.0000`
  - `timeout_rate=0.8958`
  - `progress_stall_rate=0.3333`
  - `score=-50.4208`
- 候选：
  - `success_rate=0.0000`
  - `collision_rate=0.8542`
  - `timeout_rate=0.1458`
  - `progress_stall_rate=0.6875`
  - `score=-56.8110`
- 解读：
  - 基线依然以 `time_out + orbit` 为主，但至少保住了零碰撞和少量成功
  - 候选则在 harder suite 中快速转为碰撞主导，平均终点距离也更差
  - `main` 是当前更可信的总体判断依据，因此第一阶段总评应以“候选整体劣于基线”收口

## 主要发现

- 旧在线 GeoNav 基线并不强，`quick/main` 都明显未过行为 gate。
- NavRL-style 候选复现已经成功收口到“可训练、可评测、可对比”，但当前训练结果不达标。
- 两边的失败模式不同：
  - 基线偏 `超时 + 绕圈 + 高饱和`
  - 候选偏 `碰撞 + 推进停滞`
- 这意味着下一轮优化不该再问“能不能复现”，而应直接问：
  - 如何压低 NavRL-style 的碰撞率
  - 如何在保留 NavRL reward 主干的前提下减少停滞

## 对应产物

- `quick` 基线：
  - `/home/gwh/dashgo_navrl_project/artifacts/eval/baseline_model_883_quick.json`
- `main` 基线：
  - `/home/gwh/dashgo_navrl_project/artifacts/eval/baseline_model_883_main.json`
- `quick` 对比：
  - `/home/gwh/dashgo_navrl_project/artifacts/eval/compare_quick_online_vs_navrl.json`
  - `/home/gwh/dashgo_navrl_project/artifacts/eval/compare_quick_online_vs_navrl.md`
- `main` 对比：
  - `/home/gwh/dashgo_navrl_project/artifacts/eval/compare_main_online_vs_navrl.json`
  - `/home/gwh/dashgo_navrl_project/artifacts/eval/compare_main_online_vs_navrl.md`
