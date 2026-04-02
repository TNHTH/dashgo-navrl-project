# DashGo NavRL 修复与验证记录

创建时间: 2026-04-02 23:08

## 1. 背景与处置边界
- 目标: 修复 `pilot + 96 env + dashgo_official + camoff` 长跑暴露出的 correctness blocker，并完成可审计验证。
- 当前不做的事: 不从 `pilot_20260402_210831` 续训；不在本轮结束时启动新的正式长训。
- 当前废弃 run:
  - `run_root=/home/gwh/dashgo_navrl_project/artifacts/runs/pilot_20260402_210831`
  - `latest_checkpoint=/home/gwh/dashgo_navrl_project/artifacts/runs/pilot_20260402_210831/checkpoints/checkpoint_16899072.pt`
  - 首次 NaN:
    - `train.log:3166`
    - `frames=6147072`
  - supervisor 状态:
    - `artifacts/supervisor/pilot/status.json`
    - `abandoned=true`
    - `current_run_invalid=true`
  - run 级记录:
    - `artifacts/runs/pilot_20260402_210831/supervisor_status.json`

## 2. 已完成修复

### 2.1 Supervisor / 自治链
- `tools/background_train.py`
  - 新增 `attempt_id / started_command_hash / latest_final_checkpoint / failure_reason / abandoned / current_run_invalid`
  - 支持 `abandon` 子命令
  - 废弃时同步写 `run_root/supervisor_status.json`
- `tools/autonomous_training_cycle.py`
  - cycle 绑定到 `pid + attempt_id + started_command_hash + run_root + final_checkpoint`
  - 新增 pid / attempt / command hash / run_root 漂移失败收口
  - eval / compare 改为“返回码 + JSON 可解析 + status=completed + suite/checkpoint 匹配 + 必要字段存在”
- `tools/compare_models.py`
  - 拒绝 failed / incomplete 的 baseline 或 candidate payload
  - 失败时产出 `status=failed` JSON，而不是伪成功报告
- `src/navrl_dashgo/comparison.py`
  - 新增 `validate_eval_payload()` 与 `validate_comparison_payload()`

### 2.2 benchmark 口径
- `tools/benchmark_train_modes.py`
  - 默认候选集覆盖 `official_e8/e24/e64/e96/e128_camoff`
  - 默认重复次数改为 `3`
  - 输出 `decision_ready / selection_reason / selected_candidate / repeat_index / has_nan`
  - 检测仓库内已有训练时默认拒绝并发 benchmark
  - 排序改为“稳定性优先于吞吐”

### 2.3 map_source / upstream 兼容
- `src/navrl_dashgo/env_adapter.py`
  - `env.map_source` 改成显式白名单
  - 打印解析后的 `map_source + terrain_summary + dynamic_obstacles`
  - `terrain_configuration_error / env_initialization_error` 带 traceback 明确落日志
- `src/dashgo_rl/dashgo_env_navrl_official.py`
  - `navrl_upstream` 改用兼容的 `HfDiscreteObstaclesTerrainCfg`
  - `obstacle_height_mode="choice"`
  - 移除不兼容的 `obstacle_height_probability`
  - `color_scheme` 从 `height` 改为 `none`，绕过当前 trimesh/Isaac 组合对默认 `turbo` colormap 的不兼容

### 2.4 reward / termination / PPO 稳定性
- `src/dashgo_rl/dashgo_env_navrl_official.py`
  - 恢复低权重 `stall / orbit` 抑制项
  - `reach_goal` 改成“距离阈值 + 低速门槛”
- `src/navrl_dashgo/ppo.py`
  - `done = terminated | truncated`
  - 新增 `NonFiniteTrainingStateError`
  - 覆盖 observation / alpha-beta / log_prob / entropy / value / adv / ret / loss / grad 的 finite guard
- `apps/isaac/train_navrl.py`
  - 对 `non_finite_training_state` 显式失败退出
  - 启动时打印 resolved config

### 2.5 formal profile
- 新增 `configs/train/profiles/formal.yaml`
  - `env.num_envs=96`
  - `enable_cameras=false`
  - `env.map_source=dashgo_official`

## 3. Gate 0: 静态与 CPU 校验
- 命令:
  - `PYTHONPATH=/home/gwh/dashgo_navrl_project:/home/gwh/dashgo_navrl_project/src /home/gwh/IsaacLab/_isaac_sim/python.sh -m unittest discover -s tests -v`
- 结果:
  - `18` 项测试全部通过
- 新增/覆盖的关键测试:
  - cycle 拒绝 `failed eval JSON`
  - cycle 拒绝 pid 漂移
  - benchmark 在重复次数不足或 formal 候选不稳定时不输出正式推荐
  - `env.map_source` 白名单
  - upstream terrain 参数与当前 Isaac Lab 支持 API 对齐
  - `truncated -> done` 截断语义
  - finite guard 对 NaN tensor / NaN grad 的硬失败

## 4. Gate 1: GPU 烟测

### 4.1 `dashgo_official` 训练烟测
- 命令:
  - `apps/isaac/train_navrl.py --headless profiles=smoke max_frame_num=1024 env.num_envs=8 env.map_source=dashgo_official enable_cameras=false env.static_obstacles=8 env.dynamic_obstacles=2 save_interval_batches=1 logging.print_interval_batches=1`
- 结果:
  - `run_root=/home/gwh/dashgo_navrl_project/artifacts/runs/smoke_20260402_225456`
  - `final_checkpoint=/home/gwh/dashgo_navrl_project/artifacts/runs/smoke_20260402_225456/checkpoints/checkpoint_final.pt`
  - 无 `failure_reason`
  - 无训练指标 NaN
- 证据:
  - `artifacts/verification/gate1/smoke_dashgo_official.log`

### 4.2 `navrl_upstream` 训练烟测
- 首次失败:
  - `env_initialization_error`
  - 根因: `color_scheme="height"` 触发不支持的 colormap
- 修复后重跑结果:
  - `run_root=/home/gwh/dashgo_navrl_project/artifacts/runs/smoke_20260402_225642`
  - `final_checkpoint=/home/gwh/dashgo_navrl_project/artifacts/runs/smoke_20260402_225642/checkpoints/checkpoint_final.pt`
  - 无 `failure_reason`
  - 无训练指标 NaN
- 证据:
  - `artifacts/verification/gate1/smoke_navrl_upstream.log`

### 4.3 cycle 失败收口烟测
- 命令:
  - `python tools/autonomous_training_cycle.py --profile smoke --budget-hours 0.2 --checkpoint-hours 0.001 --poll-seconds 2 --max-frame-num 1024 env.num_envs=8 env.map_source=dashgo_official enable_cameras=false env.static_obstacles=8 env.dynamic_obstacles=2 save_interval_batches=1 logging.print_interval_batches=1`
- 结果:
  - 训练成功完成并绑定:
    - `attempt_id=20260402_225727`
    - `run_root=/home/gwh/dashgo_navrl_project/artifacts/runs/smoke_20260402_225734`
    - `final_checkpoint=/home/gwh/dashgo_navrl_project/artifacts/runs/smoke_20260402_225734/checkpoints/checkpoint_final.pt`
  - `quick eval` 虽产出 JSON，但 payload 为 `status=failed`
  - cycle 最终状态:
    - `phase=failed`
    - `failure_reason=eval_quick_invalid:["status='failed' 不是 completed"]`
- 结论:
  - 新逻辑已经拒绝“artifact 存在但 status=failed”的假阳性完成
- 证据:
  - `artifacts/autonomous/20260402_225727/status.json`
  - `artifacts/autonomous/20260402_225727/cycle.log`

### 4.4 compare 成功收口验证
- 使用合成且符合 schema 的 `completed` baseline/candidate JSON 运行 `compare_models.py`
- 输出:
  - `artifacts/verification/gate1/compare_success.json`
  - `artifacts/verification/gate1/compare_success.md`
- 结果:
  - 命令退出码 `0`
  - 成功生成 Markdown 报告和对比表

## 5. 当前状态
- 已完成:
  - 修复
  - Gate 0
  - Gate 1
- 未完成:
  - Gate 2 `formal 96 env >= 8,000,000 frames` 稳定性验证
- 当前按用户最新要求:
  - 修复和验证完成后不启动新的正式长训

## 6. 建议审核入口
- 代码重点:
  - `tools/background_train.py`
  - `tools/autonomous_training_cycle.py`
  - `tools/benchmark_train_modes.py`
  - `src/navrl_dashgo/env_adapter.py`
  - `src/navrl_dashgo/ppo.py`
  - `src/dashgo_rl/dashgo_env_navrl_official.py`
- 验证重点:
  - `artifacts/supervisor/pilot/status.json`
  - `artifacts/runs/pilot_20260402_210831/supervisor_status.json`
  - `artifacts/verification/gate1/*.log`
  - `artifacts/autonomous/20260402_225727/status.json`
