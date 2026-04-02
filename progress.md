# Progress Log

## Session: 2026-03-26

### Phase 1: 仓库边界与基础资料固化
- **Status:** complete
- **Started:** 2026-03-26 23:17
- Actions taken:
  - 初始化独立 git 仓库 `/home/gwh/dashgo_navrl_project`
  - 确认 `/home/gwh/NavRL_upstream` 已 clone 成功
  - 确认 Isaac Python、Torch 版本和 `torchrl/tensordict` 缺失状态
  - 固化 DashGo 参数与评测来源
  - 通过 headless 探针确认旧 DashGo 观测是 `{"policy": [N, 246]}`
- Files created/modified:
  - `README.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### Phase 2: DashGo Isaac 环境底座迁移
- **Status:** complete
- Actions taken:
  - 复制 `dashgo_env_v2.py / dashgo_assets.py / dashgo_config.py / project_paths.py`
  - 复制 `dashgo.urdf` 与 `dashgo_driver.yaml`
  - 完成新仓库路径和 ROS2 参数 YAML 解包适配
- Files created/modified:
  - `src/dashgo_rl/*`
  - `configs/robot/dashgo.urdf`
  - `configs/robot/dashgo_driver.yaml`

### Phase 3: TorchRL/PPO 训练闭环
- **Status:** complete
- Actions taken:
  - 实现 `DashgoTensorAdapter`，把旧 `policy[246]` 观测编码为 NavRL 风格 TensorDict
  - 实现 `TorchRLDashgoEnv`，补齐 `observation_spec / action_spec / reward_spec / done_spec`
  - 实现 NavRL 风格 `PPO + BetaActor + GAE + ValueNorm`
  - 修复 actor 少单 agent 维导致的 `batch dimension mismatch`
  - 补齐 `IndependentBeta.deterministic_sample`，去掉初始化警告
  - 用最小 collector 验证 `next(iter(collector)) + algo.update(data)` 可运行
- Files created/modified:
  - `src/navrl_dashgo/env_adapter.py`
  - `src/navrl_dashgo/ppo.py`
  - `src/navrl_dashgo/torchrl_utils.py`
  - `apps/isaac/train_navrl.py`

### Phase 4: 训练值守与评测入口
- **Status:** in_progress
- Actions taken:
  - 新增 `tools/background_train.py`，提供后台训练启动、状态查询、停止
  - 已启动并跑完首轮 full smoke 后台训练
  - 训练 sanity run 成功跑通 `128` frames，并产出 checkpoint
  - 评测入口增加并发保护，避免训练进行中硬起第二个 Isaac 实例
- Files created/modified:
  - `tools/background_train.py`
  - `tools/eval_checkpoint.py`
  - `README.md`

### Phase 5: 奖励函数切换到 NavRL 论文方向
- **Status:** complete
- Actions taken:
  - 将主训练 reward 切换为 `+1 + goal_velocity + static_safety + dynamic_safety - 0.1 * smoothness`
  - 停用旧 DashGo 项目的 `stall / orbit / reverse_escape / 大终点奖 / unsafe_speed` 等启发式训练项
  - 将 `reach_goal` 判据改成纯距离阈值 `0.5m`
  - 用 headless 探针确认激活 reward 项只剩 NavRL 主干与 2 个日志项
  - 用最小 collector 再次验证新 reward 下 `collector + PPO update` 仍能正常运行
  - 确认首轮 smoke 训练完成于奖励切换之前，因此其 checkpoint 不再作为论文方向对比候选
- Files created/modified:
  - `src/dashgo_rl/dashgo_env_v2.py`
  - `findings.md`
  - `progress.md`

### Phase 6: TensorBoard 落盘与 12 小时级长跑启动
- **Status:** in_progress
- Actions taken:
  - 训练入口接入原生 `SummaryWriter`，事件文件落到 `logs/tensorboard/`
  - 将 TensorBoard flush 周期设为 `30s`，并在 checkpoint 保存时显式 flush
  - 修复 Hydra `profiles=pilot/main` 实际回落到 `smoke` 的覆盖链问题
  - 用 `pilot max_frame_num=256` 验证 `8 env / 48 static / 12 dynamic` 档位可正常训练并生成 `.tfevents`
  - 启动正式后台长跑：`pilot + max_frame_num=5376000 + save_interval_batches=200 + logging.print_interval_batches=20`
  - 当前真实运行态：`pid=34187`，`supervisor_status=running`
- Files created/modified:
  - `apps/isaac/train_navrl.py`
  - `src/navrl_dashgo/runtime.py`
  - `tools/background_train.py`
  - `configs/train/train.yaml`
  - `configs/train/profiles/smoke.yaml`
  - `configs/train/profiles/pilot.yaml`
  - `configs/train/profiles/main.yaml`
  - `findings.md`
  - `progress.md`

### Phase 7: 训练结果分析与评测收口
- **Status:** complete
- Actions taken:
  - 解析 `pilot` 长跑训练日志，按 early/mid/late 窗口汇总 `actor_loss / critic_loss / entropy / explained_var`
  - 修复 `eval_worker.py` 对 legacy checkpoint 的严格加载失败问题
  - 跑完 `quick` 标准评测，产出 `pilot_20260327_004917_quick.json`
  - 跑完 `main` 标准评测，产出 `pilot_20260327_004917_main.json`
  - 追加 episode 级分析，确认模型大量停在目标外圈 `0.5m~0.9m`
- Files created/modified:
  - `apps/isaac/eval_worker.py`
  - `apps/isaac/train_navrl.py`
  - `findings.md`
  - `progress.md`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Isaac Python 版本 | `/home/gwh/IsaacLab/_isaac_sim/python.sh -c ...` | 能打印版本 | `3.10.15 / torch 2.5.1+cu118` | ✓ |
| TorchRL 依赖探测 | `find_spec('torchrl')` | 未安装 | `False` | ✓ |
| 旧 DashGo 观测形状探针 | headless 1 env | 获得 reset/compute 形状 | `{"policy": [1, 246]}` | ✓ |
| 新 Env 合同探针 | headless 2 env | `state/lidar/dynamic_obstacle` 形状匹配 | `[2,1,8] / [2,1,72,3] / [2,1,5,10]` | ✓ |
| PPO 初始化探针 | headless 1 env | 动作/value 张量形状正确 | `action=[1,1,2], value=[1,1]` | ✓ |
| 单步 env 闭环探针 | `policy(td) -> env.step(td)` | reward/done/next obs 合同正确 | `reward=[1,1,1], done=[1,1]` | ✓ |
| 最小 collector 探针 | `frames_per_batch=8` | 可取 1 个 batch 并完成一次 `update()` | 通过 | ✓ |
| 训练入口 sanity run | `max_frame_num=128` | 至少完成 1 个 batch 并产出 checkpoint | 成功产出 `checkpoint_128.pt` 和 `checkpoint_final.pt` | ✓ |
| 后台 full smoke 启动 | `background_train.py start --profile smoke` | 有真实训练进程与日志心跳 | `pid=29388`，`batch=0` 已写日志 | ✓ |
| NavRL reward 图谱探针 | headless 1 env | 激活 reward 项仅剩 NavRL 主干 | `navrl_survival / navrl_goal_velocity / navrl_static_safety / navrl_dynamic_safety / navrl_twist_smoothness` | ✓ |
| NavRL reward collector 探针 | `frames_per_batch=8` | 新 reward 下仍可完成一次 `update()` | `actor_loss≈-2.2e-08, critic_loss≈0.4996, entropy≈7.7e-04` | ✓ |
| TensorBoard smoke 落盘探针 | `smoke max_frame_num=256` | 生成 `.tfevents` 文件 | `events.out.tfevents.*` 已落盘 | ✓ |
| TensorBoard pilot 落盘探针 | `pilot max_frame_num=256` | `pilot` 档可训练且生成 `.tfevents` | `num_envs=8` 且 event 文件已落盘 | ✓ |
| Hydra profile 覆盖探针 | `profiles=pilot/main` | 根配置应切到对应档位 | `pilot -> 8 env`, `main -> 24 env` | ✓ |
| pilot 吞吐样本 | `pilot max_frame_num=2048` | 完成多 batch 短跑 | 完成，作为 12h 档位估算参考 | ✓ |
| pilot 长跑启动 | `background_train.py start --profile pilot ...` | 真实后台运行并产生日志心跳 | `pid=34187`，`run_root=/home/gwh/dashgo_navrl_project/artifacts/runs/pilot_20260327_004917` | ✓ |
| legacy checkpoint 评测兼容 | `checkpoint_final.pt` | 应能正常加载并评测 | 通过，缺失 `value_norm / gae` 非推理键但不再阻塞 | ✓ |
| quick 标准评测 | `suite=quick` | 产出成功率/碰撞率等指标 | `success_rate=0.0, collision_rate=0.4167, timeout_rate=0.5833` | ✓ |
| main 标准评测 | `suite=main` | 产出完整 harder-suite 指标 | `success_rate=0.0, collision_rate=0.5208, timeout_rate=0.4792` | ✓ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-03-26 23:24 | Isaac 日志淹没观测输出 | 1 | 改为先写 `/tmp/inspect_dashgo_env_summary.json` 再读取 |
| 2026-03-26 23:51 | `batch dimension mismatch`，actor 写入 `("agents","action")` 少 agent 维 | 1 | 让 `BetaActor` 输出 `[N,1,action_dim]` |
| 2026-03-26 23:58 | 训练进行中并发评测触发第二个 Isaac，worker 抛 `Articulation._data` 异常 | 1 | 在 `tools/eval_checkpoint.py` 增加并发保护，默认阻止并发评测 |
| 2026-03-27 00:27 | 用户要求 reward 改成 NavRL 论文方向，不能继续沿用旧 DashGo 训练策略 | 1 | 重写 active reward 图谱并将首轮 smoke checkpoint 标记为过时样本 |
| 2026-03-27 00:46 | `profiles=pilot` 实际仍跑成 `smoke` | 1 | 修正 `train.yaml` defaults 顺序，并将 profile 配置声明为全局包 |
| 2026-03-27 09:38 | `eval_worker.py` 无法加载当前 long-run checkpoint | 1 | 对 legacy checkpoint 采用兼容加载，只对关键结构不匹配抛错 |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 7，`pilot` 长跑已结束，quick/main 评测已完成 |
| Where am I going? | 基于当前失败模式重新调整 reward/termination/训练配置，再开启下一轮训练 |
| What's the goal? | 在独立仓库内跑通 DashGo x NavRL-style 训练、评测和对比闭环 |
| What have I learned? | 训练与评测不能在这台 8GB 4060 上并发起两个 Isaac 实例；论文方向 reward 需要与旧 DashGo 启发式 reward 严格隔离；Hydra profile 覆盖链必须显式校验；数值稳定不代表导航成功 |
| What have I done? | 完成独立仓库、训练器、后台 supervisor、NavRL reward 切换、TensorBoard 落盘、`pilot` 长跑、评测兼容修复和 quick/main 收口分析 |

## Session: 2026-03-27

### Phase 8: 论文严格对齐与新 4 小时训练
- **Status:** in_progress
- Actions taken:
  - 重新核对上游 `NavRL` 的 PPO 超参和 reward/termination 主干
  - 将 `num_minibatches / entropy_loss_coefficient / clip_ratio` 改回论文同级设置
  - 将 `object_collision` 切回 NavRL 风格 clearance-based 判定
  - 关闭 `DashgoCurriculumCfg`，取消自适应课程学习
  - 保留 DashGo 的动作物理边界，不复用无人机 `2.0 m/s` 动作上限
  - 用 `pilot max_frame_num=256` 短跑验证论文对齐配置可正常启动
  - 启动新 run：`pilot_20260327_121705`，目标帧数 `3,329,024`
- Files created/modified:
  - `configs/train/train.yaml`
  - `src/dashgo_rl/dashgo_env_v2.py`
  - `findings.md`
  - `progress.md`

## Additional Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 论文对齐短跑验证 | `pilot max_frame_num=256` | `pilot` 档、无课程学习、3 个 termination term 正常启动 | 通过，`curriculum=0 term`，`termination=3 terms` | ✓ |
| 新 4 小时后台训练启动 | `background_train.py start --profile pilot max_frame_num=3329024 ...` | 真实运行并生成新 run_root | `pid=64818`，`run_root=/home/gwh/dashgo_navrl_project/artifacts/runs/pilot_20260327_121705` | ✓ |

### Phase 9: RayCaster 可行性验证与低渲染回退
- **Status:** complete
- Actions taken:
  - 主动停止 `pilot_20260327_121705`，为传感器验证释放 GPU
  - 将训练/评测入口改为由配置决定是否开启相机
  - 将环境渲染降到低开销档：关闭 AA / 直射光 / 阴影 / AO，`samples_per_pixel=1`
  - 新增 `apps/isaac/validate_raycast_backend.py`，用真实 DashGo 场景验证 `RayCaster`
  - 先修复验证脚本里手工创建 `RayCaster` 后未初始化的问题
  - 完成多 mesh 和单障碍根路径两类验证，结论是保留相机 fallback
- Files created/modified:
  - `apps/isaac/train_navrl.py`
  - `apps/isaac/eval_worker.py`
  - `apps/isaac/validate_raycast_backend.py`
  - `src/dashgo_rl/dashgo_env_v2.py`
  - `findings.md`
  - `progress.md`

## More Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| RayCaster 多 mesh 验证 | `validate_raycast_backend.py --headless` | 若可行，应支持多个障碍 mesh | 失败，`RayCaster currently only supports one mesh prim. Received: 2` | ✓ |
| RayCaster 单障碍根路径验证 | `validate_raycast_backend.py --headless` | 若可行，应能直接读 `/World/envs/env_0/Obs_In_1` | 失败，`Invalid mesh prim path` | ✓ |
| 低渲染环境启动验证 | `validate_raycast_backend.py --headless` | 低渲染下环境/观测/奖励/终止正常初始化 | 通过，camera fallback 读到 `initial_min_distance≈1.4623m` | ✓ |
| 障碍 prim 类型诊断 | headless prim tree inspect | 找到障碍真实几何类型 | `Obs_* -> Cube/Cylinder`，`ground -> Mesh` | ✓ |

## More Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-03-27 12:32 | 手工创建 `RayCaster` 后直接 `update()` 报 `'_timestamp'` 缺失 | 1 | 验证脚本中显式调用 `_initialize_callback(None)` 完成初始化 |
| 2026-03-27 12:33 | `RayCaster` 多 mesh 验证失败 | 1 | 确认本机 Isaac Lab 2.0.2 仍只支持单个静态 mesh，放弃主链替换 |
| 2026-03-27 12:33 | `RayCaster` 单障碍根路径报 `Invalid mesh prim path` | 1 | 确认当前场景障碍不是 `Mesh/Plane` 静态 mesh 目标，保留相机 fallback |

### Phase 10: 官方 NavRL 路线环境重构
- **Status:** complete
- Actions taken:
  - 新增 `src/dashgo_rl/dashgo_env_navrl_official.py`
  - 将静态障碍改为 `HfDiscreteObstaclesTerrainCfg` 生成到 `/World/ground`
  - 将 `RayCaster` 改为只打 `/World/ground` 的真实 LiDAR 传感器
  - 使用 `RigidObjectCollectionCfg` 重建动态障碍真值状态分支
  - 保留 DashGo 差速动作映射、底盘参数和 246 维 policy 观测合同
  - 将 `src/navrl_dashgo/env_adapter.py` 和 `apps/isaac/eval_worker.py` 切到新环境
  - 将 `configs/train/train.yaml` 默认相机开关改为 `false`
  - 完成 headless reset/step 探针和最小 collector 探针，并把结果写入 `artifacts/diagnostics/`
- Files created/modified:
  - `src/dashgo_rl/dashgo_env_navrl_official.py`
  - `src/navrl_dashgo/env_adapter.py`
  - `apps/isaac/eval_worker.py`
  - `configs/train/train.yaml`
  - `task_plan.md`

## Session: 2026-04-02

### Phase 16: correctness blocker 修复与 Gate 0/1 验证
- **Status:** in_progress
- Actions taken:
  - 停止并废弃 `pilot_20260402_210831`，把 `abandoned/current_run_invalid=true` 同步写入 supervisor 状态和 run 级记录
  - 为 `background_train.py` 增加 `attempt_id / started_command_hash / latest_final_checkpoint / failure_reason / abandon` 语义
  - 为 `autonomous_training_cycle.py` 增加 `pid / attempt / command_hash / run_root / final_checkpoint` 强绑定与漂移失败收口
  - 为 `compare_models.py` 与 `comparison.py` 增加 eval/comparison payload 校验，拒绝 failed payload 假阳性
  - 重写 `benchmark_train_modes.py` 的正式候选集、重复统计、决策门槛与并发保护
  - 修复 `navrl_upstream` 的 terrain API 配置，并追加 `color_scheme=\"none\"` 兼容补丁
  - 修复 reward/termination 与 PPO 的 `truncated -> done`、finite guard、non-finite fail-fast
  - 新增 `formal` profile，但按用户最新要求暂不启动正式长训
  - 跑通 Gate 0 全量单测与 Gate 1 双地图训练烟测
  - 运行最小自治 cycle 烟测，确认 failed eval 不会再把 cycle 写成 completed
- Files created/modified:
  - `tools/background_train.py`
  - `tools/autonomous_training_cycle.py`
  - `tools/benchmark_train_modes.py`
  - `tools/compare_models.py`
  - `src/navrl_dashgo/comparison.py`
  - `src/navrl_dashgo/env_adapter.py`
  - `src/navrl_dashgo/ppo.py`
  - `src/dashgo_rl/dashgo_env_navrl_official.py`
  - `apps/isaac/train_navrl.py`
  - `configs/train/profiles/formal.yaml`
  - `tests/test_autonomous_training_cycle.py`
  - `tests/test_benchmark_train_modes.py`
  - `tests/test_comparison.py`
  - `tests/test_env_and_ppo_guards.py`
  - `docs/formal_repair_validation_2026-04-02_23-08.md`
  - `task_plan.md`
  - `progress.md`

## Additional Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Gate 0 全量单测 | `unittest discover -s tests -v` | `cycle / benchmark / map_source / truncated / finite guard` 全通过 | `18` 项通过 | ✓ |
| `dashgo_official` 烟测 | `smoke max_frame_num=1024 env.num_envs=8` | 训练可启动、落 checkpoint、无 NaN | `checkpoint_final.pt` 已生成，日志无指标 NaN | ✓ |
| `navrl_upstream` 首次烟测 | 同上，`map_source=navrl_upstream` | 应稳定启动 | 首次因 `color_scheme=height` 触发 colormap 错误失败 | ✓ |
| `navrl_upstream` 修复后烟测 | 修复后同命令重跑 | 稳定启动、落 checkpoint、无 NaN | `checkpoint_final.pt` 已生成，日志无指标 NaN | ✓ |
| cycle 失败收口烟测 | `autonomous_training_cycle.py --profile smoke ...` | failed eval 不应写成 completed | 最终 `phase=failed`, `failure_reason=eval_quick_invalid[...]` | ✓ |
| compare 成功收口验证 | 合成 completed baseline/candidate JSON | compare 成功输出 JSON/MD | 退出码 `0`，报告已生成 | ✓ |

## Additional Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-04-02 22:55 | `navrl_upstream` 初始化时报 `Included color maps are ...` | 1 | 将 upstream terrain `color_scheme` 从 `height` 改为 `none` |
| 2026-04-02 23:04 | smoke cycle 的 `quick eval` 返回 `status=failed` | 1 | 新 cycle 正确拒绝该 artifact，并以 `eval_quick_invalid` 收口失败 |
  - `findings.md`
  - `progress.md`

## Latest Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 官方路线环境探针 | headless 2 env | `policy[246] + lidar[72] + token[5,10] + step` 全部正常 | 通过，详见 `artifacts/diagnostics/navrl_official_probe_20260327.json` | ✓ |
| 官方路线最小 collector 探针 | `SyncDataCollector total_frames=64` | `collector -> PPO.update()` 正常运行 | 通过，详见 `artifacts/diagnostics/navrl_official_collector_probe_20260327.json` | ✓ |

## Latest Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-03-27 13:20 | 训练入口短跑日志被 Isaac 启动噪音淹没，难以直接确认 collector 是否执行 | 1 | 改为单独跑 headless probe 和最小 collector probe，并把结构化结果写入 `artifacts/diagnostics/` |

### Phase 11: 训练入口修复与 4 小时基线校准
- **Status:** complete
- Actions taken:
  - 修复 `apps/isaac/train_navrl.py` 的 Hydra override 清理逻辑，避免把非 argparse 参数留给 Kit
  - 修复 `AppLauncher(args_cli)` 改写 `Namespace` 后再次读取 `args_cli.enable_cameras` 导致的静默秒退
  - 为 `tools/background_train.py` 增加每次启动的 supervisor 边界标记，避免旧日志污染新状态判断
  - 用真实训练入口完成 `pilot / 8 env / 8192 frames` 的吞吐基准
  - 确认真实入口可直接产出 `run_root / tensorboard_root / checkpoint / final_checkpoint`
- Files created/modified:
  - `apps/isaac/train_navrl.py`
  - `tools/background_train.py`
  - `findings.md`
  - `progress.md`

## Latest Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 真实训练入口回归 | `train_navrl.py --headless profiles=pilot max_frame_num=256 env.num_envs=2 ...` | 输出 run_root、batch、checkpoint、final_checkpoint | 通过，生成 `pilot_20260327_133748` | ✓ |
| 吞吐基准 | `train_navrl.py --headless profiles=pilot max_frame_num=8192 enable_cameras=false` | 跑完完整训练并得到真实耗时 | 通过，`elapsed_seconds=33.62`，约 `243.7 frames/s` | ✓ |

## Latest Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-03-27 13:36 | supervisor 启动后几秒变 `blocked_exited`，但没有 traceback | 1 | 给真实入口增加阶段诊断，定位到 `AppLauncher` 改写 `Namespace` 后读取 `args_cli.enable_cameras` 导致 `AttributeError` |

### Phase 12: 四小时正式训练已启动
- **Status:** in_progress
- Actions taken:
  - 依据 `pilot / 8 env / 8192 frames -> 33.62s` 的基线，把 4 小时训练预算设为 `max_frame_num=3600000`
  - 通过 `tools/background_train.py` 启动后台训练
  - 核对 supervisor 状态、进程表和训练日志三方一致
- Runtime facts:
  - `started_at=2026-03-27T13:40:25+08:00`
  - `pid=75719`
  - `run_root=/home/gwh/dashgo_navrl_project/artifacts/runs/pilot_20260327_134032`
  - `tensorboard_root=/home/gwh/dashgo_navrl_project/artifacts/runs/pilot_20260327_134032/logs/tensorboard`
  - `estimated_hours≈4.10`
  - `frames_per_batch=256`
- Verification snapshot:
  - supervisor: `running`
  - log: 已出现 `run_root / tensorboard_root / batch=0 / checkpoint_256.pt`
  - process table: `python.sh train_navrl.py ... max_frame_num=3600000 ...` 仍在运行

### Phase 13: 暂停训练并执行正式评测
- **Status:** complete
- Actions taken:
  - 按用户要求暂停当前 `pilot_20260327_134032` 训练
  - 选取最近一次已保存的 `checkpoint_2765056.pt` 作为正式评测输入
  - 修复 `apps/isaac/eval_worker.py` 对空 `value_norm` state 的兼容问题
  - 串行跑完 `quick` 与 `main` 两套正式评测
- Files created/modified:
  - `apps/isaac/eval_worker.py`
  - `artifacts/eval/pilot_20260327_134032_quick.json`
  - `artifacts/eval/pilot_20260327_134032_main.json`
  - `findings.md`
  - `progress.md`
- Evaluation summary:
  - `quick`: `success_rate=0.0`, `collision_rate=0.9167`, `progress_stall_rate=0.9167`, `score=-60.70`
  - `main`: `success_rate=0.0`, `collision_rate=0.8542`, `progress_stall_rate=0.6875`, `score=-56.81`

## Session: 2026-04-02

### Phase 14: 正式基线对比与阶段封板
- **Status:** complete
- Actions taken:
  - 新增 `src/navrl_dashgo/comparison.py`，把指标对比、终止原因统计、失败模式提炼和 Markdown 报告渲染抽成可复用逻辑
  - 增强 `tools/compare_models.py`，支持三种输入方式：
    - 直接传 baseline/candidate checkpoint
    - 读取已有 baseline/candidate eval JSON
    - 默认从旧仓库在线 manifest 自动解析 GeoNav 基线 checkpoint
  - 新增 `tests/test_comparison.py`，对对比逻辑做纯 Python 单测
  - 跑出旧仓库在线基线 `quick/main` 正式评测：
    - `artifacts/eval/baseline_model_883_quick.json`
    - `artifacts/eval/baseline_model_883_main.json`
  - 产出 quick/main 正式对比 JSON 与 Markdown：
    - `artifacts/eval/compare_quick_online_vs_navrl.json`
    - `artifacts/eval/compare_quick_online_vs_navrl.md`
    - `artifacts/eval/compare_main_online_vs_navrl.json`
    - `artifacts/eval/compare_main_online_vs_navrl.md`
  - 补齐阶段文档与边界文档：
    - `docs/phase1_formal_comparison_2026-04-02.md`
    - `docs/phase2_real_robot_boundary_2026-04-02.md`
  - 更新 `README.md / task_plan.md / findings.md / progress.md`，把第一阶段完成定义和第二阶段边界固化
- Files created/modified:
  - `src/navrl_dashgo/comparison.py`
  - `tools/compare_models.py`
  - `tests/test_comparison.py`
  - `README.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `docs/phase1_formal_comparison_2026-04-02.md`
  - `docs/phase2_real_robot_boundary_2026-04-02.md`

## Latest Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 对比逻辑单测 | `PYTHONPATH=src python3 -m unittest discover -s tests -v` | 指标对比、终止统计、Markdown 渲染通过 | `4 tests OK` | ✓ |
| 旧在线基线 quick | `model_883.pt` + `suite=quick` | 产出正式基线 JSON | `success_rate=0.0833, collision_rate=0.0, timeout_rate=0.9167, score=-65.34` | ✓ |
| 旧在线基线 main | `model_883.pt` + `suite=main` | 产出正式基线 JSON | `success_rate=0.1042, collision_rate=0.0, timeout_rate=0.8958, score=-50.42` | ✓ |
| quick 正式对比 | baseline online vs `pilot_20260327_134032` | 产出 JSON + Markdown 报告 | 候选未形成成功 episode，但 `score` 高于基线 `+4.6349` | ✓ |
| main 正式对比 | baseline online vs `pilot_20260327_134032` | 产出 JSON + Markdown 报告 | 候选整体劣于基线，`score` 低于基线 `-6.3903` | ✓ |

### Phase 15: 本机模式摸索与 17 小时自治训练
- **Status:** in_progress
- Actions taken:
  - 在 `src/dashgo_rl/dashgo_env_navrl_official.py` 中落地最小算法修正：
    - `goal_termination_threshold=0.6`
    - 新增 `navrl_waypoint_velocity`
    - 下调 `navrl_survival` 权重
  - 在 `configs/train/train.yaml` 与 `src/navrl_dashgo/env_adapter.py` 中增加 `env.map_source`
  - 新增 `tools/benchmark_train_modes.py`，批量跑本机训练模式短跑 benchmark
  - 新增 `tools/autonomous_training_cycle.py`，把“训练 -> 评测 -> 对比”串成可后台自运行的长期周期
  - 修复自治链两个问题：
    - 评测失败时仍要保留 JSON，不应仅凭返回码中断
    - 训练退出后立刻拉评测会偶发 Isaac `bad_optional_access`，增加冷却与重试
  - 通过 `1024 frame` 短周期完成一次自治闭环回归
  - 正式启动 `17` 小时自治训练周期
- Files created/modified:
  - `src/dashgo_rl/dashgo_env_navrl_official.py`
  - `src/navrl_dashgo/env_adapter.py`
  - `configs/train/train.yaml`
  - `tools/benchmark_train_modes.py`
  - `tools/autonomous_training_cycle.py`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- Runtime facts:
  - 吞吐摸索代表结果：
    - `official_e24_camoff -> 445.81 fps`
    - `official_e64_camoff -> 616.99 fps`
    - `official_e96_camoff -> 669.73 fps`
    - `official_e128_camoff -> 725.74 fps`（仅短吞吐探针）
    - `official_e8_camon -> 217.08 fps`
  - 正式长跑选择：
    - `env.num_envs=96`
    - `env.map_source=dashgo_official`
    - `enable_cameras=false`
    - `max_frame_num=165888000`
  - 当前自治周期：
    - `cycle_root=/home/gwh/dashgo_navrl_project/artifacts/autonomous/20260402_210824`
    - `run_root=/home/gwh/dashgo_navrl_project/artifacts/runs/pilot_20260402_210831`
    - `six_hour_mark_at=2026-04-03T03:08:24+08:00`
    - `expected_end_at=2026-04-03T14:08:24+08:00`
- Verification snapshot:
  - `python3 -m py_compile` 已覆盖新增/修改脚本
  - `benchmark_train_modes.py` 已实际跑出 benchmark JSON
  - `autonomous_training_cycle.py` 已通过短周期完整回归
  - 当前 `ps / status.json / background_train.py status` 三方一致，正式训练进程正在运行

## Latest Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-04-02 17:17 | `compare_models.py` 只能吃 checkpoint，不能直接复用现有 eval JSON | 1 | 新增 `--baseline-json / --candidate-json / --report-out`，并支持默认在线 manifest 基线 |
| 2026-04-02 17:20 | 第一阶段缺少“正式基线口径”，无法避免挑模型争议 | 1 | 固定旧仓库在线 TorchScript manifest 指向的 `model_883.pt` 为唯一 GeoNav 基线 |
| 2026-04-02 21:56 | 当前正式长跑 `frames=6147072` 起持续 `NaN` | 1 | 暂未修复；当前 run 已降级为 debug 样本，下一阶段先修 correctness blocker |
| 2026-04-02 22:10 | `autonomous_training_cycle.py` 只按 artifact 存在判断评测可用 | 1 | 暂未修复；已在复盘中明确要求加入 payload-level success guard |
| 2026-04-02 22:14 | `PPO/GAE` 只使用 `terminated`，忽略 `truncated` | 1 | 暂未修复；已升级为下一轮正式训练前的 correctness blocker |
| 2026-04-02 22:18 | `navrl_upstream` 地图模式与当前 Isaac Lab API 高概率不兼容 | 1 | 暂未修复；已要求至少改成显式报错或先禁用 |

### Phase 16: 长跑复盘与 correctness blocker 收口前准备
- **Status:** in_progress
- Actions taken:
  - 交叉核对外部静态审查 findings、当前 `train.log`、`status.json` 与 benchmark JSON
  - 确认当前正式长跑在 `frames=6147072` 处进入持续 NaN，已不再适合作为正式结果来源
  - 明确 checkpoint 处置边界：
    - `checkpoint_6147072.pt` 及之后视为无效
    - `checkpoint_4611072.pt` 及之前降级为 pre-fix debug/warm-start 样本
  - 编写正式复盘文档：
    - `/home/gwh/文档/Obsidian Vault/03_项目记录/DashGo/DashGo_NavRL_Longrun_Retro_2026-04-02_22-28.md`
  - 更新 `findings.md / progress.md / task_plan.md`，把阶段判断从“等待收数”改为“先修 correctness blocker”
- Runtime facts:
  - 当前 live run 仍在 supervisor 下运行，但其训练有效性已失效
  - 当前最强证据不是最终分数，而是 NaN 起点与未修 correctness 缺陷
- Verification snapshot:
  - `train.log` 已实证 `frames=6147072` 起连续 NaN
  - `status.json` 已确认当前 run 口径为 `pilot + env.num_envs=96 + dashgo_official + cameras off`
  - 复盘文档已落盘到 `docs/`

### Phase 17: TorchRL 与评测生命周期修复、记录与复盘
- **Status:** complete
- Actions taken:
  - 修复 `src/navrl_dashgo/env_adapter.py` 的 `_reset()`，避免 TorchRL partial reset 被错误放大成整批 `base_env.reset()`
  - 为训练侧增加 `_collapse_reset_mask / _resolve_reset_env_ids / _envs_already_autoreset / _current_raw_obs`
  - 在 `apps/isaac/eval_worker.py` 中新增 `step_env_without_auto_reset()`，让评测循环先读取终态指标，再显式 reset done env
  - 在 `apps/isaac/eval_worker.py` 中新增 `reset_done_envs_for_next_episode()`，确保 recycle env 在重新布场景后立即刷新观测
  - 在 `src/dashgo_rl/project_paths.py` 中新增统一的 `resolve_isaac_python()/ISAAC_PYTHON`
  - 将 `tools/background_train.py`、`tools/benchmark_train_modes.py`、`tools/eval_checkpoint.py` 统一切到项目级 Isaac Python 解析
  - 为生命周期问题补齐单测：
    - `tests/test_env_and_ppo_guards.py`
    - `tests/test_eval_worker_flow.py`
  - 跑通 smoke 训练与 quick eval smoke，并将本轮修复、验证、复盘写入项目台账与 Obsidian
- Files created/modified:
  - `src/navrl_dashgo/env_adapter.py`
  - `apps/isaac/eval_worker.py`
  - `src/dashgo_rl/project_paths.py`
  - `tools/background_train.py`
  - `tools/benchmark_train_modes.py`
  - `tools/eval_checkpoint.py`
  - `tests/test_env_and_ppo_guards.py`
  - `tests/test_eval_worker_flow.py`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- Verification snapshot:
  - `PYTHONPATH=src:. /home/gwh/IsaacLab/_isaac_sim/python.sh -m unittest tests.test_env_and_ppo_guards tests.test_eval_worker_flow -q`
    - `11` 项通过
  - `PYTHONPATH=src:. /home/gwh/IsaacLab/_isaac_sim/python.sh -m unittest discover -s tests -q`
    - `28` 项通过
  - `/home/gwh/IsaacLab/_isaac_sim/python.sh -m py_compile src/navrl_dashgo/env_adapter.py apps/isaac/eval_worker.py tools/eval_checkpoint.py tools/background_train.py tools/benchmark_train_modes.py tests/test_env_and_ppo_guards.py tests/test_eval_worker_flow.py`
    - 通过
  - smoke 训练：
    - `/home/gwh/IsaacLab/_isaac_sim/python.sh apps/isaac/train_navrl.py --headless profiles=smoke max_frame_num=64 env.num_envs=2 save_interval_batches=9999 logging.print_interval_batches=1`
    - 成功产出 `/home/gwh/dashgo_navrl_project/artifacts/runs/smoke_20260402_235946/checkpoints/checkpoint_final.pt`
  - quick eval smoke：
    - `/home/gwh/IsaacLab/_isaac_sim/python.sh tools/eval_checkpoint.py --checkpoint /home/gwh/dashgo_navrl_project/artifacts/runs/smoke_20260402_235946/checkpoints/checkpoint_final.pt --suite quick --requested-episodes 2 --json-out /tmp/dashgo_navrl_eval_smoke_20260402.json`
    - 成功产出结构化 JSON；退出码来自 `behavior_gate_veto`，不是 worker 生命周期故障

## Latest Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| partial reset 回归 | `tests.test_env_and_ppo_guards` | 只 reset 目标 env，已 auto-reset 时复用当前 obs | `7 tests OK` | ✓ |
| eval worker 生命周期回归 | `tests.test_eval_worker_flow` | 保留终态统计时机，并在 recycle 后刷新 obs | `4 tests OK` | ✓ |
| 全量单测回归 | `unittest discover -s tests -q` | 本轮修复不破坏既有单测 | `28 tests OK` | ✓ |
| smoke 训练 | `profiles=smoke max_frame_num=64 env.num_envs=2` | 完成 1 batch 并产出 final checkpoint | `checkpoint_final.pt` 已生成 | ✓ |
| quick eval smoke | `suite=quick requested_episodes=2` | 评测链路可完整产出 JSON | 已产出 `/tmp/dashgo_navrl_eval_smoke_20260402.json`；失败原因是行为 gate 而非执行崩溃 | ✓ |

## Latest Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-04-02 | `TorchRLDashgoEnv._reset()` 忽略 TorchRL partial reset，单个 done env 触发整批二次 reset | 1 | 已修复；按 `_reset` 掩码只处理目标 env，且识别 Isaac Lab auto-reset 后优先复用当前 obs |
| 2026-04-02 | `eval_worker.py` 在 `step()` 之后统计终态，读到 reset 后状态 | 1 | 已修复；引入显式 `step_env_without_auto_reset()` 并把 reset 延后到统计完成之后 |
| 2026-04-03 | quick eval smoke 返回 `status=failed` | 1 | 已确认是 `behavior_gate_veto`，不是生命周期链路再次崩溃 |

### Phase 18: 12 小时 formal 连续训练启动
- **Status:** in_progress
- Actions taken:
  - 核对 `formal / main / smoke / pilot` 四个 supervisor 状态，确认：
    - `formal=idle`
    - `main=idle`
    - `smoke=completed`
    - `pilot=abandoned`
  - 核对当前没有活动中的 `/home/gwh/dashgo_navrl_project/apps/isaac/train_navrl.py` 进程
  - 复核 `formal` 配置：
    - `env.num_envs=96`
    - `env.map_source=dashgo_official`
    - `enable_cameras=false`
    - `training_frame_num=32`
  - 按最近 benchmark `669.73 fps` 计算 12 小时预算，并对齐到 `frames_per_batch=3072`
  - 建立 Obsidian 正式执行记录：
    - `/home/gwh/文档/Obsidian Vault/03_项目记录/DashGo/dashgo_navrl_project_12h_formal_training_2026-04-03.md`
  - 记录用户新增要求：本仓库代码后续需要上传 GitHub
  - 启动后台训练：
    - `python3 tools/background_train.py start --profile formal max_frame_num=28932096`
- Runtime facts:
  - `started_at=2026-04-03T00:11:49+08:00`
  - `expected_end_at=2026-04-03 12:10:22 +0800`
  - `pid=40361`
  - `attempt_id=20260403_001149`
  - `started_command_hash=b9f64135c9e9989d5b7169ef499c57246d1a3c4d03aeacf44b8785a0803f3dcc`
  - `run_root=/home/gwh/dashgo_navrl_project/artifacts/runs/formal_20260403_001157`
  - `tensorboard_root=/home/gwh/dashgo_navrl_project/artifacts/runs/formal_20260403_001157/logs/tensorboard`
  - `frames_per_batch=3072`
  - `total_frames=28932096`
- Verification snapshot:
  - supervisor 当前为 `running`
  - 日志已出现：
    - `run_root=/home/gwh/dashgo_navrl_project/artifacts/runs/formal_20260403_001157`
    - `tensorboard_root=/home/gwh/dashgo_navrl_project/artifacts/runs/formal_20260403_001157/logs/tensorboard`
    - `batch=0 frames=3072 actor_loss=0.0002 critic_loss=0.4831 entropy=0.0002 explained_var=0.0591`
    - `checkpoint=/home/gwh/dashgo_navrl_project/artifacts/runs/formal_20260403_001157/checkpoints/checkpoint_3072.pt`
