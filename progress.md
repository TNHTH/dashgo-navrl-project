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
