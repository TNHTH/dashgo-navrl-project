# DashGo_NavRL_Longrun_Retro_2026-04-02_22-28

创建时间: 2026-04-02 22:28 +0800

## Objective
- 复盘 `/home/gwh/dashgo_navrl_project` 从独立仓库建立到 `17` 小时自治长跑翻案的完整过程。
- 记录关键操作、原始报错、根因判断、修复动作、验证证据与当前剩余风险。
- 把本轮经验沉淀成下一轮训练前必须满足的 guard，而不是继续依赖聊天上下文回忆。

## Environment
- 机器: 本机 Ubuntu 工作站
- GPU: RTX 4060 Laptop 8GB
- Isaac Python: `/home/gwh/IsaacLab/_isaac_sim/python.sh`
- 参考仓库:
  - 新实验仓库: `/home/gwh/dashgo_navrl_project`
  - 旧基线仓库: `/home/gwh/dashgo_rl_project`
  - 上游只读参考: `/home/gwh/NavRL_upstream`

## Scope
- 时间范围: 2026-03-26 到 2026-04-02
- 复盘对象:
  - 独立仓库搭建
  - DashGo x NavRL-style 训练栈落地
  - official-route 环境重构
  - quick/main 统一评测与基线对比
  - 本机训练模式摸索
  - `17` 小时自治长跑
  - 审查翻案与后续修复优先级

## Timeline
### 2026-03-26 到 2026-03-27: 独立仓库与训练闭环建立
- 操作:
  - 新建 `/home/gwh/dashgo_navrl_project`
  - clone `/home/gwh/NavRL_upstream` 作为只读参考
  - 迁入 DashGo Isaac 环境底座、URDF、参数文件
  - 定义 TensorDict 观测合同，切分旧 `246` 维观测为 `state/lidar/dynamic_obstacle`
  - 实现 `TorchRL + PPO + BetaActor + GAE + ValueNorm`
  - 打通 `train_navrl.py`、`eval_worker.py`、`compare_models.py`
- 结果:
  - 形成独立于旧仓库的训练、评测、对比最小闭环
  - 新环境 reset/step、collector、最小训练入口均可运行

### 2026-03-27: 训练入口与评测兼容修复
- 操作:
  - 修复 Hydra override 与 Kit 参数链
  - 修复 `AppLauncher(args_cli)` 改写 `Namespace` 后再次读取 `args_cli.enable_cameras` 的静默秒退
  - 修复 legacy checkpoint 在评测侧的兼容加载
- 典型命令:
```bash
/home/gwh/IsaacLab/_isaac_sim/python.sh \
  /home/gwh/dashgo_navrl_project/apps/isaac/train_navrl.py \
  --headless profiles=pilot max_frame_num=256 env.num_envs=2 enable_cameras=false
```
- 结果:
  - 训练入口能稳定输出 `run_root / checkpoint / final_checkpoint`
  - 评测入口能读取旧格式 checkpoint

### 2026-03-27 到 2026-04-02: official-route 与第一阶段正式对比
- 操作:
  - 新建 `src/dashgo_rl/dashgo_env_navrl_official.py`
  - 静态障碍切到 terrain mesh
  - 静态感知切到 `RayCaster -> /World/ground`
  - 动态障碍单独维护状态并输出 token
  - 保留 DashGo 差速动作语义，不照搬上游全向动作
  - 固定旧仓库在线 `model_883.pt` 为唯一 GeoNav 基线
- 结果:
  - 第一阶段正式 quick/main 对比完成
  - 结论是 NavRL-style 闭环已打通，但效果尚未超过 GeoNav 主线

### 2026-04-02: 本机模式摸索与自治长跑
- 操作:
  - 新增 `env.map_source`
  - 对 official-route 做最小 reward/termination 修正
  - 新增 `tools/benchmark_train_modes.py`
  - 新增 `tools/autonomous_training_cycle.py`
  - 完成自治短周期回归
  - 启动正式 `17` 小时自治长跑
- 正式长跑实际命令:
```bash
/home/gwh/IsaacLab/_isaac_sim/python.sh \
  /home/gwh/dashgo_navrl_project/apps/isaac/train_navrl.py \
  --headless \
  profiles=pilot \
  max_frame_num=165888000 \
  env.num_envs=96 \
  env.map_source=dashgo_official \
  enable_cameras=false \
  save_interval_batches=500 \
  logging.print_interval_batches=200
```
- 结果:
  - 吞吐上确认 `camoff + 高 num_envs` 明显优于旧相机链
  - 但后续审查表明 correctness blocker 尚未收敛

### 2026-04-02 晚间: 审查翻案
- 输入:
  - 代码静态审查 findings
  - live `train.log`
  - `status.json`
  - benchmark JSON
- 翻案结论:
  - 当前 `96 env + camoff` 更接近“硬件最优模式”，不是“正式可信训练模式”
  - 当前 `17` 小时长跑应降级为 debug 样本，不应再作为正式结果来源

## Operations, Errors, Root Cause, Resolution
### 1. 训练入口静默秒退
- Trigger:
  - `background_train.py start ...` 拉起后几秒退出
  - 日志只有 Isaac 启动信息，没有 `run_root / batch / final_checkpoint`
- Original Error:
  - 表面 exit code 为 `0`
  - 实际根因是 `AttributeError` 被 `finally -> simulation_app.close()` 掩盖
- Root Cause:
  - `AppLauncher(args_cli)` 改写传入的 `argparse.Namespace`
  - 后续代码仍读取已失真的 `args_cli.enable_cameras`
- Corrective Action:
  - 在 `AppLauncher` 前解析并固化 `resolved_enable_cameras`
  - 剥离 Hydra overrides，避免 Kit 误读
- Verification:
  - 真实训练入口成功产生 `run_root / checkpoint / final_checkpoint`
- Status:
  - 已修复

### 2. 评测与训练并发不稳定
- Trigger:
  - 训练进行时并发拉第二个 Isaac 实例做评测
- Original Error:
  - `eval_worker.py` 抛底层对象初始化异常
- Root Cause:
  - 8GB 显存下双 Isaac 实例并发不稳定
- Corrective Action:
  - 评测入口默认阻止与训练并发
  - 训练与评测改为串行
- Verification:
  - 正式评测可在暂停训练后稳定跑完
- Status:
  - 已修复

### 3. legacy checkpoint 无法评测
- Trigger:
  - 旧格式 checkpoint 缺少 `value_norm / gae` 相关 buffer
- Original Error:
  - `strict=True` 加载失败
- Root Cause:
  - 推理不需要的键被评测入口当成强依赖
- Corrective Action:
  - 评测侧对 `value_norm` 使用兼容加载
  - 非关键缺失只记录 note，不阻断评测
- Verification:
  - 正式 quick/main 已可读取 legacy checkpoint
- Status:
  - 已修复

### 4. official-route 从相机 fallback 切到 RayCaster
- Trigger:
  - 旧 `dashgo_env_v2.py` 的四向深度相机链吞吐与显存成本过高
- Root Cause:
  - 旧链路需要相机渲染与历史拼接
- Corrective Action:
  - 新建 official-route:
    - 静态障碍进入 terrain mesh
    - `RayCaster` 只打 `/World/ground`
    - 动态障碍由真值 token 分支提供
- Verification:
  - reset/step probe、collector probe、最小训练入口通过
- Status:
  - 已修复

### 5. `navrl_upstream` 地图模式静默退出
- Trigger:
  - `env.map_source=navrl_upstream`
- Original Error:
  - 进程在 `Simulation App Startup Complete` 后静默退出
  - 未进入 `run_root` 打印阶段
- Root Cause Judgment:
  - 本机 Isaac Lab API 只接受 `"choice"/"fixed"` 与 `tuple[float, float]`
  - 当前 upstream-style 配置使用了 `obstacle_height_mode="range"`、列表型 `obstacle_height_range` 与 `obstacle_height_probability`
  - 属于配置层面的确定性不兼容高风险项
- Corrective Action:
  - 已打通入口，但未完成修复
- Verification:
  - 短 benchmark 中 `upstream_e16_camoff` 无 `run_root/final_checkpoint`
- Status:
  - 未修复

### 6. benchmark 默认决策空间与正式口径漂移
- Trigger:
  - `benchmark_train_modes.py` 默认候选仅到 `24 env`
- Root Cause:
  - 后续手工追加了 `32/48/64/96/128` 探针，但脚本默认集合未同步更新
- Corrective Action:
  - 本轮尚未修复，只在文档中补充事实
- Impact:
  - 未来复现实验时，单跑默认脚本会得出过时结论
- Status:
  - 未修复

### 7. 自治链把“有 JSON”误当“评测成功”
- Trigger:
  - `autonomous_training_cycle.py` 只要求评测产出 artifact
- Original Error:
  - `eval_checkpoint.py` 即使失败也可能落地 `status=failed` 的 JSON
  - compare 逻辑也不会拒绝 failed payload
- Root Cause:
  - 成功判据被错误地从“进程成功 + payload 成功”降格成“文件存在”
- Corrective Action:
  - 本轮只做了“不要仅凭非零返回码立即中断”的局部修复
  - 尚未补上 payload-level success guard
- Verification:
  - 静态审查已确认假阳性完成风险存在
- Status:
  - 未修复

### 8. PPO/GAE 忽略 `truncated`
- Trigger:
  - 环境已经把 `done / terminated / truncated` 分开返回
- Original Error:
  - PPO/GAE 只用 `terminated` 做 bootstrap 截断
- Root Cause:
  - timeout episode 被跨 episode bootstrap，回报估计漂移
- Corrective Action:
  - 尚未修复
- Impact:
  - 这是训练正确性 blocker，不是普通调参项
- Status:
  - 未修复

### 9. `17` 小时长跑在 `6147072` frames 后持续 NaN
- Trigger:
  - 当前正式长跑 `pilot_20260402_210831`
- Original Error:
  - `frames=6147072` 开始 `actor_loss=nan / critic_loss=nan / entropy=nan / explained_var=nan`
- Evidence:
```text
[DashGo-NavRL] batch=2000 frames=6147072 actor_loss=nan critic_loss=nan entropy=nan explained_var=nan
[DashGo-NavRL] batch=2200 frames=6761472 actor_loss=nan critic_loss=nan entropy=nan explained_var=nan
```
- Root Cause Judgment:
  - 尚未定位到单一源头
  - 但当前已知 correctness blocker 包括:
    - `truncated` 没进 GAE
    - reward 存在开阔区局部最优风险
    - 成功终止没有停稳门槛
- Corrective Action:
  - 尚未执行停训和修复
- Checkpoint Policy:
  - `checkpoint_6147072.pt` 及之后应视为无效
  - `checkpoint_4611072.pt` 及之前可保留作 pre-fix debug/warm-start 样本
- Status:
  - 未修复

## Worked
- 独立仓库边界清晰，没有把 NavRL 复现继续混入旧仓库。
- DashGo 差速动作语义、参数来源和评测协议已固化。
- official-route 已证明不依赖相机 fallback 也能跑训练闭环。
- quick/main 正式评测与旧 GeoNav 基线对比链已经成型。
- 本机 benchmark 证明 `enable_cameras=false` 对 8GB 4060 是正确方向。

## Waste
- 在 `truncated/GAE` 和自治链成功判据未审清前，过早把 `17` 小时长跑当正式 run。
- 把“benchmark 吞吐更高”过早解释成“生产配置已最优”，忽略了 correctness 层风险。
- 对 `navrl_upstream` 地图模式先接入口、后做 API 契约核查，顺序不理想。
- 自治链为了保留 failed JSON 而弱化返回码检查后，没有同步补上 payload-level success guard。

## Missed Triggers
- 新增或切换环境配置模式时，必须先核对当前安装版 Isaac Lab 的 config schema。
- 有 `terminated/truncated` 双终止语义时，必须先审 PPO/GAE bootstrap 逻辑，再开长跑。
- 长跑进入持续 NaN 后，必须自动 fail-fast，而不是继续产出 checkpoint。
- benchmark 用来指导正式模式时，默认候选集必须覆盖真实决策空间，且不能与当前正式训练并发。
- 自治链任何阶段只要 payload 明确写 `failed`，就必须把整个 cycle 判为失败。

## Trigger Redesign
- Guard 1:
  - 任何 `eval_checkpoint.py` 的产物必须同时满足:
    - 进程 return code 符合预期
    - JSON 文件存在
    - JSON `status` 为成功态
  - 三者缺一不可
- Guard 2:
  - 新增 `env.map_source`、terrain mode 或 sensor mode 时，必须先做“本机 API 契约探针”
- Guard 3:
  - `PPO/GAE/ValueNorm` 修改后，开正式长跑前必须跑一个专门的 `done/terminated/truncated` 语义单测
- Guard 4:
  - 正式长跑必须启用 NaN fail-fast:
    - 参数 NaN
    - loss NaN
    - logprob NaN
    - value NaN
  - 任一命中即停训并标失败
- Guard 5:
  - benchmark 脚本必须自带并发保护，并记录“当前是否存在正式训练进程”

## Next Actions
### 已落地
- 已把本轮关键操作、错误、根因与解决状态写入本文档。
- 已确认当前长跑不应继续被当成正式结果来源。
- 已确认 pre-NaN 与 post-NaN checkpoint 的处置边界。

### 待跟进
1. 停止当前 `pilot_20260402_210831` 长跑，并保留日志与 checkpoint 证据。
2. 修复 `truncated -> done` 在 PPO/GAE 中的处理。
3. 修复自治链成功判据，拒绝 failed eval payload。
4. 给训练加 NaN fail-fast 与首个异常快照。
5. 禁用或修正 `navrl_upstream`，至少做到显式报错而不是静默退出。
6. 为 benchmark 增加并发保护，并把默认候选集更新到 `48/64/96/128` 级别。
7. 仅在上述 correctness blocker 修复后，再开新的短验证 run 与正式长跑。

## Deliverables
- 复盘文档: `/home/gwh/dashgo_navrl_project/docs/navrl_longrun_retro_2026-04-02_22-28.md`
- 当前自治状态: `/home/gwh/dashgo_navrl_project/artifacts/autonomous/20260402_210824/status.json`
- 当前训练日志: `/home/gwh/dashgo_navrl_project/artifacts/supervisor/pilot/train.log`
- 当前 run 根目录: `/home/gwh/dashgo_navrl_project/artifacts/runs/pilot_20260402_210831`

## Final Judgment
- 第一阶段“独立仓库 + 训练/评测/对比闭环”是成功的。
- 第二阶段开头这轮“模式摸索 + 17 小时自治长跑”在吞吐探索上有价值，但在训练正确性上尚未收口。
- 当前正确做法不是继续解释这条 run 的分数，而是先把 correctness blocker 修掉，再重启正式训练。
