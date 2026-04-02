#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from isaaclab.app import AppLauncher

from navrl_dashgo.metrics import behavior_gate_violations, summarize_eval_episodes
from navrl_dashgo.types import EvalRequest, EvalResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo NavRL Isaac 评测 worker")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=["quick", "main"], default="quick")
    parser.add_argument("--requested-episodes", type=int, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def suite_scenarios(suite: str) -> list[dict]:
    quick = [
        {"goal": (1.5, 0.0), "yaw": 0.0, "reverse_case": False},
        {"goal": (2.0, 0.5), "yaw": 0.0, "reverse_case": False},
        {"goal": (1.2, -0.8), "yaw": 0.0, "reverse_case": False},
        {"goal": (-1.0, 0.0), "yaw": 0.0, "reverse_case": True},
        {"goal": (-1.2, 0.8), "yaw": 0.0, "reverse_case": True},
        {"goal": (-1.2, -0.8), "yaw": 0.0, "reverse_case": True},
    ]
    main = quick + [
        {"goal": (2.6, 0.0), "yaw": 0.0, "reverse_case": False},
        {"goal": (2.2, 1.2), "yaw": 0.0, "reverse_case": False},
        {"goal": (2.2, -1.2), "yaw": 0.0, "reverse_case": False},
        {"goal": (-2.0, 0.0), "yaw": 0.0, "reverse_case": True},
        {"goal": (-1.8, 1.0), "yaw": 0.0, "reverse_case": True},
        {"goal": (-1.8, -1.0), "yaw": 0.0, "reverse_case": True},
    ]
    return quick if suite == "quick" else main


def resolve_total_episodes(suite: str, requested_episodes: int | None) -> int:
    defaults = {"quick": 12, "main": 48}
    return int(requested_episodes or defaults[suite])


def step_env_without_auto_reset(env, action: torch.Tensor):
    """复刻 Isaac Lab step 主流程，但把 reset 留给评测循环显式控制。"""
    env.action_manager.process_action(action.to(env.device))
    env.recorder_manager.record_pre_step()

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)

    env.episode_length_buf += 1
    env.common_step_counter += 1
    env.reset_buf = env.termination_manager.compute()
    env.reset_terminated = env.termination_manager.terminated.clone()
    env.reset_time_outs = env.termination_manager.time_outs.clone()
    env.reward_buf = env.reward_manager.compute(dt=env.step_dt)

    if len(env.recorder_manager.active_terms) > 0:
        env.obs_buf = env.observation_manager.compute()
        env.recorder_manager.record_post_step()

    env.command_manager.compute(dt=env.step_dt)
    if "interval" in env.event_manager.available_modes:
        env.event_manager.apply(mode="interval", dt=env.step_dt)
    env.obs_buf = env.observation_manager.compute()
    return env.obs_buf, env.reward_buf, env.reset_terminated, env.reset_time_outs, env.extras


def set_robot_state(env, env_ids: torch.Tensor, yaw: torch.Tensor) -> None:
    from dashgo_rl.dashgo_env_navrl_official import quat_from_euler_xyz

    robot = env.scene["robot"]
    state = robot.data.default_root_state[env_ids].clone()
    state[:, 0] = env.scene.env_origins[env_ids, 0]
    state[:, 1] = env.scene.env_origins[env_ids, 1]
    state[:, 2] = 0.20
    zeros = torch.zeros_like(yaw)
    state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
    state[:, 7:] = 0.0
    robot.write_root_state_to_sim(state, env_ids=env_ids)


def set_goal(env, env_ids: torch.Tensor, goal_xy: torch.Tensor) -> None:
    cmd_term = env.command_manager.get_term("target_pose")
    origins = env.scene.env_origins[env_ids]
    cmd_term.goal_pose_w[env_ids, 0] = origins[:, 0] + goal_xy[:, 0]
    cmd_term.goal_pose_w[env_ids, 1] = origins[:, 1] + goal_xy[:, 1]
    cmd_term.goal_pose_w[env_ids, 2] = 0.0
    cmd_term.goal_pose_w[env_ids, 3] = 1.0
    cmd_term.goal_pose_w[env_ids, 4:] = 0.0
    cmd_term.pose_command_w[env_ids] = cmd_term.goal_pose_w[env_ids]
    cmd_term.heading_command_w[env_ids] = 0.0
    if hasattr(cmd_term, "_build_linear_reference_path"):
        start_xy = origins[:, :2]
        goal_world = cmd_term.goal_pose_w[env_ids, :2]
        path, steps = cmd_term._build_linear_reference_path(start_xy, goal_world)
        cmd_term.reference_path_w[env_ids] = path
        cmd_term.reference_path_len[env_ids] = steps
        cmd_term.reference_path_cursor[env_ids] = 0
        cmd_term.waypoint_pose_w[env_ids] = cmd_term.goal_pose_w[env_ids]


def initialize_episode_state(env, env_ids: torch.Tensor, scenarios: list[dict], next_scene_idx: int, stats: dict) -> int:
    from dashgo_rl.dashgo_env_navrl_official import _get_min_obstacle_distance, _get_target_delta_and_heading
    from isaaclab.managers import SceneEntityCfg

    goals = []
    yaws = []
    for env_id in env_ids.tolist():
        scene = scenarios[next_scene_idx % len(scenarios)]
        next_scene_idx += 1
        goals.append(scene["goal"])
        yaws.append(scene["yaw"])
        stats[env_id] = {
            "scene_index": (next_scene_idx - 1) % len(scenarios),
            "reverse_case": scene["reverse_case"],
            "steps": 0,
            "path_length": 0.0,
            "spin_steps": 0,
            "clip_steps": 0,
            "near_obstacle_steps": 0,
            "near_obstacle_streak": 0,
            "orbit_progress_streak": 0,
            "orbit_yaw_accum": 0.0,
            "orbit_detected": False,
            "sensor_health_score": 1.0,
        }
    goal_tensor = torch.tensor(goals, device=env.device, dtype=torch.float32)
    yaw_tensor = torch.tensor(yaws, device=env.device, dtype=torch.float32)
    set_robot_state(env, env_ids, yaw_tensor)
    set_goal(env, env_ids, goal_tensor)

    asset_cfg = SceneEntityCfg("robot")
    min_obstacle = _get_min_obstacle_distance(env)[env_ids]
    delta_pos, _, _ = _get_target_delta_and_heading(env, "target_pose", asset_cfg)
    start_distance = torch.norm(delta_pos[env_ids], dim=-1)
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w[env_ids, :2].detach().clone()
    for idx, env_id in enumerate(env_ids.tolist()):
        stats[env_id]["start_distance"] = float(start_distance[idx].item())
        stats[env_id]["last_distance"] = float(start_distance[idx].item())
        stats[env_id]["last_position"] = pos[idx].detach().cpu().tolist()
        stats[env_id]["sensor_health_score"] = 0.0 if min_obstacle[idx].isnan().item() else 1.0
    return next_scene_idx


def reset_done_envs_for_next_episode(
    env,
    done_ids: torch.Tensor,
    scenarios: list[dict],
    next_scene_idx: int,
    stats: dict,
) -> tuple[int, dict]:
    env.reset(env_ids=done_ids)
    next_scene_idx = initialize_episode_state(env, done_ids, scenarios, next_scene_idx, stats)
    return next_scene_idx, env.observation_manager.compute()


def finalize_episode(env, env_id: int, stat: dict, reason: str) -> dict:
    from dashgo_rl.dashgo_env_navrl_official import _get_target_delta_and_heading
    from isaaclab.managers import SceneEntityCfg

    asset_cfg = SceneEntityCfg("robot")
    env_id_tensor = torch.tensor([env_id], device=env.device, dtype=torch.long)
    delta_pos, _, _ = _get_target_delta_and_heading(env, "target_pose", asset_cfg)
    end_distance = float(torch.norm(delta_pos[env_id_tensor], dim=-1)[0].item())
    steps = max(1, int(stat["steps"]))
    start_distance = float(stat["start_distance"])
    path_length = float(stat["path_length"])
    direct = max(start_distance, 1.0e-6)
    progress = max(0.0, start_distance - end_distance)
    path_eff = max(0.0, min(1.0, progress / max(path_length, direct)))
    net_progress_ratio = max(0.0, min(1.0, progress / direct))
    near_obstacle_dwell = stat["near_obstacle_steps"] / steps
    spin_proxy_ratio = stat["spin_steps"] / steps
    high_clip_ratio = stat["clip_steps"] / steps
    progress_stall = net_progress_ratio < 0.15 and reason != "reach_goal"
    return {
        "scene_index": stat["scene_index"],
        "reverse_case": bool(stat["reverse_case"]),
        "termination_reason": reason,
        "steps": steps,
        "elapsed_time": float(stat.get("control_dt", 0.0)) * steps,
        "start_distance": start_distance,
        "end_distance": end_distance,
        "path_length": path_length,
        "path_efficiency": path_eff,
        "net_progress_ratio": net_progress_ratio,
        "near_obstacle_dwell_ratio": near_obstacle_dwell,
        "spin_proxy_ratio": spin_proxy_ratio,
        "high_clip_ratio": high_clip_ratio,
        "progress_stall": progress_stall,
        "orbit_detected": bool(stat["orbit_detected"]),
        "sensor_health_score": float(stat["sensor_health_score"]),
        "heading_guard_trigger_rate": 0.0,
        "recovery_trigger_rate": 0.0,
        "plan_invalid_ratio": 0.0,
    }


def load_cfg_from_checkpoint(checkpoint: Path, suite: str):
    from omegaconf import OmegaConf

    payload = torch.load(checkpoint, map_location="cpu")
    cfg = OmegaConf.create(payload.get("config", {}))
    cfg.headless = True
    cfg.enable_cameras = bool(getattr(cfg, "enable_cameras", False))
    cfg.env.num_envs = 4 if suite == "quick" else 6
    return cfg


def emit_result(json_out: Path, result: EvalResult) -> None:
    payload = result.to_dict()
    write_json(json_out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def load_algo_checkpoint(algo: torch.nn.Module, payload: dict) -> list[str]:
    notes: list[str] = []

    inference_state = payload.get("inference_state_dict")
    if isinstance(inference_state, dict):
        algo.feature_extractor.load_state_dict(inference_state["feature_extractor"], strict=True)
        algo.actor.load_state_dict(inference_state["actor"], strict=True)
        algo.critic.load_state_dict(inference_state["critic"], strict=True)
        value_norm_state = inference_state.get("value_norm")
        if isinstance(value_norm_state, dict):
            load_result = algo.value_norm.load_state_dict(value_norm_state, strict=False)
            if load_result.missing_keys:
                notes.append(f"value_norm_missing_keys={load_result.missing_keys}")
            if load_result.unexpected_keys:
                notes.append(f"value_norm_unexpected_keys={load_result.unexpected_keys}")
        return notes

    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    load_result = algo.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = ("value_norm.", "gae.")
    critical_missing = [key for key in load_result.missing_keys if not key.startswith(allowed_missing_prefixes)]
    if critical_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "checkpoint 与评测模型结构不兼容: "
            f"critical_missing={critical_missing}, unexpected={load_result.unexpected_keys}"
        )
    if load_result.missing_keys:
        notes.append(f"legacy_checkpoint_missing_keys={load_result.missing_keys}")
    return notes


def main() -> int:
    parser = build_parser()
    args_cli, _ = parser.parse_known_args()
    cfg = load_cfg_from_checkpoint(args_cli.checkpoint.resolve(), args_cli.suite)
    args_cli.enable_cameras = bool(getattr(args_cli, "enable_cameras", False) or bool(cfg.enable_cameras))

    request = EvalRequest(
        checkpoint=args_cli.checkpoint.resolve(),
        suite=args_cli.suite,
        project_root=PROJECT_ROOT,
        requested_episodes=args_cli.requested_episodes,
        notes=["DashGo NavRL quick/main 评测 worker"],
    )

    simulation_app = None
    torchrl_env = None
    try:
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        from navrl_dashgo.env_adapter import TorchRLDashgoEnv
        from navrl_dashgo.ppo import PPO
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        torchrl_env = TorchRLDashgoEnv(cfg)
        env = torchrl_env.base_env
        adapter = torchrl_env.adapter
        algo = PPO(cfg.algo, torchrl_env.observation_spec, torchrl_env.agent_action_spec, torchrl_env.device)
        payload = torch.load(args_cli.checkpoint.resolve(), map_location=torchrl_env.device)
        load_notes = load_algo_checkpoint(algo, payload if isinstance(payload, dict) else {"model_state_dict": payload})
        algo.eval()

        scenarios = suite_scenarios(args_cli.suite)
        total_episodes = resolve_total_episodes(args_cli.suite, args_cli.requested_episodes)
        control_dt = float(env.cfg.sim.dt * env.cfg.decimation)

        episodes: list[dict] = []
        active_stats: dict[int, dict] = {}
        next_scene_idx = 0
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        env.reset()
        next_scene_idx = initialize_episode_state(env, env_ids, scenarios, next_scene_idx, active_stats)
        raw_obs = env.observation_manager.compute()

        from dashgo_rl.dashgo_env_navrl_official import _get_min_obstacle_distance, _get_target_delta_and_heading, process_stitched_lidar
        from isaaclab.managers import SceneEntityCfg

        asset_cfg = SceneEntityCfg("robot")

        while len(episodes) < total_episodes and simulation_app.is_running():
            td = adapter.encode(raw_obs).to(torchrl_env.device)
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                algo(td)
            action = td["agents", "action"].squeeze(1)
            raw_obs, _, terminated, truncated, _ = step_env_without_auto_reset(env, action)
            dones = terminated | truncated

            min_obstacle = _get_min_obstacle_distance(env)
            lidar = process_stitched_lidar(env)
            delta_pos, _, _ = _get_target_delta_and_heading(env, "target_pose", asset_cfg)
            distance = torch.norm(delta_pos, dim=-1)
            robot = env.scene["robot"]
            position = robot.data.root_pos_w[:, :2].detach()
            lin_speed = torch.abs(robot.data.root_lin_vel_b[:, 0]).detach()
            ang_speed = torch.abs(robot.data.root_ang_vel_b[:, 2]).detach()

            for env_id in range(env.num_envs):
                if env_id not in active_stats:
                    continue
                stat = active_stats[env_id]
                stat["steps"] += 1
                stat["control_dt"] = control_dt
                last_position = torch.tensor(stat["last_position"], device=env.device)
                stat["path_length"] += float(torch.norm(position[env_id] - last_position).item())
                stat["last_position"] = position[env_id].detach().cpu().tolist()
                current_distance = float(distance[env_id].item())
                delta_distance = abs(current_distance - float(stat["last_distance"]))
                stat["last_distance"] = current_distance
                if abs(float(action[env_id, 1].item())) > 0.95 or abs(float(action[env_id, 0].item())) > 0.95:
                    stat["clip_steps"] += 1
                if float(ang_speed[env_id].item()) > 0.6 and float(lin_speed[env_id].item()) < 0.03:
                    stat["spin_steps"] += 1
                if float(min_obstacle[env_id].item()) < 0.35:
                    stat["near_obstacle_steps"] += 1
                    stat["near_obstacle_streak"] += 1
                else:
                    stat["near_obstacle_streak"] = 0
                if delta_distance < 0.02:
                    stat["orbit_progress_streak"] += 1
                    stat["orbit_yaw_accum"] += float(ang_speed[env_id].item()) * control_dt
                    if stat["orbit_progress_streak"] >= 20 and stat["orbit_yaw_accum"] > (2.0 * 3.1415926):
                        stat["orbit_detected"] = True
                else:
                    stat["orbit_progress_streak"] = 0
                    stat["orbit_yaw_accum"] = 0.0
                if bool(torch.isnan(min_obstacle[env_id]) or torch.isnan(lidar[env_id]).any()):
                    stat["sensor_health_score"] = 0.0

            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if done_ids.numel() == 0:
                continue

            reach = env.termination_manager.get_term("reach_goal")
            collision = env.termination_manager.get_term("object_collision")
            timeout = env.termination_manager.get_term("time_out")
            for env_id in done_ids.tolist():
                if env_id not in active_stats:
                    continue
                if bool(reach[env_id].item()):
                    reason = "reach_goal"
                elif bool(collision[env_id].item()):
                    reason = "object_collision"
                elif bool(timeout[env_id].item()):
                    reason = "time_out"
                else:
                    reason = "unknown"
                episodes.append(finalize_episode(env, env_id, active_stats.pop(env_id), reason))
                if len(episodes) >= total_episodes:
                    break
            remaining_ids = done_ids[: max(0, min(done_ids.numel(), total_episodes - len(episodes)))]
            if remaining_ids.numel() > 0 and len(episodes) < total_episodes:
                next_scene_idx, raw_obs = reset_done_envs_for_next_episode(
                    env,
                    remaining_ids,
                    scenarios,
                    next_scene_idx,
                    active_stats,
                )

        metrics = summarize_eval_episodes(episodes, suite=args_cli.suite)
        violations = behavior_gate_violations(metrics, suite=args_cli.suite)
        status = "completed" if not violations else "failed"
        result = EvalResult(
            status=status,
            request=request,
            metrics=metrics,
            scenes=episodes,
            notes=[
                *load_notes,
                *([] if not violations else [f"behavior_gate_veto: {', '.join(violations)}"]),
            ],
            metadata={"suite": args_cli.suite, "violations": violations},
        )
        emit_result(args_cli.json_out, result)
        return 0 if status == "completed" else 1
    except BaseException as exc:  # noqa: BLE001
        result = EvalResult(
            status="failed",
            request=request,
            notes=[f"DashGo NavRL eval worker 失败: {type(exc).__name__}: {exc}", traceback.format_exc()],
            metadata={"suite": request.suite, "exception_type": type(exc).__name__},
        )
        emit_result(args_cli.json_out, result)
        return 1
    finally:
        if torchrl_env is not None:
            try:
                torchrl_env.close()
            except Exception:
                pass
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
