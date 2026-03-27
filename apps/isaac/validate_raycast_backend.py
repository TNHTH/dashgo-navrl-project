#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo RayCaster 可行性验证")
    parser.add_argument("--autopilot-profile", default="gen1", help="验证场景使用的 autopilot profile")
    parser.add_argument("--json-out", type=Path, default=None, help="可选：把验证结果写入 JSON")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_robot_pose(env, env_ids: torch.Tensor) -> None:
    from dashgo_rl.dashgo_env_v2 import quat_from_euler_xyz

    robot = env.scene["robot"]
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, 0] = env.scene.env_origins[env_ids, 0]
    root_state[:, 1] = env.scene.env_origins[env_ids, 1]
    root_state[:, 2] = 0.20
    zeros = torch.zeros(len(env_ids), device=env.device)
    root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, zeros)
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)


def place_static_obstacles(env, env_ids: torch.Tensor, focus_x: float) -> None:
    from dashgo_rl.dashgo_env_v2 import _write_kinematic_obstacle_pose

    origin_xy = env.scene.env_origins[env_ids, :2]
    zeros = torch.zeros(len(env_ids), device=env.device)
    far_xy = origin_xy + torch.tensor([[40.0, 40.0]], device=env.device)
    focus_xy = origin_xy + torch.tensor([[focus_x, 0.0]], device=env.device)

    obstacle_names = sorted(name for name in env.scene.keys() if name.startswith("obs_"))
    for asset_name in obstacle_names:
        target_xy = focus_xy if asset_name == "obs_inner_1" else far_xy
        _write_kinematic_obstacle_pose(env, env_ids, asset_name, target_xy, zeros)


def settle_scene(env, steps: int = 3) -> None:
    zero_action = torch.zeros((env.num_envs, 2), device=env.device)
    for _ in range(steps):
        env.step(zero_action)


def get_camera_min_distance(env) -> float:
    from dashgo_rl.dashgo_env_v2 import _get_min_obstacle_distance

    return float(_get_min_obstacle_distance(env)[0].item())


def make_raycaster(mesh_prim_paths: list[str], headless: bool):
    from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns

    cfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Dashgo/base_link",
        mesh_prim_paths=mesh_prim_paths,
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.22)),
        attach_yaw_only=False,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-90.0, 90.0),
            horizontal_res=1.0,
        ),
        max_distance=12.0,
        debug_vis=not headless,
    )
    sensor = RayCaster(cfg)
    if not sensor.is_initialized:
        sensor._initialize_callback(None)
    return sensor


def get_raycaster_min_distance(sensor, env) -> float:
    sensor.update(dt=env.sim.get_physics_dt(), force_recompute=True)
    hit_pos = sensor.data.ray_hits_w[0]
    sensor_pos = sensor.data.pos_w[0].unsqueeze(0)
    distances = torch.norm(hit_pos - sensor_pos, dim=-1)
    distances = torch.nan_to_num(distances, posinf=12.0, neginf=0.0)
    distances = torch.clamp(distances, min=0.0, max=12.0)
    return float(distances.min().item())


def main() -> int:
    parser = build_parser()
    args_cli = parser.parse_args()
    args_cli.enable_cameras = True

    simulation_app = AppLauncher(args_cli).app
    env = None
    try:
        os.environ["DASHGO_AUTOPILOT_PROFILE"] = args_cli.autopilot_profile.strip().lower()

        from isaaclab.envs import ManagerBasedRLEnv

        from dashgo_rl.dashgo_env_v2 import DashgoNavEnvV2Cfg

        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = 1
        env_cfg.scene.env_spacing = 15.0
        env_cfg.seed = 42
        env = ManagerBasedRLEnv(cfg=env_cfg)
        env.reset()

        env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
        set_robot_pose(env, env_ids)
        place_static_obstacles(env, env_ids, focus_x=1.2)
        settle_scene(env)

        camera_initial = get_camera_min_distance(env)

        multi_mesh = {"supported": True, "error": None}
        try:
            multi_sensor = make_raycaster(
                ["/World/envs/env_0/Obs_In_1", "/World/envs/env_0/Obs_In_2"],
                headless=bool(args_cli.headless),
            )
            _ = get_raycaster_min_distance(multi_sensor, env)
        except Exception as exc:  # noqa: BLE001
            multi_mesh["supported"] = False
            multi_mesh["error"] = repr(exc)

        single_mesh = {
            "mesh_prim_path": "/World/envs/env_0/Obs_In_1",
            "initial_min_distance": None,
            "after_move_min_distance": None,
            "error": None,
        }
        camera_after_move = None
        raycaster_stale_after_move = None

        try:
            single_sensor = make_raycaster([single_mesh["mesh_prim_path"]], headless=bool(args_cli.headless))
            single_mesh["initial_min_distance"] = get_raycaster_min_distance(single_sensor, env)

            place_static_obstacles(env, env_ids, focus_x=2.6)
            settle_scene(env)
            camera_after_move = get_camera_min_distance(env)
            single_mesh["after_move_min_distance"] = get_raycaster_min_distance(single_sensor, env)

            raycaster_stale_after_move = (
                abs(single_mesh["after_move_min_distance"] - single_mesh["initial_min_distance"]) < 0.15
                and abs(camera_after_move - camera_initial) > 0.50
            )
        except Exception as exc:  # noqa: BLE001
            single_mesh["error"] = repr(exc)

        result = {
            "autopilot_profile": args_cli.autopilot_profile,
            "camera_backend": {
                "initial_min_distance": camera_initial,
                "after_move_min_distance": camera_after_move,
            },
            "raycaster_multi_mesh": multi_mesh,
            "raycaster_single_mesh": single_mesh,
            "raycaster_stale_after_move": raycaster_stale_after_move,
        }
        result["decision"] = (
            "fallback_to_camera"
            if (not multi_mesh["supported"]) or bool(single_mesh["error"]) or bool(raycaster_stale_after_move)
            else "raycaster_viable"
        )

        if args_cli.json_out is not None:
            write_json(args_cli.json_out.resolve(), result)

        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
