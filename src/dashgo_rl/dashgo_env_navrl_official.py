from __future__ import annotations

import math

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg, mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim.simulation_cfg import RenderCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfDiscreteObstaclesTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, wrap_to_pi

from navrl_dashgo.semantics import build_reference_path_progress, compute_waypoint_lookahead_indices

from .dashgo_assets import DASHGO_D1_CFG
from .dashgo_config import DashGoROSParams


SIM_LIDAR_MAX_RANGE = 12.0
SIM_LIDAR_POLICY_DIM = 72
GOAL_DISTANCE_RANGE = (1.0, 4.0)
OBSERVATION_CONFIG = {
    "epsilon": 1.0e-6,
    "max_distance": 8.0,
    "waypoint_distance": 1.0,
    "path_resolution": 0.2,
    "max_path_points": 48,
}
MOTION_CONFIG = {
    "max_lin_vel": 0.3,
    "max_reverse_speed": 0.15,
    "max_ang_vel": 1.0,
    "max_accel_lin": 1.0,
    "max_accel_ang": 0.6,
    "max_wheel_vel": 5.0,
}
REWARD_CONFIG = {
    # 保持碰撞判据不变，避免靠放松安全线“做高”成功率。
    "goal_termination_threshold": 0.6,
    "goal_stop_velocity": 0.08,
    # 当前 DashGo 终止条件要求“到点且低速停稳”，补一个对齐该判据的终态奖励，
    # 但保留 NavRL 以速度/安全为主的主体思路，不回退到旧版大规模 shaping。
    "goal_reached_bonus_weight": 12.0,
    "dynamic_collision_threshold": 0.3,
    "static_collision_threshold": 0.3,
    "survival_weight": 0.15,
    "goal_velocity_weight": 1.0,
    "waypoint_velocity_weight": 0.35,
    "static_safety_weight": 1.0,
    "dynamic_safety_weight": 1.0,
    "twist_smoothness_weight": -0.1,
    "progress_stall_weight": -0.2,
    "orbit_weight": -0.15,
    "stall_activation_distance": 0.75,
    "stall_min_progress": 0.005,
    "stall_max_forward_speed": 0.05,
    "stall_warmup_steps": 15,
    "stall_trigger_steps": 8,
    "orbit_activation_distance": 0.75,
    "orbit_min_progress": 0.01,
    "orbit_min_angular_speed": 0.35,
    "orbit_max_forward_speed": 0.18,
    "orbit_warmup_steps": 20,
    "orbit_trigger_steps": 10,
}
MAX_DYNAMIC_SCENE_OBJECTS = 24
DEFAULT_DYNAMIC_OBSTACLE_COUNT = 8
DYNAMIC_OBSTACLE_INTERVAL_S = 0.10
DYNAMIC_OBSTACLE_LOCAL_RANGE = 4.0
DYNAMIC_OBSTACLE_SPEED_RANGE = (0.15, 0.45)
DYNAMIC_OBSTACLE_HEIGHT = 1.0
DYNAMIC_OBSTACLE_WIDTHS = tuple((0.35, 0.50, 0.65, 0.80)[index % 4] for index in range(MAX_DYNAMIC_SCENE_OBJECTS))
DYNAMIC_OBSTACLE_IS_CYLINDER = tuple(index % 2 == 0 for index in range(MAX_DYNAMIC_SCENE_OBJECTS))


def _rotate_world_to_local(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw).unsqueeze(-1)
    sin_yaw = torch.sin(yaw).unsqueeze(-1)
    rot_x = xy[..., 0:1] * cos_yaw + xy[..., 1:2] * sin_yaw
    rot_y = -xy[..., 0:1] * sin_yaw + xy[..., 1:2] * cos_yaw
    return torch.cat([rot_x, rot_y], dim=-1)


def _resolve_event_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def build_navrl_terrain_cfg(num_obstacles: int) -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            curriculum=False,
            size=(20.0, 20.0),
            border_width=5.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.1,
            slope_threshold=0.75,
            color_scheme="none",
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    proportion=1.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=0.0,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(0.4, 1.1),
                    obstacle_height_range=(1.0, 1.0),
                    num_obstacles=max(0, int(num_obstacles)),
                    platform_width=2.0,
                ),
            },
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        debug_vis=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )


def build_navrl_upstream_terrain_cfg(num_obstacles: int) -> TerrainImporterCfg:
    """保留 DashGo 底盘语义，但把静态地图形态切到更接近 upstream NavRL 的障碍地形。"""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            curriculum=False,
            size=(20.0, 20.0),
            border_width=5.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.1,
            slope_threshold=0.75,
            # 当前 Isaac Lab / trimesh 组合会把 "height" 映射到默认 turbo colormap，
            # 但本机环境只内置 magma/inferno/plasma/viridis，直接关闭颜色映射避免启动失败。
            color_scheme="none",
            use_cache=False,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    proportion=1.0,
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=0.0,
                    obstacle_height_mode="choice",
                    obstacle_width_range=(0.4, 1.1),
                    obstacle_height_range=(1.0, 6.0),
                    num_obstacles=max(0, int(num_obstacles)),
                    platform_width=0.0,
                ),
            },
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        debug_vis=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )


def terrain_debug_summary(map_source: str, num_obstacles: int) -> dict[str, float | int | str]:
    if map_source == "navrl_upstream":
        return {
            "map_source": map_source,
            "obstacle_height_mode": "choice",
            "obstacle_height_range_min": 1.0,
            "obstacle_height_range_max": 6.0,
            "platform_width": 0.0,
            "num_obstacles": int(num_obstacles),
        }
    return {
        "map_source": map_source,
        "obstacle_height_mode": "fixed",
        "obstacle_height_range_min": 1.0,
        "obstacle_height_range_max": 1.0,
        "platform_width": 2.0,
        "num_obstacles": int(num_obstacles),
    }


def build_dynamic_obstacle_collection_cfg(num_slots: int = MAX_DYNAMIC_SCENE_OBJECTS) -> RigidObjectCollectionCfg:
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for index in range(num_slots):
        width = DYNAMIC_OBSTACLE_WIDTHS[index]
        if DYNAMIC_OBSTACLE_IS_CYLINDER[index]:
            spawn_cfg = sim_utils.CylinderCfg(
                radius=width * 0.5,
                height=DYNAMIC_OBSTACLE_HEIGHT,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.2)),
            )
        else:
            spawn_cfg = sim_utils.CuboidCfg(
                size=(width, width, DYNAMIC_OBSTACLE_HEIGHT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.2)),
            )
        rigid_objects[f"dyn_obs_{index:02d}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/DynObs_{index:02d}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(50.0 + float(index), 50.0, 0.5)),
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


def _ensure_dynamic_obstacle_state(env: ManagerBasedRLEnv, num_slots: int) -> dict:
    state = getattr(env, "_navrl_dynamic_obstacle_state", None)
    if state is not None and state.get("num_envs") == env.num_envs and state.get("num_slots") == num_slots:
        return state
    slot_widths = torch.tensor(DYNAMIC_OBSTACLE_WIDTHS[:num_slots], device=env.device, dtype=torch.float32)
    slot_shapes = torch.tensor(
        [1.0 if DYNAMIC_OBSTACLE_IS_CYLINDER[idx] else 0.0 for idx in range(num_slots)],
        device=env.device,
        dtype=torch.float32,
    )
    size_xy = torch.stack([slot_widths, slot_widths], dim=-1).unsqueeze(0).repeat(env.num_envs, 1, 1)
    state = {
        "num_envs": env.num_envs,
        "num_slots": num_slots,
        "active_mask": torch.zeros((env.num_envs, num_slots), device=env.device, dtype=torch.bool),
        "goal_xy": torch.zeros((env.num_envs, num_slots, 2), device=env.device),
        "speed": torch.zeros((env.num_envs, num_slots), device=env.device),
        "size_xy": size_xy,
        "shape": slot_shapes.unsqueeze(0).repeat(env.num_envs, 1),
    }
    env._navrl_dynamic_obstacle_state = state
    return state


def _sample_dynamic_positions(
    env: ManagerBasedRLEnv,
    env_count: int,
    num_slots: int,
    inner_radius: float = 1.6,
    outer_radius: float = 6.5,
) -> torch.Tensor:
    radius = torch.empty((env_count, num_slots), device=env.device).uniform_(inner_radius, outer_radius)
    angle = torch.empty((env_count, num_slots), device=env.device).uniform_(-math.pi, math.pi)
    pos_xy = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
    return pos_xy


def _sample_dynamic_goals(env: ManagerBasedRLEnv, current_xy: torch.Tensor) -> torch.Tensor:
    displacement = torch.empty_like(current_xy).uniform_(-DYNAMIC_OBSTACLE_LOCAL_RANGE, DYNAMIC_OBSTACLE_LOCAL_RANGE)
    goal_xy = current_xy + displacement
    return torch.clamp(goal_xy, min=-8.0, max=8.0)


def _write_dynamic_obstacles_to_sim(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
        return
    collection = env.scene["dynamic_obstacles"]
    state = _ensure_dynamic_obstacle_state(env, collection.num_objects)
    origins = env.scene.env_origins[env_ids, :2]

    pose = collection.data.default_object_state[env_ids].clone()
    velocity = torch.zeros((len(env_ids), collection.num_objects, 6), device=env.device)

    active = state["active_mask"][env_ids]
    local_goal = state["goal_xy"][env_ids]
    local_pos = state.get("local_pos_xy")
    local_vel = state.get("local_vel_xy")
    if local_pos is None or local_vel is None:
        local_pos = torch.zeros((env.num_envs, collection.num_objects, 2), device=env.device)
        local_vel = torch.zeros((env.num_envs, collection.num_objects, 2), device=env.device)
        state["local_pos_xy"] = local_pos
        state["local_vel_xy"] = local_vel
    local_pos = state["local_pos_xy"][env_ids]
    local_vel = state["local_vel_xy"][env_ids]

    pose[:, :, 0:2] = origins.unsqueeze(1) + local_pos
    pose[:, :, 2] = DYNAMIC_OBSTACLE_HEIGHT * 0.5
    pose[:, :, 3] = 1.0
    pose[:, :, 4:] = 0.0
    velocity[:, :, 0:2] = local_vel
    inactive_mask = ~active
    if torch.any(inactive_mask):
        far_world = origins.unsqueeze(1) + torch.full_like(local_goal, 60.0)
        pose[:, :, 0:2] = torch.where(inactive_mask.unsqueeze(-1), far_world, pose[:, :, 0:2])
        pose[:, :, 2] = torch.where(inactive_mask, torch.full_like(pose[:, :, 2], -5.0), pose[:, :, 2])
        velocity = torch.where(inactive_mask.unsqueeze(-1), torch.zeros_like(velocity), velocity)

    object_ids = torch.arange(collection.num_objects, device=env.device, dtype=torch.long)
    collection.write_object_link_pose_to_sim(pose[..., :7], env_ids=env_ids, object_ids=object_ids)
    collection.write_object_link_velocity_to_sim(velocity, env_ids=env_ids, object_ids=object_ids)


def configure_dynamic_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    num_active: int = DEFAULT_DYNAMIC_OBSTACLE_COUNT,
) -> None:
    env_ids = _resolve_event_env_ids(env, env_ids)
    collection = env.scene["dynamic_obstacles"]
    num_slots = collection.num_objects
    state = _ensure_dynamic_obstacle_state(env, num_slots)
    num_active = max(0, min(int(num_active), num_slots))

    state["active_mask"][env_ids] = False
    state["speed"][env_ids] = 0.0
    if "local_pos_xy" not in state:
        state["local_pos_xy"] = torch.zeros((env.num_envs, num_slots, 2), device=env.device)
        state["local_vel_xy"] = torch.zeros((env.num_envs, num_slots, 2), device=env.device)
    state["local_pos_xy"][env_ids] = 0.0
    state["local_vel_xy"][env_ids] = 0.0
    state["goal_xy"][env_ids] = 0.0
    if num_active > 0:
        selected_positions = _sample_dynamic_positions(env, len(env_ids), num_slots)
        selected_goals = _sample_dynamic_goals(env, selected_positions)
        speeds = torch.empty((len(env_ids), num_slots), device=env.device).uniform_(*DYNAMIC_OBSTACLE_SPEED_RANGE)
        random_order = torch.argsort(torch.rand((len(env_ids), num_slots), device=env.device), dim=1)
        active_mask = random_order < num_active
        state["active_mask"][env_ids] = active_mask
        state["local_pos_xy"][env_ids] = selected_positions
        state["goal_xy"][env_ids] = selected_goals
        state["speed"][env_ids] = speeds
    _refresh_dynamic_obstacle_velocity(env, env_ids)
    _write_dynamic_obstacles_to_sim(env, env_ids)


def _refresh_dynamic_obstacle_velocity(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    collection = env.scene["dynamic_obstacles"]
    state = _ensure_dynamic_obstacle_state(env, collection.num_objects)
    local_pos = state["local_pos_xy"][env_ids]
    goal_xy = state["goal_xy"][env_ids]
    direction = goal_xy - local_pos
    distance = torch.norm(direction, dim=-1, keepdim=True)
    unit_dir = direction / distance.clamp_min(1.0e-6)
    velocity = unit_dir * state["speed"][env_ids].unsqueeze(-1)
    velocity = torch.where(state["active_mask"][env_ids].unsqueeze(-1), velocity, torch.zeros_like(velocity))
    state["local_vel_xy"][env_ids] = velocity


def animate_dynamic_obstacles(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, motion_dt: float) -> None:
    env_ids = _resolve_event_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    collection = env.scene["dynamic_obstacles"]
    state = _ensure_dynamic_obstacle_state(env, collection.num_objects)
    active = state["active_mask"][env_ids]
    if not torch.any(active):
        return

    local_pos = state["local_pos_xy"][env_ids]
    goal_xy = state["goal_xy"][env_ids]
    distance_to_goal = torch.norm(goal_xy - local_pos, dim=-1)
    resample_mask = active & (distance_to_goal < 0.5)
    if torch.any(resample_mask):
        refreshed_goals = _sample_dynamic_goals(env, local_pos)
        state["goal_xy"][env_ids] = torch.where(resample_mask.unsqueeze(-1), refreshed_goals, goal_xy)
        _refresh_dynamic_obstacle_velocity(env, env_ids)

    state["local_pos_xy"][env_ids] = local_pos + state["local_vel_xy"][env_ids] * float(motion_dt)
    state["local_pos_xy"][env_ids] = torch.clamp(state["local_pos_xy"][env_ids], min=-8.0, max=8.0)
    _write_dynamic_obstacles_to_sim(env, env_ids)


class UniDiffDriveAction(mdp.actions.JointVelocityAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ros_params = DashGoROSParams.from_yaml()
        self.wheel_radius = ros_params.wheel_radius
        self.track_width = ros_params.wheel_track
        self.prev_actions = None
        self.max_accel_lin = MOTION_CONFIG["max_accel_lin"]
        self.max_accel_ang = MOTION_CONFIG["max_accel_ang"]
        self.control_dt = float(env.cfg.sim.dt * env.cfg.decimation)

    def process_actions(self, actions: torch.Tensor, *args, **kwargs):
        max_lin_vel = MOTION_CONFIG["max_lin_vel"]
        max_reverse_speed = MOTION_CONFIG["max_reverse_speed"]
        max_ang_vel = MOTION_CONFIG["max_ang_vel"]

        target_v = torch.where(actions[:, 0] >= 0.0, actions[:, 0] * max_lin_vel, actions[:, 0] * max_reverse_speed)
        target_v = torch.clamp(target_v, -max_reverse_speed, max_lin_vel)
        target_w = torch.clamp(actions[:, 1] * max_ang_vel, -max_ang_vel, max_ang_vel)

        if self.prev_actions is not None:
            delta_v = target_v - self.prev_actions[:, 0]
            delta_w = target_w - self.prev_actions[:, 1]
            max_delta_v = self.max_accel_lin * self.control_dt
            max_delta_w = self.max_accel_ang * self.control_dt
            target_v = self.prev_actions[:, 0] + torch.clamp(delta_v, -max_delta_v, max_delta_v)
            target_w = self.prev_actions[:, 1] + torch.clamp(delta_w, -max_delta_w, max_delta_w)

        self.prev_actions = torch.stack([target_v, target_w], dim=-1).clone()

        v_left = (target_v - target_w * self.track_width * 0.5) / self.wheel_radius
        v_right = (target_v + target_w * self.track_width * 0.5) / self.wheel_radius
        max_wheel_vel = MOTION_CONFIG["max_wheel_vel"]
        joint_actions = torch.stack(
            [torch.clamp(v_left, -max_wheel_vel, max_wheel_vel), torch.clamp(v_right, -max_wheel_vel, max_wheel_vel)],
            dim=-1,
        )
        return super().process_actions(joint_actions, *args, **kwargs)


def _get_command_target_pos_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command_term = env.command_manager._terms[command_name]
    if hasattr(command_term, "goal_pose_w"):
        return command_term.goal_pose_w[:, :3]
    if hasattr(command_term, "pose_command_w"):
        return command_term.pose_command_w[:, :3]
    return env.command_manager.get_command(command_name)[:, :3]


def _get_command_waypoint_pos_w(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    command_term = env.command_manager._terms[command_name]
    if hasattr(command_term, "get_waypoint_pose_w"):
        return command_term.get_waypoint_pose_w(asset_cfg.name)
    return _get_command_target_pos_w(env, command_name)


def _get_target_delta_and_heading(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_kind: str = "goal",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene[asset_cfg.name]
    if target_kind == "waypoint":
        target_pos_w = _get_command_waypoint_pos_w(env, command_name, asset_cfg)
    else:
        target_pos_w = _get_command_target_pos_w(env, command_name)
    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    delta_pos_w = target_pos_w[:, :2] - robot_pos
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    robot_yaw = torch.nan_to_num(robot_yaw, nan=0.0, posinf=0.0, neginf=0.0)
    target_angle = torch.atan2(delta_pos_w[:, 1], delta_pos_w[:, 0])
    angle_error = wrap_to_pi(target_angle - robot_yaw)
    return delta_pos_w, target_angle, angle_error


def _encode_goal_vector(dist: torch.Tensor, angle_error: torch.Tensor, max_distance: float) -> torch.Tensor:
    clipped_dist = torch.clamp(dist, 0.0, max_distance) / max_distance
    return torch.stack([clipped_dist, torch.sin(angle_error), torch.cos(angle_error)], dim=-1)


def obs_waypoint_vector(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    delta_pos_w, _, angle_error = _get_target_delta_and_heading(env, command_name, asset_cfg, target_kind="waypoint")
    dist = torch.norm(delta_pos_w, dim=-1)
    return torch.nan_to_num(_encode_goal_vector(dist, angle_error, OBSERVATION_CONFIG["waypoint_distance"]), nan=0.0)


def obs_goal_vector(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    delta_pos_w, _, angle_error = _get_target_delta_and_heading(env, command_name, asset_cfg, target_kind="goal")
    dist = torch.norm(delta_pos_w, dim=-1)
    return torch.nan_to_num(_encode_goal_vector(dist, angle_error, OBSERVATION_CONFIG["max_distance"]), nan=0.0)


def obs_forward_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return torch.nan_to_num(env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0], nan=0.0).unsqueeze(-1)


def obs_yaw_rate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return torch.nan_to_num(env.scene[asset_cfg.name].data.root_ang_vel_b[:, 2], nan=0.0).unsqueeze(-1)


def _get_lidar_scan_m(env: ManagerBasedRLEnv) -> torch.Tensor:
    lidar = env.scene["lidar"]
    hits = lidar.data.ray_hits_w
    starts = lidar.data.pos_w.unsqueeze(1)
    scan = torch.linalg.norm(hits - starts, dim=-1)
    scan = torch.nan_to_num(scan, nan=SIM_LIDAR_MAX_RANGE, posinf=SIM_LIDAR_MAX_RANGE, neginf=0.0)
    return torch.clamp(scan, min=0.0, max=SIM_LIDAR_MAX_RANGE)


def process_forward_lidar(env: ManagerBasedRLEnv) -> torch.Tensor:
    scan = _get_lidar_scan_m(env)
    front_centered_scan = torch.roll(scan, shifts=-(scan.shape[1] // 2), dims=1)
    return front_centered_scan / SIM_LIDAR_MAX_RANGE


def process_stitched_lidar(env: ManagerBasedRLEnv) -> torch.Tensor:
    return process_forward_lidar(env)


def _get_min_obstacle_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.min(_get_lidar_scan_m(env), dim=1).values


def reward_distance_tracking_potential(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_kind: str = "goal",
) -> torch.Tensor:
    if target_kind == "waypoint":
        target_pos = _get_command_waypoint_pos_w(env, command_name, asset_cfg)[:, :2]
    else:
        target_pos = _get_command_target_pos_w(env, command_name)[:, :2]
    robot_pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    current_dist = torch.norm(target_pos - robot_pos, dim=-1)
    delta_pos = target_pos - robot_pos
    dist_vec = delta_pos / (current_dist.unsqueeze(-1) + OBSERVATION_CONFIG["epsilon"])
    lin_vel_w = torch.nan_to_num(env.scene[asset_cfg.name].data.root_lin_vel_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(torch.sum(lin_vel_w * dist_vec, dim=-1), -10.0, 10.0)


def reward_navrl_survival_bias(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def reward_navrl_goal_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return reward_distance_tracking_potential(env, command_name, asset_cfg, target_kind="goal")


def reward_navrl_waypoint_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return reward_distance_tracking_potential(env, command_name, asset_cfg, target_kind="waypoint")


def reward_navrl_goal_reached_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    speed_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return check_reach_goal(
        env,
        command_name=command_name,
        threshold=threshold,
        speed_threshold=speed_threshold,
        asset_cfg=asset_cfg,
    ).float()


def reward_navrl_static_safety(env: ManagerBasedRLEnv) -> torch.Tensor:
    clearance = torch.clamp(_get_lidar_scan_m(env), min=1.0e-6, max=SIM_LIDAR_MAX_RANGE)
    return torch.nan_to_num(torch.log(clearance).mean(dim=1), nan=0.0, posinf=0.0, neginf=0.0)


def _dynamic_obstacle_payload(env: ManagerBasedRLEnv, max_tokens: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    collection = env.scene["dynamic_obstacles"]
    state = _ensure_dynamic_obstacle_state(env, collection.num_objects)
    robot = env.scene["robot"]
    robot_xy = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    robot_yaw = torch.nan_to_num(robot_yaw, nan=0.0, posinf=0.0, neginf=0.0)
    robot_vel_world = torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)

    object_state = collection.data.object_link_state_w
    obstacle_pos = torch.nan_to_num(object_state[..., :2], nan=0.0, posinf=0.0, neginf=0.0)
    obstacle_vel = torch.nan_to_num(object_state[..., 7:9], nan=0.0, posinf=0.0, neginf=0.0)
    active_mask = state["active_mask"]
    size_xy = state["size_xy"]
    shape = state["shape"]

    delta_world = obstacle_pos - robot_xy.unsqueeze(1)
    delta_local = _rotate_world_to_local(delta_world, robot_yaw.unsqueeze(1))
    rel_vel_world = obstacle_vel - robot_vel_world.unsqueeze(1)
    rel_vel_local = _rotate_world_to_local(rel_vel_world, robot_yaw.unsqueeze(1))
    center_distance = torch.norm(delta_local, dim=-1)
    half_extent = torch.max(size_xy, dim=-1).values * 0.5
    clearance = torch.clamp(center_distance - half_extent, min=1.0e-6, max=SIM_LIDAR_MAX_RANGE)

    order_metric = torch.where(active_mask, center_distance, torch.full_like(center_distance, 1.0e6))
    order = torch.argsort(order_metric, dim=1)
    topk = min(max_tokens, collection.num_objects)
    gather_idx = order[:, :topk]
    batch_idx = torch.arange(env.num_envs, device=env.device).unsqueeze(1)

    rel_xy = delta_local[batch_idx, gather_idx]
    rel_vel = rel_vel_local[batch_idx, gather_idx]
    dist = center_distance[batch_idx, gather_idx].unsqueeze(-1)
    size_sel = size_xy[batch_idx, gather_idx]
    active_sel = active_mask[batch_idx, gather_idx].float().unsqueeze(-1)
    shape_sel = shape[batch_idx, gather_idx].unsqueeze(-1)
    dynamic_sel = active_sel.clone()
    tokens = torch.cat([rel_xy, dist, rel_vel, size_sel, dynamic_sel, shape_sel, active_sel], dim=-1)
    tokens = tokens * active_sel

    if topk < max_tokens:
        padding = torch.zeros((env.num_envs, max_tokens - topk, 10), device=env.device)
        tokens = torch.cat([tokens, padding], dim=1)

    return tokens, clearance, active_mask


def build_dynamic_obstacle_tokens(env: ManagerBasedRLEnv, max_tokens: int = 5) -> torch.Tensor:
    tokens, _, _ = _dynamic_obstacle_payload(env, max_tokens=max_tokens)
    return tokens


def reward_navrl_dynamic_safety(env: ManagerBasedRLEnv) -> torch.Tensor:
    _, clearance, active_mask = _dynamic_obstacle_payload(env)
    log_clearance = torch.log(torch.clamp(clearance, min=1.0e-6, max=SIM_LIDAR_MAX_RANGE))
    safe_log = torch.where(active_mask, log_clearance, torch.zeros_like(log_clearance))
    denom = active_mask.float().sum(dim=1).clamp_min(1.0)
    reward = safe_log.sum(dim=1) / denom
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def penalty_navrl_twist_smoothness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    twist = torch.stack(
        [
            torch.nan_to_num(robot.data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(robot.data.root_ang_vel_b[:, 2], nan=0.0, posinf=0.0, neginf=0.0),
        ],
        dim=-1,
    )
    if not hasattr(env, "_navrl_prev_twist"):
        env._navrl_prev_twist = twist.detach().clone()
        return torch.zeros(env.num_envs, device=env.device)
    delta = torch.norm(twist - env._navrl_prev_twist, dim=-1)
    env._navrl_prev_twist = twist.detach().clone()
    delta = torch.where(env.episode_length_buf < 2, torch.zeros_like(delta), delta)
    return torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)


def penalty_navrl_progress_stall(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_kind: str = "waypoint",
    activation_distance: float = 0.75,
    min_progress: float = 0.005,
    max_forward_speed: float = 0.05,
    warmup_steps: int = 15,
    trigger_steps: int = 8,
) -> torch.Tensor:
    if target_kind == "waypoint":
        target_pos = _get_command_waypoint_pos_w(env, command_name, asset_cfg)[:, :2]
    else:
        target_pos = _get_command_target_pos_w(env, command_name)[:, :2]

    robot = env.scene[asset_cfg.name]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(target_pos - robot_pos, dim=-1)
    forward_speed = torch.abs(torch.nan_to_num(robot.data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0))

    if not hasattr(env, "_navrl_prev_target_distance"):
        env._navrl_prev_target_distance = dist.detach().clone()
    if not hasattr(env, "_navrl_stall_counts"):
        env._navrl_stall_counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    env._navrl_stall_counts[env.episode_length_buf < 2] = 0
    prev_dist = env._navrl_prev_target_distance
    progress = prev_dist - dist
    env._navrl_prev_target_distance = dist.detach().clone()

    stalled = (
        (env.episode_length_buf > warmup_steps)
        & (dist > activation_distance)
        & (progress < min_progress)
        & (forward_speed < max_forward_speed)
    )
    env._navrl_stall_counts = torch.where(
        stalled,
        env._navrl_stall_counts + 1,
        torch.zeros_like(env._navrl_stall_counts),
    )
    penalty = torch.clamp(
        (env._navrl_stall_counts.float() - float(trigger_steps)) / float(max(trigger_steps, 1)),
        min=0.0,
        max=1.0,
    )
    return torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)


def penalty_navrl_orbiting(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_kind: str = "waypoint",
    activation_distance: float = 0.75,
    min_progress: float = 0.01,
    min_angular_speed: float = 0.35,
    max_forward_speed: float = 0.18,
    warmup_steps: int = 20,
    trigger_steps: int = 10,
) -> torch.Tensor:
    if target_kind == "waypoint":
        target_pos = _get_command_waypoint_pos_w(env, command_name, asset_cfg)[:, :2]
    else:
        target_pos = _get_command_target_pos_w(env, command_name)[:, :2]

    robot = env.scene[asset_cfg.name]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(target_pos - robot_pos, dim=-1)
    forward_speed = torch.abs(torch.nan_to_num(robot.data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0))
    angular_speed = torch.abs(torch.nan_to_num(robot.data.root_ang_vel_b[:, 2], nan=0.0, posinf=0.0, neginf=0.0))

    if not hasattr(env, "_navrl_orbit_prev_target_distance"):
        env._navrl_orbit_prev_target_distance = dist.detach().clone()
    if not hasattr(env, "_navrl_orbit_counts"):
        env._navrl_orbit_counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    env._navrl_orbit_counts[env.episode_length_buf < 2] = 0
    prev_dist = env._navrl_orbit_prev_target_distance
    progress = prev_dist - dist
    env._navrl_orbit_prev_target_distance = dist.detach().clone()

    orbiting = (
        (env.episode_length_buf > warmup_steps)
        & (dist > activation_distance)
        & (progress < min_progress)
        & (angular_speed > min_angular_speed)
        & (forward_speed < max_forward_speed)
    )
    env._navrl_orbit_counts = torch.where(
        orbiting,
        env._navrl_orbit_counts + 1,
        torch.zeros_like(env._navrl_orbit_counts),
    )
    penalty = torch.clamp(
        (env._navrl_orbit_counts.float() - float(trigger_steps)) / float(max(trigger_steps, 1)),
        min=0.0,
        max=1.0,
    )
    speed_scale = torch.clamp(
        (angular_speed - min_angular_speed) / max(MOTION_CONFIG["max_ang_vel"] - min_angular_speed, 1.0e-3),
        min=0.0,
        max=1.0,
    )
    return torch.nan_to_num(penalty * speed_scale, nan=0.0, posinf=0.0, neginf=0.0)


def log_distance_to_goal(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    target_pos = _get_command_target_pos_w(env, command_name)[:, :2]
    robot_pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.norm(target_pos - robot_pos, dim=-1)


def log_linear_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    lin_vel_b = torch.nan_to_num(env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.norm(lin_vel_b, dim=-1)


def check_collision_navrl_style(env: ManagerBasedRLEnv, static_threshold: float, dynamic_threshold: float) -> torch.Tensor:
    static_collision = _get_min_obstacle_distance(env) <= static_threshold
    _, clearance, active_mask = _dynamic_obstacle_payload(env)
    dynamic_collision = torch.any(active_mask & (clearance <= dynamic_threshold), dim=1)
    return static_collision | dynamic_collision


def check_reach_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    speed_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    target_pos_w = _get_command_target_pos_w(env, command_name)[:, :2]
    robot_pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    lin_vel_b = torch.nan_to_num(env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(target_pos_w - robot_pos, dim=-1)
    speed = torch.norm(lin_vel_b, dim=-1)
    return (dist < threshold) & (speed < speed_threshold)


def check_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def reset_root_state_safe_donut(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    min_radius: float,
    max_radius: float,
    asset_cfg: SceneEntityCfg,
) -> None:
    env_ids = _resolve_event_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    asset = env.scene[asset_cfg.name]
    min_r2 = min_radius**2
    max_r2 = max_radius**2
    r_sq = torch.rand(len(env_ids), device=env.device) * (max_r2 - min_r2) + min_r2
    radius = torch.sqrt(r_sq)
    theta = torch.rand(len(env_ids), device=env.device) * (2.0 * math.pi) - math.pi

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 0] = env.scene.env_origins[env_ids, 0] + radius * torch.cos(theta)
    root_state[:, 1] = env.scene.env_origins[env_ids, 1] + radius * torch.sin(theta)
    root_state[:, 2] = 0.20
    random_yaw = torch.rand(len(env_ids), device=env.device) * (2.0 * math.pi) - math.pi
    zeros = torch.zeros_like(random_yaw)
    root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, random_yaw)
    root_state[:, 7:] = 0.0
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)


class RelativeNavRLTargetCommand(mdp.UniformPoseCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.min_dist = GOAL_DISTANCE_RANGE[0]
        self.max_dist = GOAL_DISTANCE_RANGE[1]
        self.max_path_points = OBSERVATION_CONFIG["max_path_points"]
        self.waypoint_lookahead_steps = max(
            1,
            int(math.ceil(OBSERVATION_CONFIG["waypoint_distance"] / OBSERVATION_CONFIG["path_resolution"])),
        )
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.goal_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.waypoint_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 3] = 1.0
        self.goal_pose_w[:, 3] = 1.0
        self.waypoint_pose_w[:, 3] = 1.0
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.reference_path_w = torch.zeros(self.num_envs, self.max_path_points, 3, device=self.device)
        self.reference_path_len = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.reference_path_cursor = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _build_linear_reference_path(self, start_xy: torch.Tensor, goal_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = start_xy.shape[0]
        delta = goal_xy - start_xy
        dist = torch.norm(delta, dim=-1)
        steps = torch.clamp(
            torch.ceil(dist / OBSERVATION_CONFIG["path_resolution"]).long() + 1,
            min=2,
            max=self.max_path_points,
        )
        scaled_t = build_reference_path_progress(self.max_path_points, steps)
        interp_xy = start_xy.unsqueeze(1) + delta.unsqueeze(1) * scaled_t.unsqueeze(-1)
        path = torch.zeros(batch_size, self.max_path_points, 3, device=self.device)
        path[:, :, :2] = interp_xy
        headings = torch.atan2(delta[:, 1], delta[:, 0]).unsqueeze(-1).expand(-1, self.max_path_points)
        path[:, :, 2] = headings
        return path, steps

    def get_waypoint_pose_w(self, asset_name: str = "robot") -> torch.Tensor:
        robot = self._env.scene[asset_name]
        robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
        path_xy = self.reference_path_w[:, :, :2]
        distances = torch.norm(path_xy - robot_pos.unsqueeze(1), dim=-1)
        mask = torch.arange(self.max_path_points, device=self.device).unsqueeze(0) < self.reference_path_len.unsqueeze(1)
        masked_distances = torch.where(mask, distances, torch.full_like(distances, 1.0e6))
        nearest_idx = torch.argmin(masked_distances, dim=1)
        self.reference_path_cursor = torch.maximum(self.reference_path_cursor, nearest_idx)
        selected_idx = compute_waypoint_lookahead_indices(
            self.reference_path_cursor,
            self.reference_path_len,
            self.waypoint_lookahead_steps,
        )
        selected = self.reference_path_w[torch.arange(self.num_envs, device=self.device), selected_idx]
        self.waypoint_pose_w[:, :3] = selected
        self.waypoint_pose_w[:, 3] = 1.0
        self.waypoint_pose_w[:, 4:] = 0.0
        return self.waypoint_pose_w

    def _resample_command(self, env_ids: torch.Tensor):
        robot = self._env.scene[self.cfg.asset_name]
        robot_pos = torch.nan_to_num(robot.data.root_pos_w[env_ids, :3], nan=0.0, posinf=0.0, neginf=0.0)
        radius = torch.empty(len(env_ids), device=self.device).uniform_(self.min_dist, self.max_dist)
        theta = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        goal_xy = torch.stack([robot_pos[:, 0] + radius * torch.cos(theta), robot_pos[:, 1] + radius * torch.sin(theta)], dim=-1)

        self.goal_pose_w[env_ids, 0] = goal_xy[:, 0]
        self.goal_pose_w[env_ids, 1] = goal_xy[:, 1]
        self.goal_pose_w[env_ids, 2] = 0.0
        self.goal_pose_w[env_ids, 3] = 1.0
        self.goal_pose_w[env_ids, 4:] = 0.0
        self.pose_command_w[env_ids] = self.goal_pose_w[env_ids]
        self.heading_command_w[env_ids] = 0.0

        path, steps = self._build_linear_reference_path(robot_pos[:, :2], goal_xy)
        self.reference_path_w[env_ids] = path
        self.reference_path_len[env_ids] = steps
        self.reference_path_cursor[env_ids] = 0
        self.waypoint_pose_w[env_ids] = self.goal_pose_w[env_ids]

    def _update_metrics(self):
        robot = self._env.scene[self.cfg.asset_name]
        root_pos_w = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
        target_pos_w = self.goal_pose_w[:, :3]
        pos_error = torch.norm(target_pos_w[:, :2] - root_pos_w[:, :2], dim=-1)
        _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        robot_yaw = torch.nan_to_num(robot_yaw, nan=0.0, posinf=0.0, neginf=0.0)
        delta_pos = target_pos_w - root_pos_w
        target_yaw = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])
        rot_error = wrap_to_pi(target_yaw - robot_yaw)
        self.metrics["position_error"] = pos_error
        self.metrics["orientation_error"] = torch.abs(rot_error)

    def _update_debug_vis(self, *args, **kwargs):
        pass


@configclass
class UniDiffDriveActionCfg(mdp.actions.JointVelocityActionCfg):
    class_type = UniDiffDriveAction
    asset_name: str = "robot"
    joint_names: list[str] = ["left_wheel_joint", "right_wheel_joint"]
    scale: float = 1.0
    use_default_offset: bool = False


@configclass
class RelativeNavRLTargetCommandCfg(mdp.UniformPoseCommandCfg):
    class_type = RelativeNavRLTargetCommand
    asset_name: str = "robot"
    body_name: str = "base_link"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    ranges: mdp.UniformPoseCommandCfg.Ranges = mdp.UniformPoseCommandCfg.Ranges(
        pos_x=(-4.0, 4.0),
        pos_y=(-4.0, 4.0),
        pos_z=(0.0, 0.0),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(-math.pi, math.pi),
    )
    debug_vis: bool = False


@configclass
class DashgoActionsOfficialCfg:
    wheels = UniDiffDriveActionCfg()


@configclass
class DashgoCommandsOfficialCfg:
    target_pose = RelativeNavRLTargetCommandCfg()


@configclass
class DashgoObservationsOfficialCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        history_length = 3
        concatenate_terms = True
        flatten_history_dim = True

        lidar = ObservationTermCfg(func=process_forward_lidar, params={})
        waypoint_vector = ObservationTermCfg(
            func=obs_waypoint_vector,
            params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")},
        )
        goal_vector = ObservationTermCfg(
            func=obs_goal_vector,
            params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")},
        )
        lin_vel_x = ObservationTermCfg(func=obs_forward_velocity, params={"asset_cfg": SceneEntityCfg("robot")})
        yaw_rate = ObservationTermCfg(func=obs_yaw_rate, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False

    policy = PolicyCfg()


@configclass
class DashgoEventsOfficialCfg:
    reset_base = EventTermCfg(
        func=reset_root_state_safe_donut,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "min_radius": 0.2, "max_radius": 0.6},
    )
    configure_dynamic_obstacles = EventTermCfg(
        func=configure_dynamic_obstacles,
        mode="reset",
        params={"num_active": DEFAULT_DYNAMIC_OBSTACLE_COUNT},
    )
    drive_dynamic_obstacles = EventTermCfg(
        func=animate_dynamic_obstacles,
        mode="interval",
        interval_range_s=(DYNAMIC_OBSTACLE_INTERVAL_S, DYNAMIC_OBSTACLE_INTERVAL_S),
        params={"motion_dt": DYNAMIC_OBSTACLE_INTERVAL_S},
    )


@configclass
class DashgoSceneOfficialCfg(InteractiveSceneCfg):
    terrain = build_navrl_terrain_cfg(24)
    robot = DASHGO_D1_CFG.replace(prim_path="{ENV_REGEX_NS}/Dashgo")
    dynamic_obstacles = build_dynamic_obstacle_collection_cfg()
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link",
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.13)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=360.0 / float(SIM_LIDAR_POLICY_DIM),
        ),
        debug_vis=False,
        max_distance=SIM_LIDAR_MAX_RANGE,
        mesh_prim_paths=["/World/ground"],
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.95, 0.95, 0.95)),
    )
    sun_light = AssetBaseCfg(
        prim_path="/World/sunLight",
        spawn=sim_utils.DistantLightCfg(intensity=1600.0, color=(0.95, 0.95, 0.95), angle=0.53),
    )


@configclass
class DashgoRewardsOfficialCfg:
    navrl_survival = RewardTermCfg(func=reward_navrl_survival_bias, weight=REWARD_CONFIG["survival_weight"])
    navrl_goal_reached_bonus = RewardTermCfg(
        func=reward_navrl_goal_reached_bonus,
        weight=REWARD_CONFIG["goal_reached_bonus_weight"],
        params={
            "command_name": "target_pose",
            "threshold": REWARD_CONFIG["goal_termination_threshold"],
            "speed_threshold": REWARD_CONFIG["goal_stop_velocity"],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    navrl_goal_velocity = RewardTermCfg(
        func=reward_navrl_goal_velocity,
        weight=REWARD_CONFIG["goal_velocity_weight"],
        params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")},
    )
    navrl_waypoint_velocity = RewardTermCfg(
        func=reward_navrl_waypoint_velocity,
        weight=REWARD_CONFIG["waypoint_velocity_weight"],
        params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")},
    )
    navrl_static_safety = RewardTermCfg(func=reward_navrl_static_safety, weight=REWARD_CONFIG["static_safety_weight"])
    navrl_dynamic_safety = RewardTermCfg(func=reward_navrl_dynamic_safety, weight=REWARD_CONFIG["dynamic_safety_weight"])
    navrl_twist_smoothness = RewardTermCfg(
        func=penalty_navrl_twist_smoothness,
        weight=REWARD_CONFIG["twist_smoothness_weight"],
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    navrl_progress_stall = RewardTermCfg(
        func=penalty_navrl_progress_stall,
        weight=REWARD_CONFIG["progress_stall_weight"],
        params={
            "command_name": "target_pose",
            "asset_cfg": SceneEntityCfg("robot"),
            "target_kind": "waypoint",
            "activation_distance": REWARD_CONFIG["stall_activation_distance"],
            "min_progress": REWARD_CONFIG["stall_min_progress"],
            "max_forward_speed": REWARD_CONFIG["stall_max_forward_speed"],
            "warmup_steps": REWARD_CONFIG["stall_warmup_steps"],
            "trigger_steps": REWARD_CONFIG["stall_trigger_steps"],
        },
    )
    navrl_orbit = RewardTermCfg(
        func=penalty_navrl_orbiting,
        weight=REWARD_CONFIG["orbit_weight"],
        params={
            "command_name": "target_pose",
            "asset_cfg": SceneEntityCfg("robot"),
            "target_kind": "waypoint",
            "activation_distance": REWARD_CONFIG["orbit_activation_distance"],
            "min_progress": REWARD_CONFIG["orbit_min_progress"],
            "min_angular_speed": REWARD_CONFIG["orbit_min_angular_speed"],
            "max_forward_speed": REWARD_CONFIG["orbit_max_forward_speed"],
            "warmup_steps": REWARD_CONFIG["orbit_warmup_steps"],
            "trigger_steps": REWARD_CONFIG["orbit_trigger_steps"],
        },
    )
    log_distance = RewardTermCfg(
        func=log_distance_to_goal,
        weight=0.0,
        params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")},
    )
    log_velocity = RewardTermCfg(func=log_linear_velocity, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class DashgoTerminationsOfficialCfg:
    time_out = TerminationTermCfg(func=check_time_out, time_out=True)
    reach_goal = TerminationTermCfg(
        func=check_reach_goal,
        params={
            "command_name": "target_pose",
            "threshold": REWARD_CONFIG["goal_termination_threshold"],
            "speed_threshold": REWARD_CONFIG["goal_stop_velocity"],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    object_collision = TerminationTermCfg(
        func=check_collision_navrl_style,
        params={
            "static_threshold": REWARD_CONFIG["static_collision_threshold"],
            "dynamic_threshold": REWARD_CONFIG["dynamic_collision_threshold"],
        },
    )


@configclass
class DashgoNavOfficialEnvCfg(ManagerBasedRLEnvCfg):
    decimation = 3
    episode_length_s = 90.0
    scene = DashgoSceneOfficialCfg(num_envs=16, env_spacing=15.0, lazy_sensor_update=False)
    sim = sim_utils.SimulationCfg(
        dt=1 / 60,
        render_interval=10,
        render=RenderCfg(
            antialiasing_mode="Off",
            enable_direct_lighting=False,
            enable_shadows=False,
            enable_ambient_occlusion=False,
            samples_per_pixel=1,
        ),
    )
    viewer = ViewerCfg(
        eye=(4.5, 4.5, 3.0),
        lookat=(0.0, 0.0, 0.3),
        resolution=(1920, 1080),
        origin_type="world",
        env_index=0,
    )
    actions = DashgoActionsOfficialCfg()
    observations = DashgoObservationsOfficialCfg()
    commands = DashgoCommandsOfficialCfg()
    events = DashgoEventsOfficialCfg()
    rewards = DashgoRewardsOfficialCfg()
    terminations = DashgoTerminationsOfficialCfg()
    curriculum = None
