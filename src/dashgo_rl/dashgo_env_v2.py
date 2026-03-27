import torch
import math
import sys
import os
import json
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg, mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera, ContactSensor, ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg, EventTermCfg, CurriculumTermCfg
from isaaclab.sim.simulation_cfg import RenderCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat, quat_from_euler_xyz
# [架构师V3.4最终版] 0.46.x版本专用：Hf前缀类名
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg

# Isaac Lab 0.46.x 使用 Hf 前缀（Height Field的缩写）
from isaaclab.terrains.height_field import (
    HfTerrainBaseCfg,              # 平地（替代MeshPlaneTerrainCfg）
    HfRandomUniformTerrainCfg,     # 随机障碍（替代MoundsTerrainCfg）
    HfDiscreteObstaclesTerrainCfg, # 迷宫（保持原名）
)

TERRAIN_GEN_AVAILABLE = True
from .dashgo_assets import DASHGO_D1_CFG
from .dashgo_config import DashGoROSParams  # 新增: 导入ROS参数配置类

def _get_env_float(name: str, default: float) -> float:
    """读取环境变量浮点配置，解析失败时回退默认值。"""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

# =============================================================================
# 训练/运动参数常量
# 说明：
# - 训练主配置的唯一来源是 `configs/training/train_cfg_v2.yaml`
# - 这里保留的常量只用于环境内部计算或历史兼容
# - 若数值与 YAML 不一致，优先修正到与 YAML 对齐
# =============================================================================

# PPO训练参数
PPO_CONFIG = {
    "seed": 42,
    "num_steps_per_env": 24,
    "num_mini_batches": 4,
    "entropy_coef": 0.005,
    "max_iterations": 9000,
    "save_interval": 100,
}

# 神经网络架构参数
NETWORK_CONFIG = {
    "init_noise_std": 1.0,
    "actor_hidden_dims": [128, 64],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu",
}

# PPO算法参数
ALGORITHM_CONFIG = {
    "value_loss_coef": 1.0,
    "clip_param": 0.2,
    "num_learning_epochs": 3,
    "learning_rate": 1.5e-4,
    "max_grad_norm": 1.0,
    "gamma": 0.99,
    "lam": 0.95,
    "desired_kl": 0.005,
}

# 机器人运动参数（来自ROS配置）
MOTION_CONFIG = {
    "max_lin_vel": 0.3,       # 最大线速度 (m/s，来自ROS max_vel_x)
    "max_reverse_speed": 0.15,  # 最大倒车速度 (m/s)
    "max_ang_vel": 1.0,       # 最大角速度 (rad/s，来自ROS max_rot_vel)
    "max_accel_lin": 1.0,     # 最大线加速度 (m/s²)
    "max_accel_ang": 0.6,     # 最大角加速度 (rad/s²)
    "max_wheel_vel": 5.0,     # 最大轮速
}

# 奖励函数参数（NavRL 风格主干 + 少量 DashGo 适配）
REWARD_CONFIG = {
    "terminal_reward": 0.0,
    "goal_reward_threshold": 0.5,
    "goal_termination_threshold": 0.5,
    "goal_stop_velocity": 0.08,
    "progress_weight": 1.0,
    "target_speed": 0.22,
    "facing_threshold": 0.75,
    "high_speed_threshold": 0.18,
    "high_speed_reward": 0.1,
    "safe_distance": 0.35,
    "obstacle_penalty_threshold": 0.60,
    "collision_decay": 4.0,
    "facing_reward_scale": 0.3,
    "facing_angle_scale": 0.7,
    "alive_penalty": 1.0,
    "reward_clip_min": -20.0,
    "reward_clip_max": 120.0,
    "navrl_dynamic_obstacle_half_extent": 0.10,
    "reverse_escape_term_weight": _get_env_float("DASHGO_REVERSE_ESCAPE_WEIGHT", 0.0),
    "reverse_escape_front_blocked": _get_env_float("DASHGO_REVERSE_ESCAPE_FRONT_BLOCKED", 0.55),
    "reverse_escape_rear_clear": _get_env_float("DASHGO_REVERSE_ESCAPE_REAR_CLEAR", 0.80),
    "reverse_escape_progress_threshold": _get_env_float("DASHGO_REVERSE_ESCAPE_PROGRESS_THRESHOLD", 0.02),
    "reverse_escape_ang_penalty": _get_env_float("DASHGO_REVERSE_ESCAPE_ANG_PENALTY", 0.10),
    "progress_stall_term_weight": _get_env_float("DASHGO_PROGRESS_STALL_WEIGHT", 3.5),
    "orbit_term_weight": _get_env_float("DASHGO_ORBIT_WEIGHT", 3.0),
    "orbit_activation_distance": _get_env_float("DASHGO_ORBIT_ACTIVATION_DISTANCE", 0.75),
    "orbit_min_progress": _get_env_float("DASHGO_ORBIT_MIN_PROGRESS", 0.01),
    "orbit_min_angular_speed": _get_env_float("DASHGO_ORBIT_MIN_ANGULAR_SPEED", 0.35),
    "orbit_max_forward_speed": _get_env_float("DASHGO_ORBIT_MAX_FORWARD_SPEED", 0.18),
    "orbit_trigger_steps": _get_env_float("DASHGO_ORBIT_TRIGGER_STEPS", 10.0),
}

# 观测处理参数
OBSERVATION_CONFIG = {
    "max_distance": 8.0,   # 最大距离截断 (m，防止数值溢出)
    "waypoint_distance": 1.0,
    "epsilon": 1e-6,       # 数值稳定性epsilon（防止除零）
    "lookahead_min_forward": 0.6,
    "lookahead_max_forward": 1.2,
    "lookahead_gain_forward": 3.0,
    "lookahead_min_reverse": 0.45,
    "lookahead_max_reverse": 0.8,
    "lookahead_gain_reverse": 2.0,
    "path_resolution": 0.2,
    "max_path_points": 48,
}

AUTOPILOT_PROFILE = os.environ.get("DASHGO_AUTOPILOT_PROFILE", "").strip().lower()
USE_AUTOPILOT_FLAT_SCENE = AUTOPILOT_PROFILE in {"gen1", "gen2", "autopilot"}
USE_AUTOPILOT_GEN1_EASY_RESET = AUTOPILOT_PROFILE in {"gen1", "autopilot"}
USE_AUTOPILOT_GEN2_DYNAMIC = AUTOPILOT_PROFILE == "gen2"
CURRICULUM_TRACE_PATH = os.environ.get("DASHGO_CURRICULUM_TRACE_PATH", "").strip()

INITIAL_TARGET_MAX_DIST = 3.0 if USE_AUTOPILOT_GEN2_DYNAMIC else 1.0
INITIAL_CURRICULUM_DIST = 3.0 if USE_AUTOPILOT_GEN2_DYNAMIC else 1.0
DYNAMIC_OBSTACLE_ASSET_NAMES = ("obs_inner_1", "obs_inner_3", "obs_inner_5")
DYNAMIC_OBSTACLE_PROFILE_NAMES = ("crossing", "head_on", "stop_go")
DYNAMIC_OBSTACLE_INTERVAL_S = 0.10
RECOVERY_SCENARIO_CONFIG = {
    "enabled": USE_AUTOPILOT_GEN2_DYNAMIC,
    "probability": _get_env_float("DASHGO_RECOVERY_SCENARIO_PROBABILITY", 0.0),
    "goal_distance_min": 0.8,
    "goal_distance_max": 1.4,
    "goal_theta_min": 0.80 * math.pi,
    "goal_theta_max": math.pi,
    "front_blocker_x": 0.62,
    "front_blocker_y": 0.32,
    "front_cap_x": 1.12,
}

SIM_LIDAR_MAX_RANGE = 12.0
SIM_LIDAR_POLICY_DIM = 72


def append_curriculum_trace(payload: dict) -> None:
    """按需写入课程学习追踪，默认关闭。"""
    if not CURRICULUM_TRACE_PATH:
        return
    try:
        payload = dict(payload)
        payload["profile"] = AUTOPILOT_PROFILE or "default"
        with open(CURRICULUM_TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # 追踪失败不影响训练主流程。
        return


def _resolve_event_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    """统一把事件回调里的 env_ids 转成 LongTensor。"""
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _rotate_local_xy(local_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """将局部二维向量绕 Z 轴旋转到世界平面。"""
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rot_x = local_xy[..., 0] * cos_yaw - local_xy[..., 1] * sin_yaw
    rot_y = local_xy[..., 0] * sin_yaw + local_xy[..., 1] * cos_yaw
    return torch.stack([rot_x, rot_y], dim=-1)


def _ensure_dynamic_obstacle_state(env: ManagerBasedRLEnv, asset_names: tuple[str, ...]) -> dict:
    """按需初始化 Gen2 动态障碍状态缓存。"""
    state = getattr(env, "_dynamic_obstacle_state", None)
    if state is not None:
        if state.get("num_envs") == env.num_envs and tuple(state.get("asset_names", ())) == tuple(asset_names):
            return state

    num_slots = len(asset_names)
    state = {
        "num_envs": env.num_envs,
        "asset_names": tuple(asset_names),
        "active_slot": torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
        "center_xy": torch.zeros((env.num_envs, num_slots, 2), device=env.device),
        "axis_xy": torch.zeros((env.num_envs, num_slots, 2), device=env.device),
        "amplitude": torch.zeros((env.num_envs, num_slots), device=env.device),
        "cycle_rate": torch.zeros((env.num_envs, num_slots), device=env.device),
        "phase": torch.zeros((env.num_envs, num_slots), device=env.device),
        "yaw_w": torch.zeros((env.num_envs, num_slots), device=env.device),
        "height": torch.full((env.num_envs, num_slots), 0.5, device=env.device),
    }
    env._dynamic_obstacle_state = state
    return state


def _ensure_recovery_scenario_state(env: ManagerBasedRLEnv) -> dict:
    """按需初始化“前堵后退”课程状态缓存。"""
    state = getattr(env, "_recovery_scenario_state", None)
    if state is not None and state.get("num_envs") == env.num_envs:
        return state

    state = {
        "num_envs": env.num_envs,
        "active": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
        "goal_distance": torch.zeros(env.num_envs, device=env.device),
        "goal_theta": torch.zeros(env.num_envs, device=env.device),
    }
    env._recovery_scenario_state = state
    return state


def _write_kinematic_obstacle_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_name: str,
    root_xy_w: torch.Tensor,
    yaw_w: torch.Tensor,
    height: float = 0.5,
) -> None:
    """把静态/运动学障碍写到给定世界位姿。"""
    if env_ids.numel() == 0:
        return

    asset = env.scene[asset_name]
    root_pose = asset.data.default_root_state[env_ids, :7].clone()
    root_pose[:, 0:2] = root_xy_w
    root_pose[:, 2] = height
    zeros = torch.zeros_like(yaw_w)
    root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw_w)

    root_velocity = torch.zeros((len(env_ids), 6), device=env.device)
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)


def _compute_stop_go_motion(
    amplitude: torch.Tensor, phase: torch.Tensor, cycle_rate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """停走型障碍：四段式保持/前进/保持/后退。"""
    cycle = torch.remainder(phase / (2.0 * math.pi), 1.0)
    disp_norm = torch.empty_like(cycle)
    speed = torch.zeros_like(cycle)

    seg_hold_back = cycle < 0.25
    disp_norm[seg_hold_back] = -1.0

    seg_forward = (cycle >= 0.25) & (cycle < 0.50)
    if torch.any(seg_forward):
        alpha = (cycle[seg_forward] - 0.25) / 0.25
        disp_norm[seg_forward] = -1.0 + 2.0 * alpha
        speed[seg_forward] = amplitude[seg_forward] * (2.0 / 0.25) * cycle_rate[seg_forward]

    seg_hold_front = (cycle >= 0.50) & (cycle < 0.75)
    disp_norm[seg_hold_front] = 1.0

    seg_backward = cycle >= 0.75
    if torch.any(seg_backward):
        alpha = (cycle[seg_backward] - 0.75) / 0.25
        disp_norm[seg_backward] = 1.0 - 2.0 * alpha
        speed[seg_backward] = -amplitude[seg_backward] * (2.0 / 0.25) * cycle_rate[seg_backward]

    displacement = amplitude * disp_norm
    return displacement, speed


def _write_dynamic_obstacle_slot(env: ManagerBasedRLEnv, env_ids: torch.Tensor, slot_idx: int, state: dict) -> None:
    """把指定槽位的脚本化障碍写回仿真。"""
    if env_ids.numel() == 0:
        return

    asset_name = state["asset_names"][slot_idx]
    asset = env.scene[asset_name]
    amplitude = state["amplitude"][env_ids, slot_idx]
    phase = state["phase"][env_ids, slot_idx]
    cycle_rate = state["cycle_rate"][env_ids, slot_idx]

    if DYNAMIC_OBSTACLE_PROFILE_NAMES[slot_idx] == "stop_go":
        displacement, speed_along_axis = _compute_stop_go_motion(amplitude, phase, cycle_rate)
    else:
        phase_speed = 2.0 * math.pi * cycle_rate
        displacement = amplitude * torch.sin(phase)
        speed_along_axis = amplitude * phase_speed * torch.cos(phase)

    axis_xy = state["axis_xy"][env_ids, slot_idx]
    center_xy = state["center_xy"][env_ids, slot_idx]
    yaw_w = state["yaw_w"][env_ids, slot_idx]
    origins_xy = env.scene.env_origins[env_ids, :2]

    root_pose = asset.data.default_root_state[env_ids, :7].clone()
    root_pose[:, 0:2] = origins_xy + center_xy + axis_xy * displacement.unsqueeze(-1)
    root_pose[:, 2] = state["height"][env_ids, slot_idx]

    zeros = torch.zeros_like(yaw_w)
    root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw_w)

    root_velocity = torch.zeros((len(env_ids), 6), device=env.device)
    root_velocity[:, 0:2] = axis_xy * speed_along_axis.unsqueeze(-1)

    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)


def configure_dynamic_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_names: tuple[str, ...],
) -> None:
    """Gen2 reset 事件：为每个环境选一个动态障碍模板并写入初始状态。"""
    if not USE_AUTOPILOT_GEN2_DYNAMIC:
        return

    env_ids = _resolve_event_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    state = _ensure_dynamic_obstacle_state(env, asset_names)
    state["active_slot"][env_ids] = -1
    state["center_xy"][env_ids] = 0.0
    state["axis_xy"][env_ids] = 0.0
    state["amplitude"][env_ids] = 0.0
    state["cycle_rate"][env_ids] = 0.0
    state["phase"][env_ids] = 0.0
    state["yaw_w"][env_ids] = 0.0

    slot_ids = torch.randint(0, len(asset_names), (len(env_ids),), device=env.device)
    layout_yaw = torch.empty(len(env_ids), device=env.device).uniform_(-math.pi, math.pi)
    state["active_slot"][env_ids] = slot_ids

    templates = (
        {"center": (1.2, 0.0), "axis": (0.0, 1.0), "amp": (0.8, 1.2), "cycle": (0.02, 0.045)},
        {"center": (1.7, 0.0), "axis": (1.0, 0.0), "amp": (0.7, 1.1), "cycle": (0.018, 0.035)},
        {"center": (1.35, 0.0), "axis": (1.0, 0.0), "amp": (0.6, 0.9), "cycle": (0.012, 0.022)},
    )

    for slot_idx, template in enumerate(templates):
        slot_mask = slot_ids == slot_idx
        if not torch.any(slot_mask):
            continue

        slot_env_ids = env_ids[slot_mask]
        yaw = layout_yaw[slot_mask]
        count = len(slot_env_ids)

        local_center = torch.tensor(template["center"], device=env.device).unsqueeze(0).repeat(count, 1)
        local_axis = torch.tensor(template["axis"], device=env.device).unsqueeze(0).repeat(count, 1)
        center_xy = _rotate_local_xy(local_center, yaw)
        axis_xy = _rotate_local_xy(local_axis, yaw)
        axis_xy = axis_xy / torch.clamp(torch.norm(axis_xy, dim=-1, keepdim=True), min=1.0e-6)

        amplitude = torch.empty(count, device=env.device).uniform_(*template["amp"])
        cycle_rate = torch.empty(count, device=env.device).uniform_(*template["cycle"])
        phase = torch.empty(count, device=env.device).uniform_(0.0, 2.0 * math.pi)

        state["center_xy"][slot_env_ids, slot_idx] = center_xy
        state["axis_xy"][slot_env_ids, slot_idx] = axis_xy
        state["amplitude"][slot_env_ids, slot_idx] = amplitude
        state["cycle_rate"][slot_env_ids, slot_idx] = cycle_rate
        state["phase"][slot_env_ids, slot_idx] = phase
        state["yaw_w"][slot_env_ids, slot_idx] = yaw

        _write_dynamic_obstacle_slot(env, slot_env_ids, slot_idx, state)


def animate_dynamic_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_names: tuple[str, ...],
    motion_dt: float,
) -> None:
    """Gen2 interval 事件：推进当前激活的脚本化动态障碍。"""
    if not USE_AUTOPILOT_GEN2_DYNAMIC:
        return

    env_ids = _resolve_event_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    state = _ensure_dynamic_obstacle_state(env, asset_names)
    active_slots = state["active_slot"][env_ids]

    for slot_idx in range(len(asset_names)):
        slot_mask = active_slots == slot_idx
        if not torch.any(slot_mask):
            continue

        slot_env_ids = env_ids[slot_mask]
        state["phase"][slot_env_ids, slot_idx] = torch.remainder(
            state["phase"][slot_env_ids, slot_idx]
            + (2.0 * math.pi * state["cycle_rate"][slot_env_ids, slot_idx] * motion_dt),
            2.0 * math.pi,
        )
        _write_dynamic_obstacle_slot(env, slot_env_ids, slot_idx, state)


def configure_recovery_escape_scenarios(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """为一部分环境注入“前堵后退”的定向脱困课程。"""
    if not RECOVERY_SCENARIO_CONFIG["enabled"]:
        return

    env_ids = _resolve_event_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    scenario_state = _ensure_recovery_scenario_state(env)
    scenario_state["active"][env_ids] = False
    scenario_state["goal_distance"][env_ids] = 0.0
    scenario_state["goal_theta"][env_ids] = 0.0

    scenario_mask = torch.rand(len(env_ids), device=env.device) < RECOVERY_SCENARIO_CONFIG["probability"]
    if not torch.any(scenario_mask):
        return

    scenario_env_ids = env_ids[scenario_mask]
    robot = env.scene["robot"]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w[scenario_env_ids, :2], nan=0.0, posinf=0.0, neginf=0.0)
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w[scenario_env_ids])
    robot_yaw = torch.nan_to_num(robot_yaw, nan=0.0, posinf=0.0, neginf=0.0)

    scenario_state["active"][scenario_env_ids] = True
    goal_distance = torch.empty(len(scenario_env_ids), device=env.device).uniform_(
        RECOVERY_SCENARIO_CONFIG["goal_distance_min"],
        RECOVERY_SCENARIO_CONFIG["goal_distance_max"],
    )
    goal_theta = torch.empty(len(scenario_env_ids), device=env.device).uniform_(
        RECOVERY_SCENARIO_CONFIG["goal_theta_min"],
        RECOVERY_SCENARIO_CONFIG["goal_theta_max"],
    )
    goal_sign = torch.where(
        torch.rand(len(scenario_env_ids), device=env.device) > 0.5,
        torch.ones(len(scenario_env_ids), device=env.device),
        -torch.ones(len(scenario_env_ids), device=env.device),
    )
    scenario_state["goal_distance"][scenario_env_ids] = goal_distance
    scenario_state["goal_theta"][scenario_env_ids] = goal_theta * goal_sign

    local_left = torch.tensor(
        [RECOVERY_SCENARIO_CONFIG["front_blocker_x"], RECOVERY_SCENARIO_CONFIG["front_blocker_y"]],
        device=env.device,
    ).unsqueeze(0).repeat(len(scenario_env_ids), 1)
    local_right = torch.tensor(
        [RECOVERY_SCENARIO_CONFIG["front_blocker_x"], -RECOVERY_SCENARIO_CONFIG["front_blocker_y"]],
        device=env.device,
    ).unsqueeze(0).repeat(len(scenario_env_ids), 1)
    local_cap = torch.tensor(
        [RECOVERY_SCENARIO_CONFIG["front_cap_x"], 0.0],
        device=env.device,
    ).unsqueeze(0).repeat(len(scenario_env_ids), 1)
    left_xy = robot_pos + _rotate_local_xy(local_left, robot_yaw)
    right_xy = robot_pos + _rotate_local_xy(local_right, robot_yaw)
    cap_xy = robot_pos + _rotate_local_xy(local_cap, robot_yaw)

    # 将运动学动态障碍停到远处，避免与脱困课程互相干扰。
    dynamic_state = _ensure_dynamic_obstacle_state(env, DYNAMIC_OBSTACLE_ASSET_NAMES)
    dynamic_state["active_slot"][scenario_env_ids] = -1
    parked_positions = {
        "obs_inner_1": (2.2, 2.2),
        "obs_inner_3": (-2.2, 2.2),
        "obs_inner_5": (-2.2, -2.2),
    }
    for asset_name, offset in parked_positions.items():
        park_xy = env.scene.env_origins[scenario_env_ids, :2] + torch.tensor(offset, device=env.device)
        _write_kinematic_obstacle_pose(env, scenario_env_ids, asset_name, park_xy, torch.zeros_like(robot_yaw))

    _write_kinematic_obstacle_pose(env, scenario_env_ids, "obs_inner_2", left_xy, robot_yaw)
    _write_kinematic_obstacle_pose(env, scenario_env_ids, "obs_inner_8", right_xy, robot_yaw)
    _write_kinematic_obstacle_pose(env, scenario_env_ids, "obs_outer_1", cap_xy, robot_yaw)

# =============================================================================
# 辅助函数：检测是否 headless 模式
# =============================================================================
def is_headless_mode():
    """检测命令行参数中是否有 --headless"""
    return "--headless" in sys.argv

# =============================================================================
# 1. 自定义动作类 (Action Wrapper) - 保持不变
# =============================================================================

class UniDiffDriveAction(mdp.actions.JointVelocityAction):
    """
    差速驱动机器人的动作转换器

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/isaaclab/mdp/actions.html
    参考示例: isaaclab/exts/omni_isaac_lab_tasks/omni/isaac/lab/tasks/locomotion/velocity/action.py

    功能:
        - 将[线速度, 角速度]转换为[左轮速度, 右轮速度]
        - 应用速度限制（对齐ROS配置）
        - 应用加速度平滑（对齐ROS配置）
        - 裁剪到执行器限制

    参数来源:
        - wheel_radius: 0.0632m（ROS配置: wheel_diameter/2）
        - track_width: 0.342m（ROS配置: wheel_track）
        - max_lin_vel: 0.3 m/s（ROS配置: max_vel_x）
        - max_ang_vel: 1.0 rad/s（ROS配置: max_vel_theta）
        - max_accel_lin: 1.0 m/s²（ROS配置: acc_lim_x）
        - max_accel_ang: 0.6 rad/s²（ROS配置: acc_lim_theta）

    运动学模型:
        v_left = (v - w * track_width / 2) / wheel_radius
        v_right = (v + w * track_width / 2) / wheel_radius

    历史修改:
        - 2024-01-23: 添加速度和加速度限制（commit 9dad5de）
        - 2024-01-23: 修正轮距参数（commit 81d6ceb）
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # ✅ 从ROS配置读取参数（避免硬编码）
        ros_params = DashGoROSParams.from_yaml()
        self.wheel_radius = ros_params.wheel_radius  # wheel_diameter / 2.0
        self.track_width = ros_params.wheel_track

        self.prev_actions = None
        self.max_accel_lin = MOTION_CONFIG["max_accel_lin"]
        self.max_accel_ang = MOTION_CONFIG["max_accel_ang"]
        self.control_dt = float(env.cfg.sim.dt * env.cfg.decimation)

    def process_actions(self, actions: torch.Tensor, *args, **kwargs):
        # 对齐ROS速度限制
        max_lin_vel = MOTION_CONFIG["max_lin_vel"]
        max_reverse_speed = MOTION_CONFIG["max_reverse_speed"]
        max_ang_vel = MOTION_CONFIG["max_ang_vel"]

        # 速度裁剪：前进和倒车使用非对称上限
        target_v = torch.where(
            actions[:, 0] >= 0.0,
            actions[:, 0] * max_lin_vel,
            actions[:, 0] * max_reverse_speed,
        )
        target_v = torch.clamp(target_v, -max_reverse_speed, max_lin_vel)
        target_w = torch.clamp(actions[:, 1] * max_ang_vel, -max_ang_vel, max_ang_vel)

        # 加速度平滑
        if self.prev_actions is not None:
            delta_v = target_v - self.prev_actions[:, 0]
            delta_w = target_w - self.prev_actions[:, 1]
            max_delta_v = self.max_accel_lin * self.control_dt
            max_delta_w = self.max_accel_ang * self.control_dt
            delta_v = torch.clamp(delta_v, -max_delta_v, max_delta_v)
            delta_w = torch.clamp(delta_w, -max_delta_w, max_delta_w)
            target_v = self.prev_actions[:, 0] + delta_v
            target_w = self.prev_actions[:, 1] + delta_w

        self.prev_actions = torch.stack([target_v, target_w], dim=-1).clone()

        # 差速驱动转换
        v_left = (target_v - target_w * self.track_width / 2.0) / self.wheel_radius
        v_right = (target_v + target_w * self.track_width / 2.0) / self.wheel_radius

        # 裁剪到执行器限制
        max_wheel_vel = MOTION_CONFIG["max_wheel_vel"]
        v_left = torch.clamp(v_left, -max_wheel_vel, max_wheel_vel)
        v_right = torch.clamp(v_right, -max_wheel_vel, max_wheel_vel)

        joint_actions = torch.stack([v_left, v_right], dim=-1)
        return super().process_actions(joint_actions, *args, **kwargs)

# =============================================================================
# 2. 观测处理 (Observation) - 包含 NaN 清洗
# =============================================================================

def compute_velocity_scaled_lookahead(speed: torch.Tensor) -> torch.Tensor:
    """训练期与部署期统一的速度缩放前瞻距离。"""
    abs_speed = torch.abs(speed)
    forward_mask = speed >= 0.0
    forward_lookahead = torch.clamp(
        torch.maximum(
            torch.full_like(abs_speed, OBSERVATION_CONFIG["lookahead_min_forward"]),
            abs_speed * OBSERVATION_CONFIG["lookahead_gain_forward"],
        ),
        min=OBSERVATION_CONFIG["lookahead_min_forward"],
        max=OBSERVATION_CONFIG["lookahead_max_forward"],
    )
    reverse_lookahead = torch.clamp(
        torch.maximum(
            torch.full_like(abs_speed, OBSERVATION_CONFIG["lookahead_min_reverse"]),
            abs_speed * OBSERVATION_CONFIG["lookahead_gain_reverse"],
        ),
        min=OBSERVATION_CONFIG["lookahead_min_reverse"],
        max=OBSERVATION_CONFIG["lookahead_max_reverse"],
    )
    return torch.where(forward_mask, forward_lookahead, reverse_lookahead)


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
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, target_kind: str = "goal"
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
    return torch.stack(
        [
            clipped_dist,
            torch.sin(angle_error),
            torch.cos(angle_error),
        ],
        dim=-1,
    )


def obs_waypoint_vector(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    delta_pos_w, _, angle_error = _get_target_delta_and_heading(env, command_name, asset_cfg, target_kind="waypoint")
    dist = torch.norm(delta_pos_w, dim=-1)
    obs = _encode_goal_vector(dist, angle_error, OBSERVATION_CONFIG["waypoint_distance"])
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def obs_goal_vector(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    delta_pos_w, _, angle_error = _get_target_delta_and_heading(env, command_name, asset_cfg)
    dist = torch.norm(delta_pos_w, dim=-1)
    obs = _encode_goal_vector(dist, angle_error, OBSERVATION_CONFIG["max_distance"])
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def obs_forward_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    vel = torch.nan_to_num(robot.data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    return vel.unsqueeze(-1)


def obs_yaw_rate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    yaw_rate = torch.nan_to_num(robot.data.root_ang_vel_b[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
    return yaw_rate.unsqueeze(-1)

# =============================================================================
# [架构师新增] 核心物理计算工具 (2026-01-25)
# 作用：封装 RayCaster 距离计算逻辑，供观测和奖励共同调用
# =============================================================================

def _compute_raycaster_distance(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    [v7.1 Core Logic] 从深度相机获取模拟 LiDAR 数据

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    修改原因：
        - 2026-03-27 在本机 Isaac Lab 2.0.2 验证过原生 RayCaster
        - 当前实现仍只支持单个静态 mesh，无法覆盖本环境里的多障碍和运动学障碍
        - 因此训练主链继续使用深度相机回退，避免观测漏检

    逻辑：
        1. 从深度相机获取深度图 [N, ...]
        2. 将 batch 之后的所有维度统一展平为 [N, num_rays]
        3. 处理无效值并限制范围

    返回：原始距离数据 (单位: 米)，形状 [num_envs, 180]
    """
    # 1. 获取传感器
    sensor = env.scene[sensor_cfg.name]

    # 2. 从深度相机获取数据 [N, ...]
    depth_image = sensor.data.output["distance_to_image_plane"]

    # 3. 统一展平为 [N, num_rays] 的 LiDAR 格式。
    #    Isaac Sim / CameraCfg 在不同路径下可能给出 [N, 1, W] 或 [N, 1, 1, W]，
    #    这里不再假设只有单个高度维度。
    ranges = depth_image.reshape(depth_image.shape[0], -1)

    # 4. 处理无效值
    # 将无穷大(没打到物体)替换为最大距离
    # 将负值或NaN设为0
    max_range = SIM_LIDAR_MAX_RANGE
    ranges = torch.nan_to_num(ranges, posinf=max_range, neginf=0.0)
    ranges = torch.clamp(ranges, min=0.0, max=max_range)

    return ranges

# =============================================================================
# [架构师修复] 兼容性补丁：复活旧函数名
# 作用：防止 reward_navigation_sota 等旧代码报错
# =============================================================================

def _get_corrected_depth(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """兼容旧接口，直接转发给新的计算核心"""
    return _compute_raycaster_distance(env, sensor_cfg)

# =============================================================================
# 观测处理函数
# =============================================================================

def process_lidar_ranges(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    [v7.0 适配] 处理深度相机模拟的LiDAR数据

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04

    数据流：
        1. 深度相机 [N, 1, 180] (height=1, width=180)
        2. 展平为 [N, 180]
        3. 归一化到 [0, 1]
        4. 降采样到90个扇区 (每2°一个扇区)

    Returns:
        torch.Tensor: 形状为 [num_envs, 90] 的归一化距离数组
    """
    # 1. 调用核心工具获取米制距离 [N, 180]
    distances = _compute_raycaster_distance(env, sensor_cfg)

    # 2. 归一化到 [0, 1]
    max_range = 10.0
    distances_normalized = distances / max_range

    # 3. 降采样到90个扇区 (每2°一个，从180°降到90°)
    num_sectors = 90
    batch_size, num_rays = distances_normalized.shape

    if num_rays % num_sectors == 0:
        # 每个扇区取最小值（最安全的障碍物距离）
        depth_sectors = distances_normalized.view(batch_size, num_sectors, -1).min(dim=2)[0]
    else:
        # 如果不能整除，保持原样
        depth_sectors = distances_normalized

    return depth_sectors


def _sanitize_scan_tensor(scan: torch.Tensor, max_range: float = SIM_LIDAR_MAX_RANGE) -> torch.Tensor:
    """统一清洗深度扫描数据，确保训练和部署使用同一量纲边界。"""
    scan = torch.nan_to_num(scan, posinf=max_range, neginf=0.0)
    return torch.clamp(scan, min=0.0, max=max_range)


def _min_pool_resample_torch(scan: torch.Tensor, target_dim: int) -> torch.Tensor:
    """按等角度分桶做最小池化，避免 180→72 时直接截断尾部。"""
    batch_size, input_len = scan.shape
    edges = torch.round(torch.linspace(0, input_len, target_dim + 1, device=scan.device)).to(torch.long)
    edges[0] = 0
    edges[-1] = input_len
    pooled = []
    for index in range(target_dim):
        start = int(edges[index].item())
        end = int(edges[index + 1].item())
        if end <= start:
            start = min(start, input_len - 1)
            end = min(start + 1, input_len)
        pooled.append(torch.min(scan[:, start:end], dim=1).values)
    return torch.stack(pooled, dim=1).reshape(batch_size, target_dim)


def _get_forward_sector_scan(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    获取与实机一致的前向 180° 原始扫描。

    排序语义：
    - 先构造 [-90°, +90°] 的前向扇区
    - 再在 `process_forward_lidar()` 内重排为 front-centered 顺序
    """
    d_front_right = _compute_raycaster_distance(env, SceneEntityCfg(name="camera_front_right"))
    d_front_left = _compute_raycaster_distance(env, SceneEntityCfg(name="camera_front_left"))

    # Camera 图像默认按像素从左到右排列；这里显式翻转，使拼接后的角度顺序稳定为 [-90°, +90°]。
    scan_right = torch.flip(d_front_right, dims=[1])
    scan_left = torch.flip(d_front_left, dims=[1])
    return _sanitize_scan_tensor(torch.cat([scan_right, scan_left], dim=1), max_range=SIM_LIDAR_MAX_RANGE)


def process_forward_lidar(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    [Sim2Real V3.0] 前向 180° 双相机拼接 + front-centered 72 维 LiDAR。

    设计目标：
        1. 与实机 lakibeam 单雷达 180° 扇区对齐
        2. 保持策略输入维度 72 不变
        3. 与 ROS2 部署端 `process_lidar_ranges()` 的 front-centered 语义一致
    """
    forward_scan = _get_forward_sector_scan(env)
    front_centered_scan = torch.roll(forward_scan, shifts=-(forward_scan.shape[1] // 2), dims=1)
    downsampled = _min_pool_resample_torch(front_centered_scan, SIM_LIDAR_POLICY_DIM)
    return downsampled / SIM_LIDAR_MAX_RANGE


def process_stitched_lidar(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    向后兼容旧评估/诊断入口。

    历史上 `process_stitched_lidar()` 表示 360° 四相机拼接；
    当前实机合同已切换为前向 180°，这里保留旧函数名，仅作为
    `process_forward_lidar()` 的兼容别名，避免评估 worker 因导入失败白跑。
    """
    return process_forward_lidar(env)

# ============================================================================
# [v8.0] 业界标准避障策略 - 速度-距离动态约束
# ============================================================================

def _get_min_obstacle_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    forward_scan = _get_forward_sector_scan(env)
    return torch.min(forward_scan, dim=1)[0]


def reward_navrl_survival_bias(env: ManagerBasedRLEnv) -> torch.Tensor:
    """NavRL 风格常数偏置项，对应 reward 中的 +1。"""
    return torch.ones(env.num_envs, device=env.device)


def reward_navrl_goal_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """DashGo 版 NavRL 目标速度项：奖励沿目标方向的线速度投影。"""
    return reward_distance_tracking_potential(env, command_name, asset_cfg, target_kind="goal")


def reward_navrl_static_safety(env: ManagerBasedRLEnv) -> torch.Tensor:
    """NavRL 风格静态障碍安全项：对前向 LiDAR clearance 取对数均值。"""
    forward_scan = _get_forward_sector_scan(env)
    clearance = torch.clamp(forward_scan, min=1.0e-6, max=SIM_LIDAR_MAX_RANGE)
    reward = torch.log(clearance).mean(dim=1)
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def _get_navrl_dynamic_obstacle_clearance(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """返回当前激活动态障碍的 clearance 以及激活掩码。"""
    clearance = torch.full((env.num_envs,), SIM_LIDAR_MAX_RANGE, device=env.device)
    active_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    if not USE_AUTOPILOT_GEN2_DYNAMIC:
        return clearance, active_mask

    state = getattr(env, "_dynamic_obstacle_state", None)
    if not isinstance(state, dict):
        return clearance, active_mask

    asset_names = tuple(state.get("asset_names", ()))
    active_slot = state.get("active_slot")
    if active_slot is None:
        return clearance, active_mask

    robot_xy = torch.nan_to_num(env.scene["robot"].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)

    for slot_idx, asset_name in enumerate(asset_names):
        slot_mask = active_slot == slot_idx
        if not torch.any(slot_mask):
            continue

        obstacle_xy = torch.nan_to_num(env.scene[asset_name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
        center_distance = torch.norm(obstacle_xy - robot_xy, dim=-1)
        slot_clearance = center_distance - REWARD_CONFIG["navrl_dynamic_obstacle_half_extent"]
        slot_clearance = torch.clamp(slot_clearance, min=1.0e-6, max=SIM_LIDAR_MAX_RANGE)
        clearance = torch.where(slot_mask, slot_clearance, clearance)
        active_mask = active_mask | slot_mask

    return clearance, active_mask


def reward_navrl_dynamic_safety(env: ManagerBasedRLEnv) -> torch.Tensor:
    """NavRL 风格动态障碍安全项：对 clearance 取对数；无激活动态障碍时返回 0。"""
    clearance, active_mask = _get_navrl_dynamic_obstacle_clearance(env)
    reward = torch.log(clearance)
    reward = torch.where(active_mask, reward, torch.zeros_like(reward))
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def penalty_navrl_twist_smoothness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """NavRL 平滑项的 DashGo 适配：惩罚前向速度与偏航角速度的跃迁。"""
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


def penalty_unsafe_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_dist_threshold: float = 0.6) -> torch.Tensor:
    """
    [v8.1 修复版] 速度-距离 动态约束

    修复：先展平所有相机数据，确保 min_dist 是 [N] 形状，而不是 [N, W]

    核心逻辑："离得近没关系，但离得近还**跑得快**，就是找死。"

    数学公式：
        safe_vel_limit = clamp(min_dist, max=0.5)
        overspeed = clamp(vel - safe_vel_limit, min=0.0)
        penalty = overspeed

    Args:
        env: 环境对象
        asset_cfg: 机器人配置
        min_dist_threshold: 最小安全距离（默认0.25m）

    Returns:
        torch.Tensor: 超速惩罚 [N]

    架构师: Isaac Sim Architect (2026-01-27)
    参考方案: ETH Zurich RSL-RL, OpenAI Navigation, ROS2 Nav2
    """
    min_dist = _get_min_obstacle_distance(env)
    vel = torch.abs(env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0])

    clearance = torch.clamp(min_dist - 0.20, min=0.0)
    ratio = torch.clamp(clearance / max(min_dist_threshold - 0.20, 1.0e-3), min=0.0, max=1.0)
    safe_vel_limit = ratio * MOTION_CONFIG["max_lin_vel"]
    overspeed = torch.clamp(vel - safe_vel_limit, min=0.0)

    return torch.nan_to_num(overspeed, nan=0.0, posinf=0.0, neginf=0.0)


def penalty_undesired_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.1) -> torch.Tensor:
    """
    [v8.0] 轻微接触惩罚 - 第二层防御

    核心逻辑：只要碰到任何东西（力 > 0.1N），就每帧扣分

    设计理念：
        - 第一层（Termination）：猛烈碰撞（>50N）直接重置
        - 第二层（Reward）：轻微接触（0.1N）给予疼痛感，但不重置
        - 目的：让机器人学会"别碰我"，但不会因为轻轻蹭一下就死

    Args:
        env: 环境对象
        sensor_cfg: 接触力传感器配置
        threshold: 接触力阈值（默认0.1N，极低的阈值）

    Returns:
        torch.Tensor: 接触惩罚 [N]

    架构师: Isaac Sim Architect (2026-01-27)
    """
    # [Fix 2026-01-27] 使用正确的属性名 net_forces_w
    # Isaac Lab ContactSensor 的属性名是 net_forces_w，而非 net_contact_forces
    # data.net_forces_w 的形状是 [num_envs, num_bodies, 3]
    contact_data = env.scene[sensor_cfg.name].data.net_forces_w  # [N, num_bodies, 3]

    # [Fix 2026-01-27] 计算合力大小并降维
    # 先计算力的模长 -> [N, num_bodies]
    # 然后取最大值（假设底盘有多个碰撞体，取受力最大的那个）-> [N]
    force_mag = torch.norm(contact_data, dim=-1).max(dim=1)[0]  # [N]

    # 任何超过阈值的接触都给予惩罚
    has_contact = force_mag > threshold

    # 返回正惩罚量，权重统一由 RewardCfg 提供负号
    return torch.where(has_contact, 1.0, 0.0)


def penalty_obstacle_proximity(env: ManagerBasedRLEnv, threshold: float = 0.6) -> torch.Tensor:
    min_dist = _get_min_obstacle_distance(env)
    penalty = torch.clamp(threshold - min_dist, min=0.0) / max(threshold, 1.0e-3)
    return torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)


def penalty_progress_stall(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_kind: str = "waypoint",
    activation_distance: float = 0.6,
    min_progress: float = 0.005,
    max_forward_speed: float = 0.05,
    warmup_steps: int = 15,
    trigger_steps: int = 8,
) -> torch.Tensor:
    """惩罚持续低进展的蹭行/发呆行为。"""
    if target_kind == "waypoint":
        target_pos = _get_command_waypoint_pos_w(env, command_name, asset_cfg)[:, :2]
    else:
        target_pos = _get_command_target_pos_w(env, command_name)[:, :2]

    robot = env.scene[asset_cfg.name]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(target_pos - robot_pos, dim=-1)
    forward_speed = torch.abs(
        torch.nan_to_num(robot.data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    )

    if not hasattr(env, "_prev_target_distance"):
        env._prev_target_distance = dist.clone()
    if not hasattr(env, "_stall_counts"):
        env._stall_counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    env._stall_counts[env.episode_length_buf < 2] = 0
    prev_dist = env._prev_target_distance
    progress = prev_dist - dist
    env._prev_target_distance = dist.detach().clone()

    stalled = (
        (env.episode_length_buf > warmup_steps)
        & (dist > activation_distance)
        & (progress < min_progress)
        & (forward_speed < max_forward_speed)
    )
    env._stall_counts = torch.where(
        stalled,
        env._stall_counts + 1,
        torch.zeros_like(env._stall_counts),
    )
    penalty = torch.clamp(
        (env._stall_counts.float() - float(trigger_steps)) / float(max(trigger_steps, 1)),
        min=0.0,
        max=1.0,
    )
    return penalty


def penalty_orbiting(
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
    """惩罚远离目标时高角速度绕圈且缺乏净进展的行为。"""
    if target_kind == "waypoint":
        target_pos = _get_command_waypoint_pos_w(env, command_name, asset_cfg)[:, :2]
    else:
        target_pos = _get_command_target_pos_w(env, command_name)[:, :2]

    robot = env.scene[asset_cfg.name]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(target_pos - robot_pos, dim=-1)
    forward_speed = torch.abs(
        torch.nan_to_num(robot.data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    )
    angular_speed = torch.abs(
        torch.nan_to_num(robot.data.root_ang_vel_b[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
    )

    if not hasattr(env, "_orbit_prev_target_distance"):
        env._orbit_prev_target_distance = dist.clone()
    if not hasattr(env, "_orbit_counts"):
        env._orbit_counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    env._orbit_counts[env.episode_length_buf < 2] = 0
    prev_dist = env._orbit_prev_target_distance
    progress = prev_dist - dist
    env._orbit_prev_target_distance = dist.detach().clone()

    orbiting = (
        (env.episode_length_buf > warmup_steps)
        & (dist > activation_distance)
        & (progress < min_progress)
        & (angular_speed > min_angular_speed)
        & (forward_speed < max_forward_speed)
    )
    env._orbit_counts = torch.where(
        orbiting,
        env._orbit_counts + 1,
        torch.zeros_like(env._orbit_counts),
    )

    penalty = torch.clamp(
        (env._orbit_counts.float() - float(trigger_steps)) / float(max(trigger_steps, 1)),
        min=0.0,
        max=1.0,
    )
    speed_scale = torch.clamp(
        (angular_speed - min_angular_speed) / max(MOTION_CONFIG["max_ang_vel"] - min_angular_speed, 1.0e-3),
        min=0.0,
        max=1.0,
    )
    return penalty * speed_scale


# =============================================================================
# [v5.1 ACL] 自适应课程学习核心函数
# =============================================================================

def curriculum_adaptive_distance(env, env_ids, command_name,
                                initial_dist, max_dist, step_size,
                                upgrade_threshold, downgrade_threshold,
                                window_size):
    """
    [v5.2 Fix] 自适应课程函数 (修复 Crash + 实装难度应用)

    Fix:
    1. 返回值改为标量 Tensor (解决 RuntimeError)
    2. 增加了对 Command Manager 的实际修改，确保难度生效
    """
    # 1. 初始化统计数据
    if not hasattr(env, "curriculum_stats"):
        env.curriculum_stats = {
            "current_dist": initial_dist,
            "window_size": window_size,
            "success_history": [],
        }
        append_curriculum_trace({
            "event": "init",
            "current_dist": float(initial_dist),
            "window_size": int(window_size),
        })
        # 初始返回标量
        return torch.tensor(env.curriculum_stats["current_dist"], device=env.device)

    # 2. 直接从终止管理器读取成功项，而不是依赖 extras 里并不存在的 "success" 键。
    successes = None
    try:
        if hasattr(env, "termination_manager") and "reach_goal" in env.termination_manager.active_terms:
            successes = env.termination_manager.get_term("reach_goal")[env_ids].float()
    except Exception as e:
        if not hasattr(env, "curriculum_success_error_logged"):
            print(f"[Warning] Failed to query reach_goal for curriculum: {e}")
            env.curriculum_success_error_logged = True

    if successes is not None and len(successes) > 0:
        valid_mask = None
        if hasattr(env, "episode_length_buf"):
            try:
                episode_lengths = env.episode_length_buf[env_ids]
                valid_mask = episode_lengths > 0
            except Exception:
                valid_mask = None

        if valid_mask is not None:
            successes = successes[valid_mask]
            if successes.numel() == 0:
                append_curriculum_trace({
                    "event": "skip_pre_episode_reset",
                    "env_count": int(len(valid_mask)),
                })
                current_dist = env.curriculum_stats["current_dist"]
                return torch.tensor(current_dist, device=env.device)

        stats = env.curriculum_stats
        history = list(stats.get("success_history", []))
        history.extend(successes.detach().cpu().tolist())
        history = history[-int(window_size):]
        stats["success_history"] = history

        if history:
            success_rate = float(sum(history) / len(history))
            previous_dist = float(stats["current_dist"])
            current_dist = previous_dist

            if success_rate > upgrade_threshold:
                current_dist = min(current_dist + step_size, max_dist)
            elif success_rate < downgrade_threshold:
                current_dist = max(current_dist - step_size, initial_dist)

            stats["current_dist"] = current_dist
            env.curriculum_stats = stats

            # -------------------------------------------------------
            # 实际修改命令生成器范围，让课程真正生效。
            # -------------------------------------------------------
            try:
                cmd_term = env.command_manager.get_term(command_name)

                if hasattr(cmd_term, "max_dist"):
                    cmd_term.max_dist = max(float(current_dist), float(getattr(cmd_term, "min_dist", 0.5)))

                if hasattr(cmd_term, "cfg") and hasattr(cmd_term.cfg, "ranges"):
                    if hasattr(cmd_term.cfg.ranges, "r"):
                        cmd_term.cfg.ranges.r = (initial_dist, current_dist)
                    elif hasattr(cmd_term.cfg.ranges, "pos_x"):
                        half_dist = current_dist
                        cmd_term.cfg.ranges.pos_x = (-half_dist, half_dist)
                        cmd_term.cfg.ranges.pos_y = (-half_dist, half_dist)

            except Exception as e:
                if not hasattr(env, "cmd_update_error_logged"):
                    print(f"[Warning] Failed to update command range: {e}")
                    env.cmd_update_error_logged = True

            append_curriculum_trace({
                "event": "compute",
                "env_count": len(successes),
                "success_sum": float(successes.sum().item()),
                "success_rate": success_rate,
                "history_len": len(history),
                "previous_dist": previous_dist,
                "current_dist": current_dist,
            })

    # 3. 返回当前难度 (标量!)
    # 修复了之前返回 vector 导致的 RuntimeError
    current_dist = env.curriculum_stats["current_dist"]
    return torch.tensor(current_dist, device=env.device)

# =============================================================================
# [v5.0 Legacy] 线性课程学习（保留用于对比）
# =============================================================================

def curriculum_expand_target_range(env, env_ids, command_name, start_step, end_step, min_limit, max_limit):
    """
    [v5.0 核心] 自动化课程学习
    根据当前训练总步数，线性扩展目标生成的距离范围 (3m -> 8m)

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/isaaclab/managers.html
    参考示例: Isaac Lab官方curriculum学习示例

    原理：
        - 通过动态修改命令生成器的配置范围实现难度爬坡
        - 使用物理步数（common_step_counter）而非iteration数作为时间基准
        - 线性插值保证平滑过渡

    Args:
        env: 管理型RL环境
        env_ids: 本次重置的环境ID（未使用，保持接口一致）
        command_name: 要修改的命令名称（"target_pose"）
        start_step: 课程开始步数（物理步）
        end_step: 课程结束步数（物理步）
        min_limit: 初始距离限制（3.0m）
        max_limit: 最终距离限制（8.0m）
    """
    current_step = env.common_step_counter

    # 计算进度 alpha (0.0 ~ 1.0)
    if current_step < start_step:
        alpha = 0.0
    elif current_step > end_step:
        alpha = 1.0
    else:
        alpha = (current_step - start_step) / (end_step - start_step)

    # 计算当前难度
    current_limit = min_limit + (max_limit - min_limit) * alpha

    # 动态修改命令生成器的参数
    cmd_term = env.command_manager.get_term(command_name)
    if hasattr(cmd_term.cfg, "ranges") and hasattr(cmd_term.cfg.ranges, "pos_x"):
        # 同时修改 X 和 Y 的范围，保持正方形区域
        cmd_term.cfg.ranges.pos_x = (-current_limit, current_limit)
        cmd_term.cfg.ranges.pos_y = (-current_limit, current_limit)

# =============================================================================
# [v5.0 Hotfix] 自定义tanh距离奖励函数
# =============================================================================

def reward_position_command_error_tanh(env, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    [v5.0 Hotfix] 手动实现tanh距离奖励（Isaac Lab 4.5无此API）

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    问题修复: AttributeError: module 'isaaclab.envs.mdp.rewards' has no attribute 'position_command_error_tanh'

    奖励范围: (0, 1]
    逻辑: 距离越近，奖励越高（接近1）；距离越远，奖励越低（接近0）

    数学原理:
        reward = 1.0 - tanh(dist / std)
        - 当 dist = 0, tanh = 0, reward = 1.0（到达目标）
        - 当 dist = std, tanh ≈ 0.76, reward ≈ 0.24（中等距离）
        - 当 dist >> std, tanh ≈ 1.0, reward ≈ 0.0（远距离）

    Args:
        env: 管理型RL环境
        std: 标准化参数，控制tanh饱和速度
        command_name: 命令名称（"target_pose"）
        asset_cfg: 机器人实体配置

    Returns:
        torch.Tensor: 形状为[num_envs]的奖励张量，范围(0, 1]
    """
    # 1. 获取目标位置 (x, y)
    target_pos = env.command_manager.get_command(command_name)[:, :2]

    # 2. 获取机器人位置 (x, y)
    robot_pos = env.scene[asset_cfg.name].data.root_pos_w[:, :2]

    # 3. 计算欧几里得距离
    dist = torch.norm(target_pos - robot_pos, dim=1)

    # 4. 计算tanh奖励
    return 1.0 - torch.tanh(dist / std)

# =============================================================================
# [v5.0 Ultimate] 辅助奖励函数
# =============================================================================

def reward_target_speed(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_kind: str = "waypoint",
):
    """
    [Geo-Distill V3.0] 速度奖励：三重保护机制

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    修复原因：
        1. 防止"倒车刷分"导致醉汉走路
        2. 防止"原地转圈"（angular_penalty）
        3. 倒车惩罚太弱（-2.0 → -10.0）

    奖励逻辑：
        - 前进（vel > 0）：指数奖励（鼓励接近0.3 m/s）
        - 倒车（vel < 0）：5倍惩罚（从2倍提高到5倍）
        - 旋转（ang_vel）：-0.5 * abs(ang_vel) 新增

    [2026-01-27] 调整目标速度：0.25 → 0.3 m/s
    [V3.0] 添加角速度惩罚，防止转圈
    """
    robot = env.scene[asset_cfg.name]
    _, _, angle_error = _get_target_delta_and_heading(env, command_name, asset_cfg, target_kind=target_kind)
    forward_vel = torch.nan_to_num(robot.data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    ang_vel = torch.abs(torch.nan_to_num(robot.data.root_ang_vel_b[:, 2], nan=0.0, posinf=0.0, neginf=0.0))
    min_dist = _get_min_obstacle_distance(env)

    progress_speed = forward_vel * torch.cos(angle_error)
    alignment = torch.clamp(torch.cos(angle_error), min=0.0, max=1.0)
    clearance_gate = torch.clamp(
        (min_dist - REWARD_CONFIG["safe_distance"])
        / max(REWARD_CONFIG["obstacle_penalty_threshold"] - REWARD_CONFIG["safe_distance"], 1.0e-3),
        min=0.0,
        max=1.0,
    )
    desired_progress = REWARD_CONFIG["target_speed"] * alignment
    speed_reward = torch.exp(-torch.abs(progress_speed - desired_progress) / 0.08)
    speed_reward = speed_reward * (clearance_gate > 0.35).float() * alignment
    creep_penalty = torch.where(
        (clearance_gate <= 0.35) & (forward_vel > 0.02),
        0.25 + 0.75 * torch.clamp(
            forward_vel / max(MOTION_CONFIG["max_lin_vel"], 1.0e-3), min=0.0, max=1.0
        ),
        torch.zeros_like(forward_vel),
    )
    angular_penalty = 0.1 * ang_vel
    return torch.nan_to_num(speed_reward - angular_penalty - creep_penalty, nan=0.0, posinf=0.0, neginf=0.0)


def reward_contextual_reverse_escape(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_kind: str = "waypoint",
    front_blocked_threshold: float = 0.55,
    rear_clear_threshold: float = 0.80,
    progress_threshold: float = 0.02,
) -> torch.Tensor:
    """前向 180° 合同下默认不鼓励盲目倒车；保留接口仅用于兼容旧奖励图谱。"""
    robot = env.scene[asset_cfg.name]
    _, _, angle_error = _get_target_delta_and_heading(env, command_name, asset_cfg, target_kind=target_kind)
    forward_vel = torch.nan_to_num(robot.data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    ang_vel = torch.abs(torch.nan_to_num(robot.data.root_ang_vel_b[:, 2], nan=0.0, posinf=0.0, neginf=0.0))

    front_min = _get_min_obstacle_distance(env)
    rear_min = torch.zeros_like(front_min)
    progress_speed = forward_vel * torch.cos(angle_error)

    front_blocked_gate = torch.clamp(
        (front_blocked_threshold - front_min) / max(front_blocked_threshold - 0.20, 1.0e-3),
        min=0.0,
        max=1.0,
    )
    rear_clear_gate = torch.clamp(
        (rear_min - rear_clear_threshold) / max(1.50 - rear_clear_threshold, 1.0e-3),
        min=0.0,
        max=1.0,
    )
    stalled_gate = (
        (progress_speed < progress_threshold)
        & (front_min < front_blocked_threshold)
        & (rear_min > rear_clear_threshold)
    ).float()
    reverse_gate = torch.clamp(
        (-forward_vel) / max(MOTION_CONFIG["max_reverse_speed"], 1.0e-3),
        min=0.0,
        max=1.0,
    )
    angular_penalty = REWARD_CONFIG["reverse_escape_ang_penalty"] * torch.clamp(
        ang_vel / max(MOTION_CONFIG["max_ang_vel"], 1.0e-3),
        min=0.0,
        max=1.0,
    )
    reward = front_blocked_gate * rear_clear_gate * stalled_gate * reverse_gate - angular_penalty * stalled_gate
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

# =============================================================================
# 3. 奖励函数 (包含 Goal Fixing 和 NaN 清洗)
# =============================================================================

def reward_navigation_sota(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """
    SOTA风格导航奖励函数

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/isaaclab/mdp/rewards.html
    参考示例: isaaclab/exts/omni_isaac_lab_tasks/omni/isaac/lab/tasks/locomotion/velocity/rewards.py:120

    奖励项组成:
        1. 进度奖励: forward_vel * cos(angle_error) - 鼓励向目标前进
        2. 极速奖励: 速度>0.25且朝向正确时给予 - 鼓励快速前进
        3. 倒车惩罚: 惩罚倒车行为
        4. 避障惩罚: 基于LiDAR距离的指数惩罚

    设计依据:
        - 进度奖励: 势能差奖励的简化版本，避免过度优化
        - 极速奖励: 鼓励机器人使用接近max_vel_x的速度（0.25 vs 0.3）
        - 避障阈值: 0.55m（约2.7倍robot_radius），符合ROS安全距离

    Args:
        env: 管理器基于RL环境
        asset_cfg: 机器人实体配置
        sensor_cfg: LiDAR传感器配置
        command_name: 目标命令名称

    Returns:
        torch.Tensor: 形状为[num_envs]的奖励张量，范围已裁剪到[-10, 10]

    历史修改:
        - 2024-01-20: 降低平滑度惩罚权重（commit 123abc）
        - 2024-01-22: 添加极速奖励项（commit 456def）
    """
    robot = env.scene[asset_cfg.name]
    target_pos_w = env.command_manager.get_command(command_name)[:, :3]

    # 基础数据清洗
    forward_vel = torch.nan_to_num(robot.data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    # ✅ [V3.0] 恢复ang_vel惩罚（之前注释掉导致转圈问题）
    ang_vel = torch.nan_to_num(robot.data.root_ang_vel_b[:, 2], nan=0.0, posinf=0.0, neginf=0.0)

    forward_vel = torch.clamp(forward_vel, -10.0, 10.0)
    
    robot_pos = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    
    # [架构师修复] 严格 2D 计算
    delta_pos_w = target_pos_w[:, :2] - robot_pos[:, :2]
    
    target_angle = torch.atan2(delta_pos_w[:, 1], delta_pos_w[:, 0])
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    robot_yaw = torch.nan_to_num(robot_yaw, nan=0.0, posinf=0.0, neginf=0.0)
    angle_error = wrap_to_pi(target_angle - robot_yaw)
    
    # [1] 进度奖励
    reward_progress = REWARD_CONFIG["progress_weight"] * forward_vel * torch.cos(angle_error)

    # [2] 极速奖励
    is_facing_target = torch.cos(angle_error) > REWARD_CONFIG["facing_threshold"]
    reward_high_speed = (
        (forward_vel > REWARD_CONFIG["high_speed_threshold"]).float() *
        is_facing_target.float() *
        REWARD_CONFIG["high_speed_reward"]
    )

    # [3] 倒车惩罚
    reward_backward = -REWARD_CONFIG["backward_penalty"] * torch.abs(
        torch.min(forward_vel, torch.zeros_like(forward_vel))
    )

    # [4] 避障惩罚
    # [兼容] headless 模式下传感器不存在，跳过避障惩罚
    if sensor_cfg is not None:
        depth_radial = _get_corrected_depth(env, sensor_cfg)
        min_dist = torch.min(depth_radial, dim=-1)[0]
        safe_dist = REWARD_CONFIG["safe_distance"]
        reward_collision = torch.zeros_like(min_dist)
        mask_danger = min_dist < safe_dist
        reward_collision[mask_danger] = -REWARD_CONFIG["collision_penalty"] * torch.exp(
            REWARD_CONFIG["collision_decay"] * (safe_dist - min_dist[mask_danger])
        )
    else:
        # headless 模式：没有传感器数据，使用零避障惩罚
        reward_collision = torch.zeros(forward_vel.shape, device=env.device)

    # [5] 动作平滑 (移除，改为单独项并降低权重)
    # reward_rot = -0.05 * torch.abs(ang_vel)**2

    # ✅ [V3.0] 角速度惩罚（防止转圈）
    reward_angular = -REWARD_CONFIG["angular_penalty"] * torch.abs(ang_vel)

    # ✅ [V3.0] 停车诱导逻辑（势能井）
    # 只有同时满足 dist<0.25 AND vel<0.1 才给100分
    is_at_goal = torch.norm(delta_pos_w, p=2, dim=-1) < REWARD_CONFIG["stop_dist_thresh"]
    is_stopped = torch.abs(forward_vel) < REWARD_CONFIG["stop_vel_thresh"]
    reward_terminal = torch.where(
        is_at_goal & is_stopped,
        torch.tensor(REWARD_CONFIG["terminal_reward"], device=env.device),
        torch.tensor(0.0, device=env.device)
    )

    total_reward = (reward_progress + reward_high_speed + reward_backward +
                   reward_collision + reward_angular + reward_terminal)
    return torch.clamp(
        torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0),
        REWARD_CONFIG["reward_clip_min"],
        REWARD_CONFIG["reward_clip_max"]
    )

# [架构师重构] 基于势能差的引导奖励
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
    lin_vel_w = env.scene[asset_cfg.name].data.root_lin_vel_w[:, :2]
    lin_vel_w = torch.nan_to_num(lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)

    approach_velocity = torch.sum(lin_vel_w * dist_vec, dim=-1)
    return torch.clamp(approach_velocity, -10.0, 10.0)

# [架构师新增] 对准奖励：只要车头对得准，就给分。鼓励原地转向。
def reward_facing_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_kind: str = "goal",
) -> torch.Tensor:
    _, _, angle_error = _get_target_delta_and_heading(env, command_name, asset_cfg, target_kind=target_kind)
    return REWARD_CONFIG["facing_reward_scale"] * torch.exp(
        -torch.abs(angle_error) / REWARD_CONFIG["facing_angle_scale"]
    )

# [架构师新增] 生存惩罚：逼迫机器人动起来
def reward_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

# [架构师新增] 动作平滑度奖励
def reward_action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    diff = env.action_manager.action - env.action_manager.prev_action
    return -torch.sum(torch.square(diff), dim=1)

# [删除] 冲突的奖励函数定义（第二个版本，导致机器人倒车刷分）
# 原因：Python使用最后一个定义，而这个版本奖励任意方向的0.25m/s速度
# 后果：机器人学会倒车来刷分，导致"醉汉走路"
#
# 正确版本在line 409，只奖励前进速度

# 日志记录函数
def log_distance_to_goal(
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
    dist = torch.norm(target_pos - robot_pos, dim=-1)
    return dist

def log_linear_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    lin_vel_b = torch.nan_to_num(robot.data.root_lin_vel_b[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.norm(lin_vel_b, dim=-1)

# 稀疏到达奖励
def reward_near_goal(env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    target_pos_w = _get_command_target_pos_w(env, command_name)[:, :2]
    robot_pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(target_pos_w - robot_pos, dim=-1)
    speed = torch.abs(torch.nan_to_num(env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0))
    stopped = speed < REWARD_CONFIG["goal_stop_velocity"]
    return ((dist < threshold) & stopped).float()

def penalty_collision_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    net_forces = torch.norm(sensor.data.net_forces_w, dim=-1)
    if net_forces.dim() > 1: net_forces = torch.max(net_forces, dim=-1)[0]
    net_forces = torch.nan_to_num(net_forces, nan=0.0, posinf=0.0, neginf=0.0)
    is_startup = env.episode_length_buf < 50
    penalty = (net_forces > threshold).float()
    penalty[is_startup] = 0.0
    return penalty

def penalty_out_of_bounds(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(robot_pos - env.scene.env_origins, dim=-1)
    return (dist > threshold).float()

# =============================================================================
# 4. 终止条件
# =============================================================================

def check_out_of_bounds(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(robot_pos - env.scene.env_origins, dim=-1)
    return (dist > threshold)

def check_collision_simple(
    env: ManagerBasedRLEnv,
    sensor_cfg_base: SceneEntityCfg,
    threshold: float,
    sustained_frames: int = 3,
    grace_steps: int = 20,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_cfg_base.name]
    forces = torch.norm(sensor.data.net_forces_w, dim=-1)
    if forces.dim() > 1:
        forces = torch.max(forces, dim=-1)[0]
    forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
    if not hasattr(env, "_hard_collision_counts"):
        env._hard_collision_counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    env._hard_collision_counts[env.episode_length_buf < 2] = 0
    hard_contact = forces > threshold
    env._hard_collision_counts = torch.where(
        hard_contact,
        env._hard_collision_counts + 1,
        torch.zeros_like(env._hard_collision_counts),
    )
    is_safe = env.episode_length_buf < grace_steps
    return (env._hard_collision_counts >= sustained_frames) & (~is_safe)


def check_collision_navrl_style(env: ManagerBasedRLEnv, static_threshold: float = 0.3, dynamic_threshold: float = 0.3) -> torch.Tensor:
    """NavRL 风格碰撞终止：安全半径内视为碰撞，不再依赖旧 DashGo 的接触力逻辑。"""
    static_collision = _get_min_obstacle_distance(env) <= static_threshold
    dynamic_clearance, dynamic_active = _get_navrl_dynamic_obstacle_clearance(env)
    dynamic_collision = dynamic_active & (dynamic_clearance <= dynamic_threshold)
    return static_collision | dynamic_collision

def check_reach_goal(env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    target_pos_w = _get_command_target_pos_w(env, command_name)[:, :2]
    robot_pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.norm(target_pos_w - robot_pos, dim=-1)
    return dist < threshold

def check_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (env.episode_length_buf >= env.max_episode_length)

def check_velocity_explosion(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    lin_vel = torch.norm(robot.data.root_lin_vel_w, dim=-1)
    ang_vel = torch.norm(robot.data.root_ang_vel_w, dim=-1)
    is_bad = torch.isnan(lin_vel) | torch.isnan(ang_vel) | torch.isinf(lin_vel) | torch.isinf(ang_vel)
    return (lin_vel > threshold) | (ang_vel > threshold) | is_bad

def check_base_height_bad(env: ManagerBasedRLEnv, min_height: float, max_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    z_pos = robot.data.root_pos_w[:, 2]
    is_bad = torch.isnan(z_pos) | torch.isinf(z_pos)
    return (z_pos < min_height) | (z_pos > max_height) | is_bad

def reset_root_state_safe_donut(env: ManagerBasedRLEnv, env_ids: torch.Tensor, min_radius: float, max_radius: float, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    min_r2 = min_radius ** 2
    max_r2 = max_radius ** 2
    r_sq = torch.rand(len(env_ids), device=env.device) * (max_r2 - min_r2) + min_r2
    r = torch.sqrt(r_sq)
    theta = torch.rand(len(env_ids), device=env.device) * 2 * math.pi - math.pi
    
    pos_x_local = r * torch.cos(theta)
    pos_y_local = r * torch.sin(theta)
    
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 0] = pos_x_local + env.scene.env_origins[env_ids, 0]
    root_state[:, 1] = pos_y_local + env.scene.env_origins[env_ids, 1]
    root_state[:, 2] = 0.20 
    
    random_yaw = torch.rand(len(env_ids), device=env.device) * 2 * math.pi - math.pi
    zeros = torch.zeros_like(random_yaw)
    quat = quat_from_euler_xyz(zeros, zeros, random_yaw)
    root_state[:, 3:7] = quat
    
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)

class RelativeRandomTargetCommand(mdp.UniformPoseCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # [架构师修正 2026-01-24] 课程学习：从近到远
        # 修改历史：1.0-2.0 → 0.5-1.5 → 0.1-0.5（送分题测试）→ 0.5-1.5（恢复正常）
        # 验证：送分题测试确认 reach_goal 系统正常（已达到100%）
        # 现状：坐标系不一致问题已修复，可以恢复正常训练
        self.min_dist = 0.5  # 保持近距离采样下限，避免目标贴到车体内部
        self.max_dist = INITIAL_TARGET_MAX_DIST  # Gen1 从 1.0m 起跑；Gen2 动态阶段直接从中距离续训
        self.max_path_points = OBSERVATION_CONFIG["max_path_points"]
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
        t = torch.linspace(0.0, 1.0, self.max_path_points, device=self.device).unsqueeze(0)
        goal_progress = (steps - 1).clamp(min=1).unsqueeze(-1).float()
        scaled_t = torch.clamp(t * goal_progress, max=goal_progress) / goal_progress
        interp_xy = start_xy.unsqueeze(1) + delta.unsqueeze(1) * scaled_t.unsqueeze(-1)
        path = torch.zeros(batch_size, self.max_path_points, 3, device=self.device)
        path[:, :, :2] = interp_xy
        headings = torch.atan2(delta[:, 1], delta[:, 0]).unsqueeze(-1).expand(-1, self.max_path_points)
        path[:, :, 2] = headings
        return path, steps

    def get_waypoint_pose_w(self, asset_name: str = "robot") -> torch.Tensor:
        robot = self._env.scene[asset_name]
        robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
        speed = torch.nan_to_num(robot.data.root_lin_vel_b[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
        lookahead = compute_velocity_scaled_lookahead(speed)

        path_xy = self.reference_path_w[:, :, :2]
        distances = torch.norm(path_xy - robot_pos.unsqueeze(1), dim=-1)
        mask = (
            torch.arange(self.max_path_points, device=self.device).unsqueeze(0)
            < self.reference_path_len.unsqueeze(1)
        )
        masked_distances = torch.where(mask, distances, torch.full_like(distances, 1.0e6))
        nearest_idx = torch.argmin(masked_distances, dim=1)
        self.reference_path_cursor = torch.maximum(self.reference_path_cursor, nearest_idx)

        cursor_dist = torch.gather(masked_distances, 1, self.reference_path_cursor.unsqueeze(1)).squeeze(1)
        target_mask = mask & (distances >= lookahead.unsqueeze(1))
        target_mask &= (
            torch.arange(self.max_path_points, device=self.device).unsqueeze(0)
            >= self.reference_path_cursor.unsqueeze(1)
        )
        fallback_idx = (self.reference_path_len - 1).clamp(min=0)
        has_target = torch.any(target_mask, dim=1)
        candidate_idx = torch.argmax(target_mask.to(torch.int64), dim=1)
        selected_idx = torch.where(has_target, candidate_idx, fallback_idx)
        self.reference_path_cursor = torch.maximum(self.reference_path_cursor, selected_idx)

        selected = self.reference_path_w[torch.arange(self.num_envs, device=self.device), self.reference_path_cursor]
        self.waypoint_pose_w[:, :3] = selected
        self.waypoint_pose_w[:, 3] = 1.0
        self.waypoint_pose_w[:, 4:] = 0.0
        return self.waypoint_pose_w
    
    def _resample_command(self, env_ids: torch.Tensor):
        robot = self._env.scene[self.cfg.asset_name]
        if robot is not None and robot.data.root_pos_w is not None:
            robot_pos = robot.data.root_pos_w[env_ids, :3]
            robot_pos = torch.nan_to_num(robot_pos, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            robot_pos = torch.zeros((len(env_ids), 3), device=self.device)
        r = torch.empty(len(env_ids), device=self.device).uniform_(self.min_dist, self.max_dist)
        theta = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        # 显式增加“目标在车后方”的采样比例，避免经验池几乎全是前进轨迹。
        reverse_mask = torch.rand(len(env_ids), device=self.device) < 0.35
        if torch.any(reverse_mask):
            reverse_count = int(reverse_mask.sum().item())
            reverse_angles = torch.empty(reverse_count, device=self.device).uniform_(0.65 * math.pi, math.pi)
            reverse_sign = torch.where(
                torch.rand(reverse_count, device=self.device) > 0.5,
                torch.ones(reverse_count, device=self.device),
                -torch.ones(reverse_count, device=self.device),
            )
            theta[reverse_mask] = reverse_angles * reverse_sign
        scenario_state = _ensure_recovery_scenario_state(self._env)
        recovery_mask = scenario_state["active"][env_ids]
        if torch.any(recovery_mask):
            r[recovery_mask] = scenario_state["goal_distance"][env_ids][recovery_mask]
            theta[recovery_mask] = scenario_state["goal_theta"][env_ids][recovery_mask]
        goal_xy = torch.stack(
            [
                robot_pos[:, 0] + r * torch.cos(theta),
                robot_pos[:, 1] + r * torch.sin(theta),
            ],
            dim=-1,
        )
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

# =============================================================================
# 配置类
# =============================================================================

@configclass
class UniDiffDriveActionCfg(mdp.actions.JointVelocityActionCfg):
    class_type = UniDiffDriveAction
    asset_name: str = "robot"
    joint_names: list[str] = ["left_wheel_joint", "right_wheel_joint"]
    # [架构师修正 2026-01-27] 必须设为 1.0！
    # 因为 UniDiffDriveAction 内部已经完成了从 [0,1] 到 [rad/s] 的物理转换
    # 如果 scale != 1.0，会导致双重缩放，速度失控
    scale: float = 1.0
    use_default_offset: bool = False

@configclass
class RelativeRandomTargetCommandCfg(mdp.UniformPoseCommandCfg):
    class_type = RelativeRandomTargetCommand
    asset_name: str = "robot"
    body_name: str = "base_link"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    # [架构师紧急修复 2026-01-27] 降低初始难度：从3m→1.5m
    # 问题：机器人连路都不会走，3m范围太难
    # 解决：先在幼儿园（1.5m范围）学会基本导航，再扩展
    ranges: mdp.UniformPoseCommandCfg.Ranges = mdp.UniformPoseCommandCfg.Ranges(
        pos_x=(-1.5, 1.5), pos_y=(-1.5, 1.5), pos_z=(0.0, 0.0),  # ✅ 1.5m x 1.5m正方形区域（难度降低75%）
        roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(-math.pi, math.pi)
    )
    debug_vis: bool = False

@configclass
class DashgoActionsCfg:
    wheels = UniDiffDriveActionCfg()

@configclass
class DashgoCommandsCfg:
    target_pose = RelativeRandomTargetCommandCfg()

@configclass
class DashgoObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        history_length = 3

        # [架构师警告 2026-01-27] ⚠️ lidar 必须保持在第一位！
        # 原因：GeoNavPolicy依赖lidar在最前面进行数据切片
        # 风险：如果lidar移到其他位置，网络会将速度数据当成雷达数据
        # 操作：添加/删除观测项时，确保lidar始终是第一个定义的

        # [Sim2Real V3.0] 使用前向 180° 双相机拼接 LiDAR (72维)
        # 说明：保持输入维度不变，但彻底移除实机不存在的后向观测。
        lidar = ObservationTermCfg(
            func=process_forward_lidar,
            params={}  # 无需 sensor_cfg，函数内部直接访问前左/前右双相机
        )

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

        # [优化] 开启观测噪声，增强Sim2Real泛化能力（架构师建议，2026-01-24）
        def __post_init__(self):
            self.enable_corruption = True

    policy = PolicyCfg()


# ============================================================================
# 自定义辅助函数
# ============================================================================

# [架构师新增 2026-01-24] 自定义辅助函数：支持正则匹配的批量障碍物随机化
# 问题：SceneEntityCfg 不支持正则表达式，无法直接匹配 "obs_.*"
# 解决：编写"中间层"函数，先正则匹配找到所有障碍物，再逐个调用官方随机化函数
def randomize_obstacles_by_pattern(env: ManagerBasedRLEnv, env_ids: torch.Tensor, pattern: str, pose_range: dict):
    """
    使用正则表达式匹配障碍物并批量随机化位置

    Args:
        env: 管理型RL环境
        env_ids: 需要重置的环境ID
        pattern: 正则表达式字符串（如 "obs_.*" 匹配所有障碍物）
        pose_range: 位置和旋转范围字典
    """
    import re

    # 1. 遍历场景中的所有资产名称
    all_assets = list(env.scene.keys())

    # 2. 筛选出匹配正则模式的资产 (例如 "obs_.*" 匹配 "obs_inner_1", "obs_outer_2" 等)
    matched_assets = [name for name in all_assets if re.match(pattern, name)]

    # 3. 对每个匹配到的障碍物执行随机化
    for asset_name in matched_assets:
        # 临时构造 asset_cfg（借用 SceneEntityCfg 来传递名字）
        temp_cfg = SceneEntityCfg(asset_name)

        # 调用官方的随机化函数（利用 GPU 并行处理 env_ids）
        mdp.reset_root_state_uniform(
            env,
            env_ids,
            pose_range=pose_range,
            velocity_range={},  # 静态障碍物不需要速度
            asset_cfg=temp_cfg
        )


def build_terrain_cfg() -> TerrainImporterCfg:
    if USE_AUTOPILOT_FLAT_SCENE:
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=42,
                size=(20.0, 20.0),
                border_width=2.5,
                num_rows=1,
                num_cols=1,
                sub_terrains={
                    "flat": HfRandomUniformTerrainCfg(
                        proportion=1.0,
                        horizontal_scale=0.1,
                        vertical_scale=0.005,
                        noise_range=(0.0, 0.0),
                        noise_step=0.01,
                    ),
                },
                curriculum=False,
            ),
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            size=(20.0, 20.0),
            border_width=2.5,
            num_rows=5,
            num_cols=5,
            sub_terrains={
                "flat": HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    noise_range=(0.0, 0.0),
                    noise_step=0.01,
                ),
                "random_obstacles": HfRandomUniformTerrainCfg(
                    proportion=0.4,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    noise_range=(0.05, 0.2),
                    noise_step=0.01,
                ),
                "maze": HfDiscreteObstaclesTerrainCfg(
                    proportion=0.4,
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=1.0,
                    obstacle_height_range=(0.5, 1.0),
                    obstacle_width_range=(0.5, 2.0),
                    num_obstacles=40,
                ),
            },
            curriculum=True,
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )


# ============================================================================
# 配置类定义
# ============================================================================

@configclass
class DashgoEventsCfg:
    reset_base = EventTermCfg(
        func=reset_root_state_safe_donut,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_radius": 0.2 if USE_AUTOPILOT_GEN1_EASY_RESET else 0.5,
            "max_radius": 0.5 if USE_AUTOPILOT_GEN1_EASY_RESET else 0.8,
        }
    )
    
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0e9, 1.0e9) if USE_AUTOPILOT_FLAT_SCENE else (10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.5, 0.5), 
                "y": (-0.5, 0.5), 
                "yaw": (-math.pi/6, math.pi/6)
            }
        }
    )
    
    randomize_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        }
    )
    
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # [架构师新增 2026-01-24] 障碍物随机化 - 赋予泛化能力
    # 每次重置时，障碍物的位置在原位置基础上随机偏移 +/- 0.5米，随机旋转
    # 逼迫机器人学会看路，而不是背地图，实现真正的泛化能力
    # [API修复 2026-01-24] SceneEntityCfg不支持正则，使用自定义函数
    randomize_obstacles = EventTermCfg(
        func=randomize_obstacles_by_pattern,  # ✅ 自定义函数（支持正则匹配）
        mode="reset",
        params={
            "pattern": "obs_.*",  # 正则表达式：匹配所有名字带 obs_ 的物体
            "pose_range": {
                "x": (-0.2, 0.2) if USE_AUTOPILOT_GEN1_EASY_RESET else (-0.5, 0.5),
                "y": (-0.2, 0.2) if USE_AUTOPILOT_GEN1_EASY_RESET else (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),  # 随机旋转 +/- 180度
            },
        }
    )

    configure_dynamic_obstacles = (
        EventTermCfg(
            func=configure_dynamic_obstacles,
            mode="reset",
            params={
                "asset_names": DYNAMIC_OBSTACLE_ASSET_NAMES,
            },
        )
        if USE_AUTOPILOT_GEN2_DYNAMIC
        else None
    )

    configure_recovery_escape_scenarios = (
        EventTermCfg(
            func=configure_recovery_escape_scenarios,
            mode="reset",
            params={},
        )
        if USE_AUTOPILOT_GEN2_DYNAMIC
        else None
    )

    drive_dynamic_obstacles = (
        EventTermCfg(
            func=animate_dynamic_obstacles,
            mode="interval",
            interval_range_s=(DYNAMIC_OBSTACLE_INTERVAL_S, DYNAMIC_OBSTACLE_INTERVAL_S),
            params={
                "asset_names": DYNAMIC_OBSTACLE_ASSET_NAMES,
                "motion_dt": DYNAMIC_OBSTACLE_INTERVAL_S,
            },
        )
        if USE_AUTOPILOT_GEN2_DYNAMIC
        else None
    )

@configclass
class DashgoSceneV2Cfg(InteractiveSceneCfg):
    terrain = build_terrain_cfg()

    robot = DASHGO_D1_CFG.replace(prim_path="{ENV_REGEX_NS}/Dashgo")

    # GUI 可视化需要至少一个场景光源，否则 RTX 视口会接近纯黑。
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(0.95, 0.95, 0.95),
        ),
    )

    sun_light = AssetBaseCfg(
        prim_path="/World/sunLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(0.95, 0.95, 0.95),
            angle=0.53,
        ),
    )
    
    contact_forces_base = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link", 
        history_length=3, track_air_time=True
    )

    # ============================================================================
    # [Sim2Real V3.0] 前向 180° 双相机方案
    # ============================================================================
    #
    # 设计目标：
    # 1. 与实机 lakibeam 驱动配置的 90°→270° 有效扇区对齐
    # 2. 移除训练中虚假的后向 180° 观测
    # 3. 保持 72 维策略输入不变
    #
    # 实现方式：
    # - 前右相机：中心 -45°，覆盖 [-90°, 0°]
    # - 前左相机：中心 +45°，覆盖 [0°, +90°]
    # - 每台相机 108 列，总计 216 原始射线，对应 72 维最小池化时正好 3:1
    # ============================================================================

    camera_front_right = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link/cam_front_right",
        update_period=0.1,
        height=1,
        width=108,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 12.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.22),
            rot=(0.9238795, 0.0, 0.0, -0.3826834),
            convention="world",
        ),
    )

    camera_front_left = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Dashgo/base_link/cam_front_left",
        update_period=0.1,
        height=1,
        width=108,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 12.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.22),
            rot=(0.9238795, 0.0, 0.0, 0.3826834),
            convention="world",
        ),
    )
    
    obs_inner_1 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_1", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(1.6, 0.0, 0.5)))
    obs_inner_2 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_2", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(1.13, 1.13, 0.5)))
    obs_inner_3 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_3", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.6, 0.5)))
    obs_inner_4 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_4", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.13, 1.13, 0.5)))
    obs_inner_5 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_5", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.6, 0.0, 0.5)))
    obs_inner_6 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_6", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.13, -1.13, 0.5)))
    obs_inner_7 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_7", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.6, 0.5)))
    obs_inner_8 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_In_8", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.2)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(1.13, -1.13, 0.5)))
    obs_outer_1 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_1", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(2.3, 0.95, 0.5)))
    obs_outer_2 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_2", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.95, 2.3, 0.5)))
    obs_outer_3 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_3", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.95, 2.3, 0.5)))
    obs_outer_4 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_4", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.3, 0.95, 0.5)))
    obs_outer_5 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_5", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.3, -0.95, 0.5)))
    obs_outer_6 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_6", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.95, -2.3, 0.5)))
    obs_outer_7 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_7", spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.95, -2.3, 0.5)))
    obs_outer_8 = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obs_Out_8", spawn=sim_utils.CylinderCfg(radius=0.1, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), mass_props=sim_utils.MassPropertiesCfg(mass=20.0), collision_props=sim_utils.CollisionPropertiesCfg()), init_state=RigidObjectCfg.InitialStateCfg(pos=(2.3, -0.95, 0.5)))

@configclass
class DashgoRewardsCfg:
    """
    NavRL 风格主干 reward，仅保留适配 DashGo 差速底盘的必要项。

    设计理念：
        - 常数偏置 +1：与上游 NavRL 保持一致
        - 目标方向速度：鼓励沿目标方向前进
        - 静态/动态障碍安全项：对 clearance 取对数，鼓励保持安全余量
        - 平滑项：惩罚速度/角速度的突变
        - 不再沿用旧 DashGo 项目的 stall / orbit / reverse_escape / 大终点奖策略
    """

    navrl_survival = RewardTermCfg(
        func=reward_navrl_survival_bias,
        weight=1.0,
    )

    navrl_goal_velocity = RewardTermCfg(
        func=reward_navrl_goal_velocity,
        weight=1.0,
        params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")},
    )

    navrl_static_safety = RewardTermCfg(
        func=reward_navrl_static_safety,
        weight=1.0,
    )

    navrl_dynamic_safety = RewardTermCfg(
        func=reward_navrl_dynamic_safety,
        weight=1.0,
    )

    navrl_twist_smoothness = RewardTermCfg(
        func=penalty_navrl_twist_smoothness,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # [融合方案: Assistant优化] 日志项不参与训练，但设为1.0方便TensorBoard观察
    log_distance = RewardTermCfg(
        func=log_distance_to_goal,
        weight=0.0,
        params={
            "command_name": "target_pose",
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    log_velocity = RewardTermCfg(
        func=log_linear_velocity,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

@configclass
class DashgoTerminationsCfg:
    time_out = TerminationTermCfg(func=check_time_out, time_out=True)

    # NavRL 风格到点判据：仅按与目标的平面距离终止，不再叠加停稳门槛。
    reach_goal = TerminationTermCfg(
        func=check_reach_goal,
        params={
            "command_name": "target_pose",
            "threshold": REWARD_CONFIG["goal_termination_threshold"],
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    object_collision = TerminationTermCfg(
        func=check_collision_navrl_style,
        params={"static_threshold": 0.3, "dynamic_threshold": 0.3},
    )

# =============================================================================
# [v5.0 Ultimate] 课程学习配置
# =============================================================================

@configclass
class DashgoCurriculumCfg:
    """
    [v5.1 ACL] 自适应课程学习配置

    架构师审计发现：线性课程可能导致机器人陷入瓶颈
    解决方案：基于成功率动态调整难度，保持在ZPD [40%, 80%]

    两种模式选择：
        1. ACL模式（推荐）：根据成功率自动调整
        2. 线性模式（传统）：固定步数线性增加

    选择方法：注释掉不需要的模式
    """
    # [v5.1 ACL] 模式1：自适应课程学习（推荐）
    # 优势：动态调整，避免瓶颈，学习效率+30-50%
    target_adaptive = CurriculumTermCfg(
        func=curriculum_adaptive_distance,
        params={
            "command_name": "target_pose",
            "initial_dist": INITIAL_CURRICULUM_DIST,  # Gen2 动态阶段不再从 1.0m 重新学会走路
            "max_dist": 8.0,              # 毕业难度：8米（专家区）
            "step_size": 0.125 if USE_AUTOPILOT_GEN2_DYNAMIC else 0.25,  # wave19 证明 0.0625 未继续推迟断崖，回到当前最佳 Gen2 基线
            "upgrade_threshold": 0.8,     # wave21 证明放宽到 0.75 仍无法推动升级，回到已验证更稳的基线阈值
            "downgrade_threshold": 0.6,   # 当前最佳证据仍是 60%，更高阈值会让 Gen2 从 model_320.pt 直接塌成 timeout
            "window_size": 100,           # 评估最近100个episode
        }
    )

    # [v5.0 Legacy] 模式2：线性课程学习（传统，已禁用）
    # 优势：可预测，稳定
    # 劣势：可能导致瓶颈（机器人长期失败）
    # target_expansion = CurriculumTermCfg(
    #     func=curriculum_expand_target_range,
    #     params={
    #         "command_name": "target_pose",
    #         "min_limit": 1.5,
    #         "max_limit": 8.0,
    #         "start_step": 0,
    #         "end_step": 300_000_000,
    #     }
    # )

@configclass
class DashgoNavEnvV2Cfg(ManagerBasedRLEnvCfg):
    decimation = 3
    episode_length_s = 90.0  # ✅ [架构师修正 2026-01-24] 课程学习：从 60s 增加到 90s（1350步），给机器人更多时间绕过障碍物
    scene = DashgoSceneV2Cfg(num_envs=16, env_spacing=15.0)
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

    actions = DashgoActionsCfg()
    observations = DashgoObservationsCfg()
    commands = DashgoCommandsCfg()
    events = DashgoEventsCfg()
    rewards = DashgoRewardsCfg()
    terminations = DashgoTerminationsCfg()
    curriculum = None
