from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from dashgo_rl.dashgo_env_navrl_official import (
    DashgoNavOfficialEnvCfg,
    build_dynamic_obstacle_tokens,
    build_navrl_terrain_cfg,
    build_navrl_upstream_terrain_cfg,
    terrain_debug_summary,
)
from .semantics import restore_flat_history, restore_lidar_history


POLICY_OBS_DIM = 246
LIDAR_HISTORY_DIM = 216
LIDAR_SECTORS = 72
LIDAR_HISTORY = 3
STATE_DIM = 8
MAX_DYNAMIC_OBS = 5
DYNAMIC_TOKEN_DIM = 10
ALLOWED_MAP_SOURCES = {"dashgo_official", "navrl_upstream"}


def _collapse_reset_mask(reset_mask: torch.Tensor) -> torch.Tensor:
    """把 TorchRL 传入的 reset 标记压成 [num_envs] 布尔掩码。"""
    mask = reset_mask.to(dtype=torch.bool)
    while mask.ndim > 1:
        mask = mask.any(dim=-1)
    return mask


def _current_raw_obs(base_env) -> dict[str, torch.Tensor]:
    """读取当前仿真状态对应的最新观测。"""
    cached = getattr(base_env, "obs_buf", None)
    if isinstance(cached, dict) and "policy" in cached:
        return cached
    return base_env.observation_manager.compute()


def _resolve_reset_env_ids(tensordict: TensorDict | None, device: torch.device) -> torch.Tensor | None:
    if tensordict is None:
        return None
    reset_mask = tensordict.get("_reset", None)
    if reset_mask is None:
        return None
    flat_mask = _collapse_reset_mask(reset_mask)
    env_ids = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
    return env_ids.to(device=device, dtype=torch.long)


def _envs_already_autoreset(base_env, env_ids: torch.Tensor | None) -> bool:
    """Isaac Lab step() 已对 done env 做过 reset 时，episode_length_buf 会回到 0。"""
    if env_ids is None or env_ids.numel() == 0:
        return False
    episode_length_buf = getattr(base_env, "episode_length_buf", None)
    if episode_length_buf is None:
        return False
    return bool(torch.all(episode_length_buf[env_ids] == 0))

@dataclass
class ObservationSlices:
    lidar_end: int = 216
    waypoint_end: int = 225
    goal_end: int = 234
    lin_vel_end: int = 237
    yaw_rate_end: int = 240
    last_action_end: int = 246

class DashgoTensorAdapter:
    def __init__(self, env) -> None:
        self.env = env
        self.device = env.device
        self.slices = ObservationSlices()

    def encode(self, raw_obs: dict[str, torch.Tensor]) -> TensorDict:
        policy_obs = raw_obs["policy"]
        if policy_obs.shape[-1] != POLICY_OBS_DIM:
            raise ValueError(f"意外的 DashGo policy 观测维度: {policy_obs.shape}")

        num_envs = policy_obs.shape[0]
        policy_obs = torch.nan_to_num(policy_obs, nan=0.0, posinf=0.0, neginf=0.0)

        lidar = restore_lidar_history(
            policy_obs[:, : self.slices.lidar_end],
            history_length=LIDAR_HISTORY,
            num_sectors=LIDAR_SECTORS,
        )
        waypoint_hist = restore_flat_history(
            policy_obs[:, self.slices.lidar_end : self.slices.waypoint_end],
            history_length=LIDAR_HISTORY,
            feature_dim=3,
        )
        goal_hist = restore_flat_history(
            policy_obs[:, self.slices.waypoint_end : self.slices.goal_end],
            history_length=LIDAR_HISTORY,
            feature_dim=3,
        )
        lin_vel_hist = policy_obs[:, self.slices.goal_end : self.slices.lin_vel_end]
        yaw_rate_hist = policy_obs[:, self.slices.lin_vel_end : self.slices.yaw_rate_end]

        state = torch.cat(
            [
                waypoint_hist[:, -1, :],
                goal_hist[:, -1, :],
                lin_vel_hist[:, -1:].clone(),
                yaw_rate_hist[:, -1:].clone(),
            ],
            dim=-1,
        ).reshape(num_envs, 1, STATE_DIM)

        dynamic_obstacle = self._build_dynamic_obstacle_tokens().reshape(num_envs, 1, MAX_DYNAMIC_OBS, DYNAMIC_TOKEN_DIM)

        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": TensorDict(
                            {
                                "state": state,
                                "lidar": lidar,
                                "dynamic_obstacle": dynamic_obstacle,
                            },
                            batch_size=[num_envs, 1],
                            device=self.device,
                        )
                    },
                    batch_size=[num_envs, 1],
                    device=self.device,
                )
            },
            batch_size=[num_envs],
            device=self.device,
        )

    def _build_dynamic_obstacle_tokens(self) -> torch.Tensor:
        return build_dynamic_obstacle_tokens(self.env, max_tokens=MAX_DYNAMIC_OBS)


def resolve_map_source(value: Any) -> str:
    map_source = str(value if value is not None else "dashgo_official").strip().lower()
    if map_source not in ALLOWED_MAP_SOURCES:
        raise ValueError(f"未知 env.map_source={map_source!r}，允许值: {sorted(ALLOWED_MAP_SOURCES)}")
    return map_source


class TorchRLDashgoEnv(EnvBase):
    def __init__(self, cfg: Any):
        self.cfg = cfg
        autopilot_profile = str(cfg.env.autopilot_profile)
        os.environ["DASHGO_AUTOPILOT_PROFILE"] = autopilot_profile

        from isaaclab.envs import ManagerBasedRLEnv

        env_cfg = DashgoNavOfficialEnvCfg()
        env_cfg.seed = int(cfg.seed)
        env_cfg.scene.num_envs = int(cfg.env.num_envs)
        env_cfg.scene.env_spacing = float(cfg.env.env_spacing)
        env_cfg.episode_length_s = float(cfg.env.episode_length_s)
        map_source = resolve_map_source(getattr(cfg.env, "map_source", "dashgo_official"))
        cfg.env.map_source = map_source
        static_obstacles = int(cfg.env.static_obstacles)
        terrain_summary = terrain_debug_summary(map_source, static_obstacles)
        print(
            "[DashGo-NavRL] env_adapter_resolved "
            f"map_source={map_source} terrain_summary={terrain_summary} "
            f"dynamic_obstacles={int(cfg.env.dynamic_obstacles)}",
            flush=True,
        )
        try:
            if map_source == "navrl_upstream":
                env_cfg.scene.terrain = build_navrl_upstream_terrain_cfg(static_obstacles)
            else:
                env_cfg.scene.terrain = build_navrl_terrain_cfg(static_obstacles)
        except Exception as exc:  # noqa: BLE001
            print(
                "[DashGo-NavRL] failure_reason=terrain_configuration_error "
                f"map_source={map_source} summary={terrain_summary} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            raise RuntimeError(f"terrain_configuration_error map_source={map_source}") from exc
        env_cfg.events.configure_dynamic_obstacles.params["num_active"] = int(cfg.env.dynamic_obstacles)
        try:
            self.base_env = ManagerBasedRLEnv(cfg=env_cfg)
        except Exception as exc:  # noqa: BLE001
            print(
                "[DashGo-NavRL] failure_reason=env_initialization_error "
                f"map_source={map_source} summary={terrain_summary} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            raise RuntimeError(f"env_initialization_error map_source={map_source}") from exc
        self.map_source = map_source
        self.terrain_summary = terrain_summary
        self.adapter = DashgoTensorAdapter(self.base_env)
        self.num_envs = int(self.base_env.num_envs)
        self.n_agents = 1

        super().__init__(device=self.base_env.device, batch_size=torch.Size([self.num_envs]))
        self.observation_spec = Composite(
            agents=Composite(
                observation=Composite(
                    state=Unbounded(shape=torch.Size([self.num_envs, 1, STATE_DIM]), device=self.device),
                    lidar=Unbounded(shape=torch.Size([self.num_envs, 1, LIDAR_SECTORS, LIDAR_HISTORY]), device=self.device),
                    dynamic_obstacle=Unbounded(
                        shape=torch.Size([self.num_envs, 1, MAX_DYNAMIC_OBS, DYNAMIC_TOKEN_DIM]),
                        device=self.device,
                    ),
                    shape=torch.Size([self.num_envs, 1]),
                ),
                shape=torch.Size([self.num_envs, 1]),
            ),
            shape=torch.Size([self.num_envs]),
        )
        self.agent_action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=torch.Size([self.num_envs, 1, 2]),
            device=self.device,
        )
        self.action_spec = Composite(
            agents=Composite(
                action=self.agent_action_spec,
                shape=torch.Size([self.num_envs, 1]),
            ),
            shape=torch.Size([self.num_envs]),
        )
        self.reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(shape=torch.Size([self.num_envs, 1, 1]), device=self.device),
                shape=torch.Size([self.num_envs, 1]),
            ),
            shape=torch.Size([self.num_envs]),
        )
        self.done_spec = Composite(
            done=Bounded(
                low=0,
                high=1,
                shape=torch.Size([self.num_envs, 1]),
                dtype=torch.bool,
                device=self.device,
            ),
            terminated=Bounded(
                low=0,
                high=1,
                shape=torch.Size([self.num_envs, 1]),
                dtype=torch.bool,
                device=self.device,
            ),
            truncated=Bounded(
                low=0,
                high=1,
                shape=torch.Size([self.num_envs, 1]),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=torch.Size([self.num_envs]),
        )
        self.max_episode_length = int(self.base_env.max_episode_length)

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        reset_env_ids = _resolve_reset_env_ids(tensordict, self.device)
        if tensordict is None:
            raw_obs, _ = self.base_env.reset()
        elif reset_env_ids is not None and reset_env_ids.numel() > 0:
            if _envs_already_autoreset(self.base_env, reset_env_ids):
                raw_obs = _current_raw_obs(self.base_env)
            else:
                raw_obs, _ = self.base_env.reset(env_ids=reset_env_ids)
        else:
            raw_obs = _current_raw_obs(self.base_env)
        return self.adapter.encode(raw_obs)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["agents", "action"]
        env_action = action.squeeze(1) if action.ndim == 3 else action
        step_ret = self.base_env.step(env_action)
        if len(step_ret) == 5:
            raw_obs, reward, terminated, truncated, _ = step_ret
            done = terminated | truncated
        else:
            raw_obs, reward, done, _ = step_ret
            terminated = done
            truncated = torch.zeros_like(done, dtype=torch.bool)

        next_td = self.adapter.encode(raw_obs)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0).reshape(self.num_envs, 1, 1)
        next_td["agents", "reward"] = reward.to(self.device)
        next_td["done"] = done.reshape(self.num_envs, 1).to(torch.bool)
        next_td["terminated"] = terminated.reshape(self.num_envs, 1).to(torch.bool)
        next_td["truncated"] = truncated.reshape(self.num_envs, 1).to(torch.bool)
        return next_td

    def _set_seed(self, seed: int | None) -> int | None:
        if seed is None:
            return None
        torch.manual_seed(seed)
        return seed

    def close(self) -> None:
        self.base_env.close()
        super().close()
