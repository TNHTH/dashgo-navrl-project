from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from dashgo_rl.dashgo_env_navrl_official import DashgoNavOfficialEnvCfg, build_dynamic_obstacle_tokens


POLICY_OBS_DIM = 246
LIDAR_HISTORY_DIM = 216
LIDAR_SECTORS = 72
LIDAR_HISTORY = 3
STATE_DIM = 8
MAX_DYNAMIC_OBS = 5
DYNAMIC_TOKEN_DIM = 10

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

        lidar = policy_obs[:, : self.slices.lidar_end].reshape(num_envs, 1, LIDAR_SECTORS, LIDAR_HISTORY)
        waypoint_hist = policy_obs[:, self.slices.lidar_end : self.slices.waypoint_end].reshape(num_envs, 3, 3)
        goal_hist = policy_obs[:, self.slices.waypoint_end : self.slices.goal_end].reshape(num_envs, 3, 3)
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
        from dashgo_rl.dashgo_env_navrl_official import build_navrl_terrain_cfg

        env_cfg.scene.terrain = build_navrl_terrain_cfg(int(cfg.env.static_obstacles))
        env_cfg.events.configure_dynamic_obstacles.params["num_active"] = int(cfg.env.dynamic_obstacles)
        self.base_env = ManagerBasedRLEnv(cfg=env_cfg)
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
        raw_obs, _ = self.base_env.reset()
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
