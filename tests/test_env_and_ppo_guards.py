from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path
from types import ModuleType
import sys
import importlib.util

import torch
import torch.nn as nn
from tensordict import TensorDict

if "dashgo_rl.dashgo_env_navrl_official" not in sys.modules:
    dashgo_stub = ModuleType("dashgo_rl.dashgo_env_navrl_official")
    dashgo_stub.DashgoNavOfficialEnvCfg = object
    dashgo_stub.build_dynamic_obstacle_tokens = lambda *args, **kwargs: torch.zeros((1, 5, 10))
    dashgo_stub.build_navrl_terrain_cfg = lambda *args, **kwargs: None
    dashgo_stub.build_navrl_upstream_terrain_cfg = lambda *args, **kwargs: None
    dashgo_stub.terrain_debug_summary = lambda *args, **kwargs: {}
    sys.modules["dashgo_rl.dashgo_env_navrl_official"] = dashgo_stub

from navrl_dashgo.env_adapter import STATE_DIM, build_state_observation, resolve_map_source
from navrl_dashgo.env_adapter import TorchRLDashgoEnv
from navrl_dashgo.ppo import (
    NonFiniteTrainingStateError,
    bootstrap_done_flags,
    ensure_finite_module_grads,
    ensure_finite_tensor,
)
from navrl_dashgo.torchrl_utils import GAE


def load_hf_discrete_obstacle_cfg_class():
    base = Path("/home/gwh/IsaacLab/source/isaaclab/isaaclab/terrains")
    height_base = base / "height_field"

    terrains_pkg = sys.modules.get("isaaclab.terrains")
    if terrains_pkg is None:
        terrains_pkg = ModuleType("isaaclab.terrains")
        terrains_pkg.__path__ = [str(base)]
        sys.modules["isaaclab.terrains"] = terrains_pkg

    height_pkg = sys.modules.get("isaaclab.terrains.height_field")
    if height_pkg is None:
        height_pkg = ModuleType("isaaclab.terrains.height_field")
        height_pkg.__path__ = [str(height_base)]
        sys.modules["isaaclab.terrains.height_field"] = height_pkg

    def load_module(name: str, path: Path):
        module = sys.modules.get(name)
        if module is not None:
            return module
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    load_module("isaaclab.terrains.terrain_generator_cfg", base / "terrain_generator_cfg.py")
    load_module("isaaclab.terrains.height_field.utils", height_base / "utils.py")
    load_module("isaaclab.terrains.height_field.hf_terrains", height_base / "hf_terrains.py")
    module = load_module("isaaclab.terrains.height_field.hf_terrains_cfg", height_base / "hf_terrains_cfg.py")
    return module.HfDiscreteObstaclesTerrainCfg


class EnvAndPPOGuardsTest(unittest.TestCase):
    def _build_fake_torchrl_env(self, *, current_obs: torch.Tensor, reset_obs: torch.Tensor, episode_lengths: torch.Tensor):
        class FakeObservationManager:
            def __init__(self, obs: torch.Tensor) -> None:
                self.obs = obs
                self.compute_calls = 0

            def compute(self):
                self.compute_calls += 1
                return {"policy": self.obs.clone()}

        class FakeBaseEnv:
            def __init__(self) -> None:
                self.device = torch.device("cpu")
                self.observation_manager = FakeObservationManager(current_obs)
                self.episode_length_buf = episode_lengths.clone()
                self.reset_calls: list[torch.Tensor | None] = []
                self.reset_obs = reset_obs

            def reset(self, env_ids=None):
                if env_ids is None:
                    self.reset_calls.append(None)
                else:
                    self.reset_calls.append(env_ids.clone())
                return {"policy": self.reset_obs.clone()}, {}

        class FakeAdapter:
            def encode(self, raw_obs):
                obs = raw_obs["policy"].clone()
                return TensorDict({"policy": obs}, batch_size=[obs.shape[0]])

        base_env = FakeBaseEnv()
        fake_env = SimpleNamespace(
            base_env=base_env,
            adapter=FakeAdapter(),
            device=torch.device("cpu"),
        )
        return fake_env, base_env

    def test_resolve_map_source_accepts_known_values(self) -> None:
        self.assertEqual(resolve_map_source("dashgo_official"), "dashgo_official")
        self.assertEqual(resolve_map_source(" NAVRL_UPSTREAM "), "navrl_upstream")

    def test_resolve_map_source_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "未知 env.map_source"):
            resolve_map_source("legacy_map")

    def test_navrl_upstream_source_and_obstacle_cfg_match_supported_api(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src" / "dashgo_rl" / "dashgo_env_navrl_official.py"
        ).read_text(encoding="utf-8")
        self.assertIn('obstacle_height_mode="choice"', source)
        self.assertIn("obstacle_height_range=(1.0, 6.0)", source)
        self.assertIn('color_scheme="none"', source)
        self.assertNotIn("obstacle_height_probability", source)

        obstacle_cls = load_hf_discrete_obstacle_cfg_class()
        obstacle_cfg = obstacle_cls(
            proportion=1.0,
            horizontal_scale=0.1,
            vertical_scale=0.1,
            border_width=0.0,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.4, 1.1),
            obstacle_height_range=(1.0, 6.0),
            num_obstacles=12,
            platform_width=0.0,
        )
        self.assertEqual(obstacle_cfg.obstacle_height_mode, "choice")
        self.assertEqual(tuple(obstacle_cfg.obstacle_height_range), (1.0, 6.0))
        self.assertEqual(int(obstacle_cfg.num_obstacles), 12)

    def test_official_env_keeps_navrl_full_circle_lidar_and_goal_bonus(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "src" / "dashgo_rl" / "dashgo_env_navrl_official.py"
        ).read_text(encoding="utf-8")
        self.assertIn("horizontal_fov_range=(-180.0, 180.0)", source)
        self.assertIn("horizontal_res=360.0 / float(SIM_LIDAR_POLICY_DIM)", source)
        self.assertIn('"goal_reached_bonus_weight": 12.0', source)
        self.assertIn("navrl_goal_reached_bonus = RewardTermCfg(", source)

    def test_build_state_observation_keeps_full_non_lidar_slice(self) -> None:
        policy_obs = torch.arange(246, dtype=torch.float32).reshape(1, -1)
        state = build_state_observation(policy_obs)
        expected = torch.arange(216, 246, dtype=torch.float32).reshape(1, -1)
        self.assertEqual(state.shape, (1, STATE_DIM))
        self.assertTrue(torch.equal(state, expected))

    def test_bootstrap_done_flags_treats_truncated_as_done(self) -> None:
        next_tensordict = TensorDict(
            {
                "terminated": torch.zeros((1, 1, 1), dtype=torch.bool),
                "truncated": torch.ones((1, 1, 1), dtype=torch.bool),
            },
            batch_size=[1, 1],
        )
        done = bootstrap_done_flags(next_tensordict)
        self.assertTrue(bool(done.item()))

    def test_gae_truncation_stops_bootstrap(self) -> None:
        gae = GAE(gamma=1.0, lmbda=1.0)
        reward = torch.zeros((1, 1, 1))
        value = torch.zeros((1, 1, 1))
        next_value = torch.full((1, 1, 1), 5.0)

        running_next = TensorDict(
            {
                "terminated": torch.zeros((1, 1, 1), dtype=torch.bool),
                "truncated": torch.zeros((1, 1, 1), dtype=torch.bool),
            },
            batch_size=[1, 1],
        )
        running_adv, _ = gae(reward, bootstrap_done_flags(running_next), value, next_value)
        self.assertAlmostEqual(float(running_adv.item()), 5.0)

        truncated_next = TensorDict(
            {
                "terminated": torch.zeros((1, 1, 1), dtype=torch.bool),
                "truncated": torch.ones((1, 1, 1), dtype=torch.bool),
            },
            batch_size=[1, 1],
        )
        truncated_adv, truncated_ret = gae(reward, bootstrap_done_flags(truncated_next), value, next_value)
        self.assertAlmostEqual(float(truncated_adv.item()), 0.0)
        self.assertAlmostEqual(float(truncated_ret.item()), 0.0)

    def test_ensure_finite_tensor_raises_on_nan(self) -> None:
        with self.assertRaisesRegex(NonFiniteTrainingStateError, "含非有限值"):
            ensure_finite_tensor("collector.reward", torch.tensor([float("nan")]))

    def test_ensure_finite_module_grads_raises_on_nan(self) -> None:
        module = nn.Linear(1, 1)
        module.weight.grad = torch.full_like(module.weight, float("nan"))
        module.bias.grad = torch.zeros_like(module.bias)
        with self.assertRaisesRegex(NonFiniteTrainingStateError, "grad 含非有限值"):
            ensure_finite_module_grads("actor", module)

    def test_partial_reset_reuses_current_obs_after_isaac_autoreset(self) -> None:
        current_obs = torch.ones((3, 246), dtype=torch.float32)
        reset_obs = torch.full((3, 246), 9.0, dtype=torch.float32)
        fake_env, base_env = self._build_fake_torchrl_env(
            current_obs=current_obs,
            reset_obs=reset_obs,
            episode_lengths=torch.tensor([0, 5, 0], dtype=torch.long),
        )
        tensordict = TensorDict({"_reset": torch.tensor([[True], [False], [True]], dtype=torch.bool)}, batch_size=[3])

        result = TorchRLDashgoEnv._reset(fake_env, tensordict)

        self.assertEqual(base_env.reset_calls, [])
        self.assertEqual(base_env.observation_manager.compute_calls, 1)
        self.assertTrue(torch.equal(result["policy"], current_obs))

    def test_partial_reset_calls_base_env_reset_when_env_not_yet_reset(self) -> None:
        current_obs = torch.ones((3, 246), dtype=torch.float32)
        reset_obs = torch.full((3, 246), 7.0, dtype=torch.float32)
        fake_env, base_env = self._build_fake_torchrl_env(
            current_obs=current_obs,
            reset_obs=reset_obs,
            episode_lengths=torch.tensor([4, 5, 3], dtype=torch.long),
        )
        tensordict = TensorDict({"_reset": torch.tensor([[True], [False], [True]], dtype=torch.bool)}, batch_size=[3])

        result = TorchRLDashgoEnv._reset(fake_env, tensordict)

        self.assertEqual(len(base_env.reset_calls), 1)
        self.assertTrue(torch.equal(base_env.reset_calls[0], torch.tensor([0, 2], dtype=torch.long)))
        self.assertEqual(base_env.observation_manager.compute_calls, 0)
        self.assertTrue(torch.equal(result["policy"], reset_obs))


if __name__ == "__main__":
    unittest.main()
