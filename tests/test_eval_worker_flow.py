from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def load_eval_worker_module():
    module_name = "dashgo_navrl_eval_worker_test"
    module = globals().get(module_name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parents[1] / "apps" / "isaac" / "eval_worker.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    globals()[module_name] = module
    return module


eval_worker = load_eval_worker_module()


class EvalWorkerFlowTest(unittest.TestCase):
    def test_step_without_auto_reset_keeps_done_state_for_caller(self) -> None:
        class FakeActionManager:
            def __init__(self) -> None:
                self.process_calls = 0
                self.apply_calls = 0

            def process_action(self, action):
                self.process_calls += 1
                self.last_action = action.clone()

            def apply_action(self):
                self.apply_calls += 1

        class FakeRecorderManager:
            def __init__(self) -> None:
                self.active_terms = []
                self.pre_step_calls = 0
                self.post_step_calls = 0

            def record_pre_step(self):
                self.pre_step_calls += 1

            def record_post_step(self):
                self.post_step_calls += 1

        class FakeSim:
            def __init__(self) -> None:
                self.step_calls = 0
                self.render_calls = 0

            def has_gui(self):
                return False

            def has_rtx_sensors(self):
                return False

            def step(self, render=False):
                self.step_calls += 1

            def render(self):
                self.render_calls += 1

        class FakeScene:
            def __init__(self) -> None:
                self.write_calls = 0
                self.update_calls = 0

            def write_data_to_sim(self):
                self.write_calls += 1

            def update(self, dt):
                self.update_calls += 1
                self.last_dt = dt

        class FakeTerminationManager:
            def __init__(self) -> None:
                self.terminated = torch.tensor([True, False], dtype=torch.bool)
                self.time_outs = torch.tensor([False, True], dtype=torch.bool)

            def compute(self):
                return self.terminated | self.time_outs

        class FakeRewardManager:
            def compute(self, dt):
                self.last_dt = dt
                return torch.tensor([1.0, 2.0], dtype=torch.float32)

        class FakeCommandManager:
            def __init__(self) -> None:
                self.compute_calls = 0

            def compute(self, dt):
                self.compute_calls += 1
                self.last_dt = dt

        class FakeEventManager:
            def __init__(self) -> None:
                self.available_modes = {"interval"}
                self.apply_calls = 0

            def apply(self, mode, dt):
                self.apply_calls += 1
                self.last_mode = mode
                self.last_dt = dt

        class FakeObservationManager:
            def __init__(self) -> None:
                self.compute_calls = 0

            def compute(self):
                self.compute_calls += 1
                return {"policy": torch.zeros((2, 246), dtype=torch.float32)}

        env = SimpleNamespace(
            device=torch.device("cpu"),
            cfg=SimpleNamespace(decimation=1, sim=SimpleNamespace(render_interval=1)),
            physics_dt=1.0 / 60.0,
            step_dt=0.05,
            _sim_step_counter=0,
            common_step_counter=0,
            episode_length_buf=torch.zeros(2, dtype=torch.long),
            action_manager=FakeActionManager(),
            recorder_manager=FakeRecorderManager(),
            sim=FakeSim(),
            scene=FakeScene(),
            termination_manager=FakeTerminationManager(),
            reward_manager=FakeRewardManager(),
            command_manager=FakeCommandManager(),
            event_manager=FakeEventManager(),
            observation_manager=FakeObservationManager(),
            extras={"log": {}},
        )

        raw_obs, reward, terminated, truncated, extras = eval_worker.step_env_without_auto_reset(
            env,
            torch.zeros((2, 2), dtype=torch.float32),
        )

        self.assertEqual(env.action_manager.process_calls, 1)
        self.assertEqual(env.action_manager.apply_calls, 1)
        self.assertEqual(env.sim.step_calls, 1)
        self.assertEqual(env.scene.write_calls, 1)
        self.assertEqual(env.scene.update_calls, 1)
        self.assertEqual(env.command_manager.compute_calls, 1)
        self.assertEqual(env.event_manager.apply_calls, 1)
        self.assertEqual(env.observation_manager.compute_calls, 1)
        self.assertTrue(torch.equal(env.episode_length_buf, torch.tensor([1, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(env.reset_buf, torch.tensor([True, True], dtype=torch.bool)))
        self.assertTrue(torch.equal(terminated, torch.tensor([True, False], dtype=torch.bool)))
        self.assertTrue(torch.equal(truncated, torch.tensor([False, True], dtype=torch.bool)))
        self.assertTrue(torch.equal(reward, torch.tensor([1.0, 2.0], dtype=torch.float32)))
        self.assertIn("policy", raw_obs)
        self.assertEqual(extras, {"log": {}})

    def test_reset_done_envs_for_next_episode_recomputes_obs_after_scene_init(self) -> None:
        class FakeObservationManager:
            def __init__(self) -> None:
                self.compute_calls = 0

            def compute(self):
                self.compute_calls += 1
                return {"policy": torch.full((4, 246), 3.0, dtype=torch.float32)}

        class FakeEnv:
            def __init__(self) -> None:
                self.reset_calls: list[torch.Tensor] = []
                self.observation_manager = FakeObservationManager()

            def reset(self, env_ids=None):
                assert env_ids is not None
                self.reset_calls.append(env_ids.clone())
                return {"policy": torch.zeros((4, 246), dtype=torch.float32)}, {}

        env = FakeEnv()
        done_ids = torch.tensor([1, 3], dtype=torch.long)
        scenarios = [{"goal": (1.0, 0.0), "yaw": 0.0, "reverse_case": False}]
        stats = {}

        with patch.object(eval_worker, "initialize_episode_state", return_value=9) as init_mock:
            next_scene_idx, raw_obs = eval_worker.reset_done_envs_for_next_episode(
                env,
                done_ids,
                scenarios,
                7,
                stats,
            )

        self.assertEqual(len(env.reset_calls), 1)
        self.assertTrue(torch.equal(env.reset_calls[0], done_ids))
        init_mock.assert_called_once_with(env, done_ids, scenarios, 7, stats)
        self.assertEqual(env.observation_manager.compute_calls, 1)
        self.assertEqual(next_scene_idx, 9)
        self.assertTrue(torch.equal(raw_obs["policy"], torch.full((4, 246), 3.0, dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
