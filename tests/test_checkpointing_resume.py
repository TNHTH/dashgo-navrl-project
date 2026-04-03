from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from navrl_dashgo.checkpointing import (
    build_checkpoint_payload,
    load_training_checkpoint,
    resolve_frame_count,
    resolve_remaining_frames,
)


class DummyAlgo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Linear(3, 4)
        self.actor = nn.Linear(4, 2)
        self.critic = nn.Linear(4, 1)
        self.value_norm = nn.LayerNorm(1)
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=1.0e-3)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=1.0e-3)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1.0e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_extractor(x)
        _ = self.actor(feature)
        return self.critic(feature)


def populate_optimizer_state(algo: DummyAlgo) -> None:
    batch = torch.randn(8, 3)
    feature = algo.feature_extractor(batch)
    actor_term = algo.actor(feature).sum()
    critic_term = algo.critic(feature).sum()
    loss = actor_term + critic_term
    loss.backward()
    algo.feature_extractor_optim.step()
    algo.actor_optim.step()
    algo.critic_optim.step()
    algo.feature_extractor_optim.zero_grad()
    algo.actor_optim.zero_grad()
    algo.critic_optim.zero_grad()


class CheckpointingResumeTest(unittest.TestCase):
    def test_build_checkpoint_payload_includes_optimizer_state(self) -> None:
        algo = DummyAlgo()
        populate_optimizer_state(algo)

        payload = build_checkpoint_payload(algo, config={"profile": "formal"}, frame_count=1234, profile="formal")

        self.assertEqual(payload["checkpoint_version"], 2)
        self.assertEqual(payload["frame_count"], 1234)
        self.assertIn("optimizer_state_dict", payload)
        self.assertIn("feature_extractor", payload["optimizer_state_dict"])
        self.assertTrue(payload["optimizer_state_dict"]["feature_extractor"]["state"])

    def test_load_training_checkpoint_restores_legacy_payload_without_optimizer(self) -> None:
        source = DummyAlgo()
        populate_optimizer_state(source)
        legacy_payload = build_checkpoint_payload(source, config={"profile": "formal"}, frame_count=4096, profile="formal")
        legacy_payload.pop("optimizer_state_dict")

        target = DummyAlgo()
        for param in target.parameters():
            nn.init.constant_(param, 0.0)

        notes = load_training_checkpoint(target, legacy_payload)

        self.assertIn("optimizer_state_missing=all", notes)
        self.assertTrue(torch.allclose(target.feature_extractor.weight, source.feature_extractor.weight))
        self.assertEqual(resolve_frame_count(legacy_payload), 4096)

    def test_load_training_checkpoint_restores_optimizer_state(self) -> None:
        source = DummyAlgo()
        populate_optimizer_state(source)
        payload = build_checkpoint_payload(source, config={"profile": "formal"}, frame_count=8192, profile="formal")

        target = DummyAlgo()
        notes = load_training_checkpoint(target, payload)

        self.assertEqual(notes, [])
        self.assertTrue(torch.allclose(target.actor.weight, source.actor.weight))
        self.assertTrue(target.actor_optim.state_dict()["state"])
        self.assertEqual(resolve_remaining_frames(16384, 8192), 8192)

    def test_resolve_remaining_frames_rejects_non_positive_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须大于 checkpoint.frame_count"):
            resolve_remaining_frames(8192, 8192)


if __name__ == "__main__":
    unittest.main()
