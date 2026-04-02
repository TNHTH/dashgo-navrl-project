from __future__ import annotations

import unittest

import torch

from navrl_dashgo.semantics import (
    build_reference_path_progress,
    compute_waypoint_lookahead_indices,
    restore_lidar_history,
)


class NavRLSemanticsTest(unittest.TestCase):
    def test_reference_path_progress_reaches_goal_at_last_valid_index(self) -> None:
        progress = build_reference_path_progress(48, torch.tensor([6, 2]))
        self.assertAlmostEqual(float(progress[0, 5].item()), 1.0)
        self.assertLess(float(progress[0, 4].item()), 1.0)
        self.assertAlmostEqual(float(progress[1, 1].item()), 1.0)

    def test_compute_waypoint_lookahead_indices_advances_without_overshoot(self) -> None:
        indices = compute_waypoint_lookahead_indices(
            torch.tensor([2, 44], dtype=torch.long),
            torch.tensor([10, 48], dtype=torch.long),
            lookahead_steps=5,
        )
        self.assertTrue(torch.equal(indices, torch.tensor([7, 47], dtype=torch.long)))

    def test_restore_lidar_history_preserves_history_major_layout(self) -> None:
        restored = restore_lidar_history(
            torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]),
            history_length=2,
            num_sectors=3,
        )
        expected = torch.tensor([[[[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]]])
        self.assertTrue(torch.equal(restored, expected))


if __name__ == "__main__":
    unittest.main()
