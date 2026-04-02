from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tools.autonomous_training_cycle as cycle


class AutonomousTrainingCycleTest(unittest.TestCase):
    def _read_cycle_status(self, root: Path) -> dict:
        status_files = list((root / "autonomous").rglob("status.json"))
        self.assertEqual(len(status_files), 1)
        return json.loads(status_files[0].read_text(encoding="utf-8"))

    def test_cycle_rejects_pid_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            start_status = {
                "pid": 101,
                "attempt_id": "attempt-a",
                "started_command_hash": "hash-a",
                "run_root": "/tmp/run_a",
                "supervisor_status": "running",
                "abandoned": False,
                "current_run_invalid": False,
            }
            drift_status = {
                "pid": 202,
                "attempt_id": "attempt-a",
                "started_command_hash": "hash-a",
                "run_root": "/tmp/run_a",
                "supervisor_status": "running",
                "abandoned": False,
                "current_run_invalid": False,
            }
            with patch.object(cycle, "ARTIFACTS_ROOT", tmp_root), patch.object(
                cycle, "start_training", return_value=start_status
            ), patch.object(cycle, "refresh_status", return_value=drift_status), patch.object(
                cycle.time, "sleep", lambda *_args, **_kwargs: None
            ):
                result = cycle.main(["--profile", "pilot", "--max-frame-num", "128", "--poll-seconds", "0"])
            self.assertEqual(result, 1)
            payload = self._read_cycle_status(tmp_root)
            self.assertIn("training_pid_drift", payload["failure_reason"])

    def test_cycle_rejects_failed_eval_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            start_status = {
                "pid": 101,
                "attempt_id": "attempt-a",
                "started_command_hash": "hash-a",
                "run_root": "/tmp/run_a",
                "supervisor_status": "running",
                "abandoned": False,
                "current_run_invalid": False,
            }
            completed_status = {
                "pid": 101,
                "attempt_id": "attempt-a",
                "started_command_hash": "hash-a",
                "run_root": "/tmp/run_a",
                "latest_final_checkpoint": "/tmp/run_a/checkpoint_final.pt",
                "supervisor_status": "completed",
                "abandoned": False,
                "current_run_invalid": False,
            }
            fake_completed = subprocess.CompletedProcess(args=["eval"], returncode=1)
            with patch.object(cycle, "ARTIFACTS_ROOT", tmp_root), patch.object(
                cycle, "start_training", return_value=start_status
            ), patch.object(cycle, "refresh_status", return_value=completed_status), patch.object(
                cycle, "run_command_until_valid_artifact",
                return_value=(
                    fake_completed,
                    {
                        "status": "failed",
                        "request": {
                            "suite": "quick",
                            "checkpoint": "/tmp/run_a/checkpoint_final.pt",
                        },
                        "metrics": {},
                        "scenes": [],
                    },
                    ["status='failed' 不是 completed"],
                ),
            ), patch.object(cycle.time, "sleep", lambda *_args, **_kwargs: None):
                result = cycle.main(["--profile", "pilot", "--max-frame-num", "128", "--poll-seconds", "0"])
            self.assertEqual(result, 1)
            payload = self._read_cycle_status(tmp_root)
            self.assertIn("eval_quick_invalid", payload["failure_reason"])


if __name__ == "__main__":
    unittest.main()
