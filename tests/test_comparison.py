from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from navrl_dashgo.comparison import (
    build_metric_rows,
    render_markdown_report,
    resolve_online_baseline_checkpoint,
    termination_counts,
    validate_comparison_payload,
    validate_eval_payload,
)


class ComparisonHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline_payload = {
            "status": "completed",
            "request": {
                "suite": "quick",
                "checkpoint": "/tmp/baseline.pt",
            },
            "metrics": {
                "success_rate": 0.5,
                "collision_rate": 0.2,
                "timeout_rate": 0.3,
                "time_to_goal": 10.0,
                "path_efficiency": 0.6,
                "spin_proxy_rate": 0.1,
                "progress_stall_rate": 0.2,
                "score": 12.0,
            },
            "scenes": [
                {"termination_reason": "reach_goal", "end_distance": 0.2},
                {"termination_reason": "time_out", "end_distance": 1.4},
            ],
        }
        self.candidate_payload = {
            "status": "failed",
            "request": {
                "suite": "quick",
                "checkpoint": "/tmp/candidate.pt",
            },
            "metrics": {
                "success_rate": 0.0,
                "collision_rate": 0.8,
                "timeout_rate": 0.2,
                "time_to_goal": 0.0,
                "path_efficiency": 0.1,
                "spin_proxy_rate": 0.0,
                "progress_stall_rate": 0.7,
                "score": -20.0,
                "orbit_score": 0.3,
            },
            "scenes": [
                {"termination_reason": "object_collision", "end_distance": 1.8},
                {"termination_reason": "object_collision", "end_distance": 2.0},
            ],
        }

    def test_build_metric_rows(self) -> None:
        rows = build_metric_rows(self.baseline_payload, self.candidate_payload)
        row_map = {row["metric"]: row for row in rows}
        self.assertAlmostEqual(row_map["success_rate"]["delta"], -0.5)
        self.assertAlmostEqual(row_map["score"]["delta"], -32.0)

    def test_termination_counts(self) -> None:
        counts = termination_counts(self.candidate_payload)
        self.assertEqual(counts, {"object_collision": 2})

    def test_render_markdown_report(self) -> None:
        rows = build_metric_rows(self.baseline_payload, self.candidate_payload)
        report = render_markdown_report(
            suite="quick",
            baseline_payload=self.baseline_payload,
            candidate_payload=self.candidate_payload,
            rows=rows,
            baseline_source="baseline.pt",
            candidate_source="candidate.pt",
            generated_on="2026-04-03",
        )
        self.assertIn("NavRL-style 候选整体劣于基线", report)
        self.assertIn("object_collision", report)
        self.assertIn("生成时间: 2026-04-03", report)

    def test_validate_eval_payload_rejects_failed_status(self) -> None:
        errors = validate_eval_payload(
            self.candidate_payload,
            expected_suite="quick",
            expected_checkpoint="/tmp/candidate.pt",
        )
        self.assertTrue(errors)
        self.assertIn("status='failed' 不是 completed", errors[0])

    def test_validate_eval_payload_can_allow_failed_status_for_comparison(self) -> None:
        errors = validate_eval_payload(
            self.candidate_payload,
            expected_suite="quick",
            expected_checkpoint="/tmp/candidate.pt",
            allowed_statuses=("completed", "failed"),
        )
        self.assertFalse(errors)

    def test_validate_comparison_payload_rejects_failed_candidate(self) -> None:
        rows = build_metric_rows(self.baseline_payload, self.candidate_payload)
        payload = {
            "status": "completed",
            "suite": "quick",
            "baseline": self.baseline_payload,
            "candidate": self.candidate_payload,
            "rows": rows,
        }
        errors = validate_comparison_payload(payload, expected_suite="quick")
        self.assertTrue(errors)
        self.assertTrue(any("candidate:" in item for item in errors))

    def test_validate_comparison_payload_can_allow_failed_eval_statuses(self) -> None:
        rows = build_metric_rows(self.baseline_payload, self.candidate_payload)
        payload = {
            "status": "completed",
            "suite": "quick",
            "baseline": {**self.baseline_payload, "status": "failed"},
            "candidate": self.candidate_payload,
            "rows": rows,
        }
        errors = validate_comparison_payload(
            payload,
            expected_suite="quick",
            allowed_eval_statuses=("completed", "failed"),
        )
        self.assertFalse(errors)

    def test_compare_models_cli_accepts_failed_eval_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            baseline_json = tmp_root / "baseline.json"
            candidate_json = tmp_root / "candidate.json"
            out_json = tmp_root / "compare.json"
            out_md = tmp_root / "compare.md"
            baseline_json.write_text(
                json.dumps({**self.baseline_payload, "status": "failed"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            candidate_json.write_text(json.dumps(self.candidate_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "tools" / "compare_models.py"),
                    "--suite",
                    "quick",
                    "--baseline-json",
                    str(baseline_json),
                    "--candidate-json",
                    str(candidate_json),
                    "--json-out",
                    str(out_json),
                    "--report-out",
                    str(out_md),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertTrue(out_md.exists())
            self.assertIn("基线状态: `failed`，候选状态: `failed`", out_md.read_text(encoding="utf-8"))

    def test_resolve_online_baseline_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            checkpoint = tmp_root / "model.pt"
            manifest = tmp_root / "manifest.json"
            checkpoint.write_text("stub", encoding="utf-8")
            manifest.write_text(f'{{"checkpoint_path": "{checkpoint}"}}', encoding="utf-8")
            self.assertEqual(resolve_online_baseline_checkpoint(manifest), checkpoint.resolve())


if __name__ == "__main__":
    unittest.main()
