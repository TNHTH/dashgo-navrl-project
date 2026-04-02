#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from navrl_dashgo.comparison import (
    build_metric_rows,
    read_json,
    render_markdown_report,
    resolve_online_baseline_checkpoint,
    validate_comparison_payload,
    validate_eval_payload,
)


DEFAULT_BASELINE_MANIFEST = Path(
    "/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/models/policy_torchscript.manifest.json"
)
COMPARE_ALLOWED_EVAL_STATUSES = ("completed", "failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对比旧 DashGo 模型与新 NavRL 模型")
    parser.add_argument("--baseline-root", type=Path, default=Path("/home/gwh/dashgo_rl_project"))
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--candidate-checkpoint", type=Path, default=None)
    parser.add_argument("--candidate-json", type=Path, default=None)
    parser.add_argument("--suite", choices=["quick", "main"], default="quick")
    parser.add_argument("--requested-episodes", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def run_eval(script: Path, checkpoint: Path, suite: str, requested_episodes: int | None, json_out: Path) -> dict:
    command = [sys.executable, str(script), "--checkpoint", str(checkpoint), "--suite", suite, "--json-out", str(json_out)]
    if requested_episodes is not None:
        command.extend(["--requested-episodes", str(requested_episodes)])
    completed = subprocess.run(command, cwd=script.parent.parent, check=False, capture_output=True, text=True)
    if not json_out.exists():
        raise RuntimeError(f"评测未产出 JSON: {script}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return json.loads(json_out.read_text(encoding="utf-8"))


def print_table(rows: list[dict]) -> None:
    header = f"{'metric':<28} {'baseline':>12} {'candidate':>12} {'delta':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        base = "n/a" if row["baseline"] is None else f"{row['baseline']:.4f}"
        cand = "n/a" if row["candidate"] is None else f"{row['candidate']:.4f}"
        delta = "n/a" if row["delta"] is None else f"{row['delta']:+.4f}"
        print(f"{row['metric']:<28} {base:>12} {cand:>12} {delta:>12}")


def load_payload(
    *,
    eval_script: Path,
    checkpoint: Path | None,
    eval_json: Path | None,
    suite: str,
    requested_episodes: int | None,
    default_json: Path,
) -> tuple[dict, str]:
    if eval_json is not None:
        payload = read_json(eval_json.resolve())
        errors = validate_eval_payload(
            payload,
            expected_suite=suite,
            expected_checkpoint=checkpoint,
            allowed_statuses=COMPARE_ALLOWED_EVAL_STATUSES,
        )
        if errors:
            raise ValueError(f"评测 JSON 非法: {eval_json.resolve()} -> {errors}")
        return payload, str(eval_json.resolve())
    if checkpoint is None:
        raise ValueError("缺少 checkpoint 或现成评测 JSON。")
    payload = run_eval(eval_script, checkpoint.resolve(), suite, requested_episodes, default_json)
    errors = validate_eval_payload(
        payload,
        expected_suite=suite,
        expected_checkpoint=checkpoint,
        allowed_statuses=COMPARE_ALLOWED_EVAL_STATUSES,
    )
    if errors:
        raise ValueError(f"评测结果非法: checkpoint={checkpoint.resolve()} -> {errors}")
    return payload, str(checkpoint.resolve())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    baseline_script = args.baseline_root / "tools" / "diagnostics" / "eval_checkpoint.py"
    candidate_script = PROJECT_ROOT / "tools" / "eval_checkpoint.py"
    eval_root = PROJECT_ROOT / "artifacts" / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    baseline_checkpoint = args.baseline_checkpoint
    if baseline_checkpoint is None and args.baseline_json is None:
        baseline_checkpoint = resolve_online_baseline_checkpoint(args.baseline_manifest.resolve())
    baseline_default_json = eval_root / f"baseline_{Path(str(baseline_checkpoint)).stem if baseline_checkpoint else 'from_json'}_{args.suite}.json"
    candidate_default_json = eval_root / f"candidate_{Path(str(args.candidate_checkpoint)).stem if args.candidate_checkpoint else 'from_json'}_{args.suite}.json"

    payload: dict[str, object]
    try:
        baseline_payload, baseline_source = load_payload(
            eval_script=baseline_script,
            checkpoint=baseline_checkpoint,
            eval_json=args.baseline_json,
            suite=args.suite,
            requested_episodes=args.requested_episodes,
            default_json=baseline_default_json,
        )
        candidate_payload, candidate_source = load_payload(
            eval_script=candidate_script,
            checkpoint=args.candidate_checkpoint,
            eval_json=args.candidate_json,
            suite=args.suite,
            requested_episodes=args.requested_episodes,
            default_json=candidate_default_json,
        )

        rows = build_metric_rows(baseline_payload, candidate_payload)
        payload = {
            "status": "completed",
            "suite": args.suite,
            "baseline_source": baseline_source,
            "candidate_source": candidate_source,
            "baseline": baseline_payload,
            "candidate": candidate_payload,
            "rows": rows,
        }
        compare_errors = validate_comparison_payload(
            payload,
            expected_suite=args.suite,
            expected_candidate_checkpoint=args.candidate_checkpoint,
            allowed_eval_statuses=COMPARE_ALLOWED_EVAL_STATUSES,
        )
        if compare_errors:
            raise ValueError(f"comparison payload 非法: {compare_errors}")
        print_table(rows)
        if args.report_out is not None:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            report = render_markdown_report(
                suite=args.suite,
                baseline_payload=baseline_payload,
                candidate_payload=candidate_payload,
                rows=rows,
                baseline_source=baseline_source,
                candidate_source=candidate_source,
            )
            args.report_out.write_text(report, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "failed",
            "suite": args.suite,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "baseline_source": None,
            "candidate_source": None,
            "baseline": None,
            "candidate": None,
            "rows": [],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
