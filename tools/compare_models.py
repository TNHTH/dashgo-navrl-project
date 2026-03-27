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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对比旧 DashGo 模型与新 NavRL 模型")
    parser.add_argument("--baseline-root", type=Path, default=Path("/home/gwh/dashgo_rl_project"))
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=["quick", "main"], default="quick")
    parser.add_argument("--requested-episodes", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def run_eval(script: Path, checkpoint: Path, suite: str, requested_episodes: int | None, json_out: Path) -> dict:
    command = [sys.executable, str(script), "--checkpoint", str(checkpoint), "--suite", suite, "--json-out", str(json_out)]
    if requested_episodes is not None:
        command.extend(["--requested-episodes", str(requested_episodes)])
    completed = subprocess.run(command, cwd=script.parent.parent, check=False, capture_output=True, text=True)
    if not json_out.exists():
        raise RuntimeError(f"评测未产出 JSON: {script}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return json.loads(json_out.read_text(encoding="utf-8"))


def metric_line(name: str, baseline: dict, candidate: dict) -> dict:
    base_val = baseline.get(name)
    cand_val = candidate.get(name)
    delta = None if base_val is None or cand_val is None else cand_val - base_val
    return {"metric": name, "baseline": base_val, "candidate": cand_val, "delta": delta}


def print_table(rows: list[dict]) -> None:
    header = f"{'metric':<28} {'baseline':>12} {'candidate':>12} {'delta':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        base = "n/a" if row["baseline"] is None else f"{row['baseline']:.4f}"
        cand = "n/a" if row["candidate"] is None else f"{row['candidate']:.4f}"
        delta = "n/a" if row["delta"] is None else f"{row['delta']:+.4f}"
        print(f"{row['metric']:<28} {base:>12} {cand:>12} {delta:>12}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    baseline_script = args.baseline_root / "tools" / "diagnostics" / "eval_checkpoint.py"
    candidate_script = PROJECT_ROOT / "tools" / "eval_checkpoint.py"
    eval_root = PROJECT_ROOT / "artifacts" / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    baseline_json = eval_root / f"baseline_{args.baseline_checkpoint.stem}_{args.suite}.json"
    candidate_json = eval_root / f"candidate_{args.candidate_checkpoint.stem}_{args.suite}.json"
    baseline_payload = run_eval(
        baseline_script,
        args.baseline_checkpoint.resolve(),
        args.suite,
        args.requested_episodes,
        baseline_json,
    )
    candidate_payload = run_eval(
        candidate_script,
        args.candidate_checkpoint.resolve(),
        args.suite,
        args.requested_episodes,
        candidate_json,
    )

    baseline_metrics = (baseline_payload.get("metrics") or {})
    candidate_metrics = (candidate_payload.get("metrics") or {})
    metric_names = [
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "time_to_goal",
        "path_efficiency",
        "spin_proxy_rate",
        "progress_stall_rate",
        "score",
    ]
    rows = [metric_line(name, baseline_metrics, candidate_metrics) for name in metric_names]
    print_table(rows)

    payload = {
        "suite": args.suite,
        "baseline": baseline_payload,
        "candidate": candidate_payload,
        "rows": rows,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

