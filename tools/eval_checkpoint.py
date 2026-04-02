#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dashgo_rl.project_paths import ISAAC_PYTHON
from navrl_dashgo.types import EvalRequest, EvalResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo NavRL checkpoint 评测")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=["quick", "main"], default="quick")
    parser.add_argument("--requested-episodes", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--allow-concurrent", action="store_true", help="允许与训练并发启动第二个 Isaac 实例")
    return parser


def detect_active_training() -> list[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    pids: list[int] = []
    target = str(PROJECT_ROOT / "apps" / "isaac" / "train_navrl.py")
    for line in result.stdout.splitlines():
        if target not in line:
            continue
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            pids.append(int(parts[0]))
        except ValueError:
            continue
    return pids


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    request = EvalRequest(
        checkpoint=args.checkpoint.resolve(),
        suite=args.suite,
        project_root=PROJECT_ROOT,
        requested_episodes=args.requested_episodes,
        notes=["DashGo NavRL 评测入口"],
    )

    if not args.checkpoint.exists():
        result = EvalResult(
            status="failed",
            request=request,
            notes=["checkpoint 文件不存在。"],
            metadata={"checkpoint_exists": False},
        )
        payload = result.to_dict()
        if args.json_out is not None:
            args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    active_training_pids = detect_active_training()
    if active_training_pids and not args.allow_concurrent:
        result = EvalResult(
            status="failed",
            request=request,
            notes=[
                "检测到当前仓库训练仍在运行，已阻止并发启动第二个 Isaac 实例。",
                f"active_training_pids={active_training_pids}",
                "如需强制并发，请追加 --allow-concurrent；但在 RTX 4060 8GB 上不推荐。",
            ],
            metadata={"active_training_pids": active_training_pids, "blocked_by_concurrent_training": True},
        )
        payload = result.to_dict()
        if args.json_out is not None:
            args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    json_out = args.json_out.resolve() if args.json_out is not None else PROJECT_ROOT / "artifacts" / "eval" / f"{args.checkpoint.stem}_{args.suite}.json"
    worker = PROJECT_ROOT / "apps" / "isaac" / "eval_worker.py"
    command = [
        str(ISAAC_PYTHON),
        str(worker),
        "--headless",
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--suite",
        args.suite,
        "--json-out",
        str(json_out),
    ]
    if args.requested_episodes is not None:
        command.extend(["--requested-episodes", str(args.requested_episodes)])

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    if json_out.exists():
        payload = json.loads(json_out.read_text(encoding="utf-8"))
    else:
        payload = {
            "status": "failed",
            "request": request.to_dict(),
            "notes": ["评测 worker 未产出 JSON。", completed.stderr.strip() or completed.stdout.strip()],
            "metadata": {"returncode": completed.returncode},
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
