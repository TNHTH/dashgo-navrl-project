#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_event(log_path: Path, event: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": now_iso(), **event}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_background_train(command: str, profile: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "background_train.py"), command, "--profile", profile],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description="值守后台训练直到指定绝对时间")
    parser.add_argument("--profile", required=True, choices=["smoke", "pilot", "main", "formal"])
    parser.add_argument("--until", required=True, help="绝对时间，格式示例: 2026-04-03T13:10:00+08:00")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args()

    target_at = datetime.fromisoformat(args.until)
    append_event(
        args.log_path,
        {
            "event": "watchdog_started",
            "target_stop_at": target_at.isoformat(timespec="seconds"),
            "poll_seconds": args.poll_seconds,
        },
    )

    while True:
        status = run_background_train("status", args.profile)
        append_event(
            args.log_path,
            {
                "event": "heartbeat",
                "supervisor_status": status.get("supervisor_status"),
                "active_process_count": status.get("active_process_count"),
                "run_root": status.get("run_root"),
                "latest_final_checkpoint": status.get("latest_final_checkpoint"),
            },
        )

        now = datetime.now(target_at.tzinfo)
        if now >= target_at:
            if int(status.get("active_process_count", 0)) > 0:
                stop_status = run_background_train("stop", args.profile)
                append_event(
                    args.log_path,
                    {
                        "event": "stop_requested_at_target",
                        "status_after_stop": stop_status,
                    },
                )
            else:
                append_event(
                    args.log_path,
                    {
                        "event": "target_reached_without_active_process",
                        "final_status": status,
                    },
                )
            return 0

        if int(status.get("active_process_count", 0)) <= 0 and status.get("supervisor_status") != "running":
            append_event(
                args.log_path,
                {
                    "event": "early_exit_detected",
                    "final_status": status,
                },
            )
            return 0

        time.sleep(max(int(args.poll_seconds), 5))


if __name__ == "__main__":
    raise SystemExit(main())
