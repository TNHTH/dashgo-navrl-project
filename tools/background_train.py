#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dashgo_rl.project_paths import ARTIFACTS_ROOT, ISAAC_PYTHON, PROJECT_ROOT as DASHGO_PROJECT_ROOT
from navrl_dashgo.runtime import write_json


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def now_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def supervisor_root(profile: str) -> Path:
    return ARTIFACTS_ROOT / "supervisor" / profile


def status_path(profile: str) -> Path:
    return supervisor_root(profile) / "status.json"


def log_path(profile: str) -> Path:
    return supervisor_root(profile) / "train.log"


def pid_path(profile: str) -> Path:
    return supervisor_root(profile) / "train.pid"


def cmd_path(profile: str) -> Path:
    return supervisor_root(profile) / "train.cmd.json"


def _tail_text(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def _current_attempt_tail(text: str) -> str:
    marker = "[Supervisor] start_at="
    segments = text.split(marker)
    if len(segments) <= 1:
        return text
    return marker + segments[-1]


def _extract_latest_value(text: str, marker: str) -> str | None:
    value = None
    for line in text.splitlines():
        if marker in line:
            value = line.split(marker, 1)[1].strip()
    return value


def _command_hash(command: list[str]) -> str:
    encoded = json.dumps(command, ensure_ascii=False, sort_keys=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_run_status_record(run_root_value: str | None, payload: dict[str, Any]) -> None:
    if not run_root_value:
        return
    run_root = Path(run_root_value)
    run_root.mkdir(parents=True, exist_ok=True)
    record = {
        "profile": payload.get("profile"),
        "run_root": str(run_root),
        "supervisor_status": payload.get("supervisor_status"),
        "abandoned": bool(payload.get("abandoned")),
        "current_run_invalid": bool(payload.get("current_run_invalid")),
        "abandoned_reason": payload.get("abandoned_reason"),
        "abandoned_at": payload.get("abandoned_at"),
        "abandonment_metadata": payload.get("abandonment_metadata") or {},
        "updated_at": payload.get("updated_at"),
    }
    write_json(run_root / "supervisor_status.json", record)


def _process_state(pid: int) -> str | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        check=False,
        capture_output=True,
        text=True,
    )
    state = result.stdout.strip()
    return state or None


def _is_running(pid: int | None) -> bool:
    if pid is None:
        return False
    state = _process_state(pid)
    return bool(state) and "Z" not in state


def _load_status(profile: str) -> dict[str, Any]:
    path = status_path(profile)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_pid(profile: str) -> int | None:
    path = pid_path(profile)
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _base_status(profile: str) -> dict[str, Any]:
    root = supervisor_root(profile)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "profile": profile,
        "project_root": str(DASHGO_PROJECT_ROOT),
        "supervisor_root": str(root),
        "log_path": str(log_path(profile)),
        "pid_path": str(pid_path(profile)),
        "command_path": str(cmd_path(profile)),
        "supervisor_status": "idle",
        "active_process_count": 0,
        "last_heartbeat_at": None,
        "desired_state": "idle",
        "pause_scope": "none",
        "next_action": "start",
        "resume_from": None,
        "run_root": None,
        "tensorboard_root": None,
        "attempt_id": None,
        "started_command_hash": None,
        "latest_final_checkpoint": None,
        "failure_reason": None,
        "abandoned": False,
        "current_run_invalid": False,
        "abandoned_reason": None,
        "abandoned_at": None,
        "abandonment_metadata": {},
        "last_codex_requested_model": "n/a",
        "last_codex_effective_model": "n/a",
        "last_codex_requested_reasoning_effort": "n/a",
        "last_codex_effective_reasoning_effort": "n/a",
        "updated_at": now_iso(),
    }


def refresh_status(profile: str) -> dict[str, Any]:
    payload = _base_status(profile)
    payload.update(_load_status(profile))
    pid = _read_pid(profile)
    payload["pid"] = pid
    running = _is_running(pid)
    payload["active_process_count"] = 1 if running else 0

    log_file = log_path(profile)
    tail = ""
    if log_file.exists():
        payload["last_heartbeat_at"] = datetime.fromtimestamp(log_file.stat().st_mtime).astimezone().isoformat(
            timespec="seconds"
        )
        tail = _current_attempt_tail(_tail_text(log_file))
        run_root = _extract_latest_value(tail, "run_root=")
        tensorboard_root = _extract_latest_value(tail, "tensorboard_root=")
        final_checkpoint = _extract_latest_value(tail, "final_checkpoint=")
        failure_reason = _extract_latest_value(tail, "failure_reason=")
        if run_root is not None:
            payload["run_root"] = run_root
        if tensorboard_root is not None:
            payload["tensorboard_root"] = tensorboard_root
        if final_checkpoint is not None:
            payload["latest_final_checkpoint"] = final_checkpoint
        if failure_reason is not None:
            payload["failure_reason"] = failure_reason

    if running:
        payload["supervisor_status"] = "running"
        payload["desired_state"] = "running"
        payload["next_action"] = "poll_status"
    else:
        if payload.get("abandoned") or payload.get("current_run_invalid"):
            payload["supervisor_status"] = "abandoned"
            payload["desired_state"] = "abandoned"
            payload["next_action"] = "start_new_run"
        elif payload.get("latest_final_checkpoint") or "final_checkpoint=" in tail:
            payload["supervisor_status"] = "completed"
            payload["desired_state"] = "completed"
            payload["next_action"] = "run_eval"
        elif payload.get("failure_reason"):
            payload["supervisor_status"] = "failed"
            payload["desired_state"] = "failed"
            payload["next_action"] = "inspect_log"
        elif pid is not None:
            payload["supervisor_status"] = "blocked_exited"
            payload["desired_state"] = "blocked_exited"
            payload["next_action"] = "inspect_log"

    payload["updated_at"] = now_iso()
    write_json(status_path(profile), payload)
    return payload


def start_training(profile: str, overrides: list[str]) -> dict[str, Any]:
    current = refresh_status(profile)
    if current.get("active_process_count", 0) > 0:
        raise RuntimeError(f"训练已在运行: pid={current.get('pid')}")

    root = supervisor_root(profile)
    root.mkdir(parents=True, exist_ok=True)

    command = [
        str(ISAAC_PYTHON),
        str(PROJECT_ROOT / "apps" / "isaac" / "train_navrl.py"),
        "--headless",
        f"profiles={profile}",
    ]
    command.extend(overrides)
    attempt_id = now_slug()
    command_hash = _command_hash(command)

    with log_path(profile).open("ab") as log_handle:
        start_marker = (
            f"\n[Supervisor] start_at={now_iso()} attempt_id={attempt_id} profile={profile} "
            f"command_hash={command_hash} "
            f"overrides={json.dumps(overrides, ensure_ascii=False)}\n"
        )
        log_handle.write(start_marker.encode("utf-8"))
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    pid_path(profile).write_text(str(process.pid), encoding="utf-8")
    cmd_path(profile).write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = _base_status(profile)
    payload.update(
        {
            "pid": process.pid,
            "supervisor_status": "running",
            "active_process_count": 1,
            "last_heartbeat_at": now_iso(),
            "desired_state": "running",
            "pause_scope": "none",
            "next_action": "poll_status",
            "resume_from": None,
            "started_at": now_iso(),
            "command": command,
            "attempt_id": attempt_id,
            "started_command_hash": command_hash,
            "latest_final_checkpoint": None,
            "failure_reason": None,
            "abandoned": False,
            "current_run_invalid": False,
            "abandoned_reason": None,
            "abandoned_at": None,
            "abandonment_metadata": {},
        }
    )
    write_json(status_path(profile), payload)
    return payload


def abandon_run(profile: str, reason: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    current = refresh_status(profile)
    current.update(
        {
            "abandoned": True,
            "current_run_invalid": True,
            "abandoned_reason": reason,
            "abandoned_at": now_iso(),
            "desired_state": "abandoned",
            "next_action": "start_new_run",
        }
    )
    if metadata:
        merged = dict(current.get("abandonment_metadata") or {})
        merged.update(metadata)
        current["abandonment_metadata"] = merged
    if not _is_running(current.get("pid")):
        current["supervisor_status"] = "abandoned"
        current["active_process_count"] = 0
    write_json(status_path(profile), current)
    _write_run_status_record(current.get("run_root"), current)
    return current


def stop_training(profile: str, force: bool) -> dict[str, Any]:
    current = refresh_status(profile)
    pid = current.get("pid")
    if not _is_running(pid):
        if current.get("abandoned") or current.get("current_run_invalid"):
            current["supervisor_status"] = "abandoned"
            current["desired_state"] = "abandoned"
            current["next_action"] = "start_new_run"
        else:
            current["supervisor_status"] = "paused_inactive"
            current["desired_state"] = "paused"
            current["pause_scope"] = "whole_run"
        write_json(status_path(profile), current)
        return current

    sig = signal.SIGKILL if force else signal.SIGTERM
    os.killpg(os.getpgid(pid), sig)
    current["desired_state"] = "paused"
    current["pause_scope"] = "whole_run"
    current["next_action"] = "resume_manually"
    current["updated_at"] = now_iso()
    write_json(status_path(profile), current)
    return refresh_status(profile)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo NavRL 后台训练 supervisor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="启动后台训练")
    start_parser.add_argument("--profile", default="smoke", choices=["smoke", "pilot", "main", "formal"])
    start_parser.add_argument("overrides", nargs="*", help="额外 Hydra 覆盖，例如 max_frame_num=1000")

    status_parser = subparsers.add_parser("status", help="查询后台训练状态")
    status_parser.add_argument("--profile", default="smoke", choices=["smoke", "pilot", "main", "formal"])

    stop_parser = subparsers.add_parser("stop", help="停止后台训练")
    stop_parser.add_argument("--profile", default="smoke", choices=["smoke", "pilot", "main", "formal"])
    stop_parser.add_argument("--force", action="store_true", help="使用 SIGKILL 强制停止")

    abandon_parser = subparsers.add_parser("abandon", help="标记当前 run 已废弃，不允许 resume")
    abandon_parser.add_argument("--profile", default="smoke", choices=["smoke", "pilot", "main", "formal"])
    abandon_parser.add_argument("--reason", required=True, help="废弃原因")
    abandon_parser.add_argument("--metadata-json", default=None, help="额外 metadata 的 JSON 字符串")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "start":
        payload = start_training(args.profile, args.overrides)
    elif args.command == "status":
        payload = refresh_status(args.profile)
    elif args.command == "stop":
        payload = stop_training(args.profile, args.force)
    elif args.command == "abandon":
        metadata = json.loads(args.metadata_json) if args.metadata_json else None
        payload = abandon_run(args.profile, args.reason, metadata)
    else:
        raise ValueError(f"未知命令: {args.command}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
