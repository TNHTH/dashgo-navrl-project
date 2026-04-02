#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dashgo_rl.project_paths import ARTIFACTS_ROOT
from navrl_dashgo.comparison import validate_comparison_payload, validate_eval_payload
from navrl_dashgo.runtime import write_json
from tools.background_train import log_path as train_log_path
from tools.background_train import refresh_status, start_training


def now() -> datetime:
    return datetime.now().astimezone()


def now_iso() -> str:
    return now().isoformat(timespec="seconds")


def now_slug() -> str:
    return now().strftime("%Y%m%d_%H%M%S")


def extract_latest_value(text: str, marker: str) -> str | None:
    value = None
    for line in text.splitlines():
        if marker in line:
            value = line.split(marker, 1)[1].strip()
    return value


def normalize_path_str(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    return str(Path(str(value)).expanduser().resolve())


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message}\n")


def fail_cycle(
    status: dict[str, Any],
    *,
    status_path: Path,
    log_file: Path,
    phase: str,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> int:
    status["phase"] = phase
    status["failure_reason"] = reason
    if extra:
        status.update(extra)
    write_json(status_path, status)
    append_log(log_file, f"failure_reason={reason}")
    return 1


def run_command(command: list[str], *, cwd: Path, log_file: Path) -> subprocess.CompletedProcess[str]:
    append_log(log_file, f"run_command={' '.join(command)}")
    completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    append_log(log_file, f"returncode={completed.returncode}")
    if completed.stdout.strip():
        append_log(log_file, f"stdout={completed.stdout.strip()[:4000]}")
    if completed.stderr.strip():
        append_log(log_file, f"stderr={completed.stderr.strip()[:4000]}")
    return completed


def load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def validate_eval_artifact(
    artifact_path: Path,
    *,
    expected_suite: str,
    expected_checkpoint: str | Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    payload = load_json_file(artifact_path)
    if payload is None:
        return None, [f"artifact_missing_or_invalid_json:{artifact_path}"]
    errors = validate_eval_payload(
        payload,
        expected_suite=expected_suite,
        expected_checkpoint=expected_checkpoint,
    )
    return payload, errors


def validate_compare_artifact(
    artifact_path: Path,
    *,
    expected_suite: str,
    expected_candidate_checkpoint: str | Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    payload = load_json_file(artifact_path)
    if payload is None:
        return None, [f"artifact_missing_or_invalid_json:{artifact_path}"]
    errors = validate_comparison_payload(
        payload,
        expected_suite=expected_suite,
        expected_candidate_checkpoint=expected_candidate_checkpoint,
    )
    return payload, errors


def run_command_until_valid_artifact(
    command: list[str],
    *,
    cwd: Path,
    artifact_path: Path,
    attempts: int,
    sleep_seconds: int,
    log_file: Path,
    validator,
) -> tuple[subprocess.CompletedProcess[str] | None, dict[str, Any] | None, list[str]]:
    last_completed: subprocess.CompletedProcess[str] | None = None
    last_payload: dict[str, Any] | None = None
    last_errors: list[str] = [f"artifact_missing:{artifact_path}"]
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            append_log(log_file, f"retrying command attempt={attempt} after sleep={sleep_seconds}s")
            time.sleep(sleep_seconds)
        completed = run_command(command, cwd=cwd, log_file=log_file)
        last_completed = completed
        payload, errors = validator(artifact_path)
        last_payload = payload
        last_errors = list(errors)
        if completed.returncode == 0 and not errors:
            return completed, payload, []
        append_log(
            log_file,
            f"artifact_validation_failed attempt={attempt} returncode={completed.returncode} errors={last_errors}",
        )
    return last_completed, last_payload, last_errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo NavRL 自治训练周期")
    parser.add_argument("--profile", default="pilot", choices=["smoke", "pilot", "main", "formal"])
    parser.add_argument("--budget-hours", type=float, default=17.0)
    parser.add_argument("--checkpoint-hours", type=float, default=6.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-frame-num", type=int, required=True)
    parser.add_argument("overrides", nargs="*", help="传给 background_train.py 的 Hydra overrides")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cycle_root = ARTIFACTS_ROOT / "autonomous" / now_slug()
    cycle_root.mkdir(parents=True, exist_ok=True)
    cycle_log = cycle_root / "cycle.log"
    cycle_status_path = cycle_root / "status.json"
    eval_root = cycle_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    overrides = list(args.overrides)
    if not any(item.startswith("max_frame_num=") for item in overrides):
        overrides.insert(0, f"max_frame_num={int(args.max_frame_num)}")

    started_at = now()
    six_hour_mark = started_at + timedelta(hours=float(args.checkpoint_hours))
    expected_end = started_at + timedelta(hours=float(args.budget_hours))

    status: dict[str, Any] = {
        "created_at": now_iso(),
        "profile": args.profile,
        "budget_hours": float(args.budget_hours),
        "checkpoint_hours": float(args.checkpoint_hours),
        "started_at": started_at.isoformat(timespec="seconds"),
        "six_hour_mark_at": six_hour_mark.isoformat(timespec="seconds"),
        "expected_end_at": expected_end.isoformat(timespec="seconds"),
        "phase": "starting_training",
        "overrides": overrides,
        "cycle_log": str(cycle_log),
        "eval_root": str(eval_root),
        "completed": False,
    }
    write_json(cycle_status_path, status)
    append_log(cycle_log, f"starting cycle_root={cycle_root}")

    try:
        training_status = start_training(args.profile, overrides)
    except Exception as exc:  # noqa: BLE001
        return fail_cycle(
            status,
            status_path=cycle_status_path,
            log_file=cycle_log,
            phase="failed",
            reason=f"start_training_failed:{type(exc).__name__}:{exc}",
        )
    bound_pid = training_status.get("pid")
    bound_attempt_id = training_status.get("attempt_id")
    bound_command_hash = training_status.get("started_command_hash")
    bound_run_root = normalize_path_str(training_status.get("run_root"))
    status["phase"] = "training"
    status["training_status"] = training_status
    status["bound_pid"] = bound_pid
    status["bound_attempt_id"] = bound_attempt_id
    status["bound_command_hash"] = bound_command_hash
    status["bound_run_root"] = bound_run_root
    write_json(cycle_status_path, status)
    append_log(
        cycle_log,
        "training_started "
        f"pid={bound_pid} attempt_id={bound_attempt_id} command_hash={bound_command_hash} overrides={overrides}",
    )

    passed_six_hour_mark = False
    final_checkpoint: str | None = None
    while True:
        time.sleep(int(args.poll_seconds))
        training_status = refresh_status(args.profile)
        status["training_status"] = training_status
        status["last_polled_at"] = now_iso()
        current_pid = training_status.get("pid")
        current_attempt_id = training_status.get("attempt_id")
        current_command_hash = training_status.get("started_command_hash")
        current_run_root = normalize_path_str(training_status.get("run_root"))
        if bound_pid is not None and current_pid != bound_pid:
            return fail_cycle(
                status,
                status_path=cycle_status_path,
                log_file=cycle_log,
                phase="failed",
                reason=f"training_pid_drift:expected={bound_pid}:actual={current_pid}",
            )
        if bound_attempt_id is not None and current_attempt_id != bound_attempt_id:
            return fail_cycle(
                status,
                status_path=cycle_status_path,
                log_file=cycle_log,
                phase="failed",
                reason=f"training_attempt_drift:expected={bound_attempt_id}:actual={current_attempt_id}",
            )
        if bound_command_hash is not None and current_command_hash != bound_command_hash:
            return fail_cycle(
                status,
                status_path=cycle_status_path,
                log_file=cycle_log,
                phase="failed",
                reason=f"training_command_hash_drift:expected={bound_command_hash}:actual={current_command_hash}",
            )
        if current_run_root:
            if bound_run_root is None:
                bound_run_root = current_run_root
                status["bound_run_root"] = bound_run_root
                append_log(cycle_log, f"bound_run_root={bound_run_root}")
            elif current_run_root != bound_run_root:
                return fail_cycle(
                    status,
                    status_path=cycle_status_path,
                    log_file=cycle_log,
                    phase="failed",
                    reason=f"training_run_root_drift:expected={bound_run_root}:actual={current_run_root}",
                )
        if (not passed_six_hour_mark) and now() >= six_hour_mark:
            passed_six_hour_mark = True
            append_log(cycle_log, "reached_checkpoint_hours_mark")
            status["checkpoint_hours_reached_at"] = now_iso()
        write_json(cycle_status_path, status)

        if training_status.get("abandoned") or training_status.get("current_run_invalid"):
            return fail_cycle(
                status,
                status_path=cycle_status_path,
                log_file=cycle_log,
                phase="failed",
                reason="training_abandoned_or_invalidated",
            )

        supervisor_status = str(training_status.get("supervisor_status"))
        if supervisor_status == "completed":
            final_checkpoint = normalize_path_str(
                training_status.get("latest_final_checkpoint")
                or extract_latest_value(
                    train_log_path(args.profile).read_text(encoding="utf-8", errors="ignore"),
                    "final_checkpoint=",
                )
            )
            if bound_run_root is not None and final_checkpoint is not None and not final_checkpoint.startswith(bound_run_root):
                return fail_cycle(
                    status,
                    status_path=cycle_status_path,
                    log_file=cycle_log,
                    phase="failed",
                    reason=f"final_checkpoint_outside_run_root:{final_checkpoint}",
                )
            status["phase"] = "training_completed"
            status["final_checkpoint"] = final_checkpoint
            write_json(cycle_status_path, status)
            append_log(cycle_log, f"training_completed final_checkpoint={final_checkpoint}")
            break

        if supervisor_status in {"failed", "blocked_exited", "abandoned"}:
            failure_reason = training_status.get("failure_reason") or f"training_{supervisor_status}"
            return fail_cycle(
                status,
                status_path=cycle_status_path,
                log_file=cycle_log,
                phase="failed",
                reason=str(failure_reason),
            )

    if not final_checkpoint:
        return fail_cycle(
            status,
            status_path=cycle_status_path,
            log_file=cycle_log,
            phase="failed",
            reason="missing_final_checkpoint",
        )

    status["phase"] = "evaluating"
    status["bound_final_checkpoint"] = final_checkpoint
    write_json(cycle_status_path, status)
    append_log(cycle_log, "cooldown before eval launch")
    time.sleep(20)

    quick_json = eval_root / "quick.json"
    main_json = eval_root / "main.json"
    compare_quick_json = eval_root / "compare_quick.json"
    compare_main_json = eval_root / "compare_main.json"
    compare_quick_md = eval_root / "compare_quick.md"
    compare_main_md = eval_root / "compare_main.md"

    quick_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "eval_checkpoint.py"),
        "--checkpoint",
        str(Path(final_checkpoint)),
        "--suite",
        "quick",
        "--json-out",
        str(quick_json),
    ]
    main_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "eval_checkpoint.py"),
        "--checkpoint",
        str(Path(final_checkpoint)),
        "--suite",
        "main",
        "--json-out",
        str(main_json),
    ]
    compare_quick_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "compare_models.py"),
        "--candidate-json",
        str(quick_json),
        "--suite",
        "quick",
        "--json-out",
        str(compare_quick_json),
        "--report-out",
        str(compare_quick_md),
    ]
    compare_main_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "compare_models.py"),
        "--candidate-json",
        str(main_json),
        "--suite",
        "main",
        "--json-out",
        str(compare_main_json),
        "--report-out",
        str(compare_main_md),
    ]

    _, quick_payload, quick_errors = run_command_until_valid_artifact(
        quick_cmd,
        cwd=PROJECT_ROOT,
        artifact_path=quick_json,
        attempts=3,
        sleep_seconds=20,
        log_file=cycle_log,
        validator=lambda path: validate_eval_artifact(
            path,
            expected_suite="quick",
            expected_checkpoint=final_checkpoint,
        ),
    )
    if quick_errors:
        return fail_cycle(
            status,
            status_path=cycle_status_path,
            log_file=cycle_log,
            phase="failed",
            reason=f"eval_quick_invalid:{quick_errors}",
        )
    status["quick_eval"] = quick_payload

    _, main_payload, main_errors = run_command_until_valid_artifact(
        main_cmd,
        cwd=PROJECT_ROOT,
        artifact_path=main_json,
        attempts=3,
        sleep_seconds=20,
        log_file=cycle_log,
        validator=lambda path: validate_eval_artifact(
            path,
            expected_suite="main",
            expected_checkpoint=final_checkpoint,
        ),
    )
    if main_errors:
        return fail_cycle(
            status,
            status_path=cycle_status_path,
            log_file=cycle_log,
            phase="failed",
            reason=f"eval_main_invalid:{main_errors}",
        )
    status["main_eval"] = main_payload

    _, compare_quick_payload, compare_quick_errors = run_command_until_valid_artifact(
        compare_quick_cmd,
        cwd=PROJECT_ROOT,
        artifact_path=compare_quick_json,
        attempts=3,
        sleep_seconds=20,
        log_file=cycle_log,
        validator=lambda path: validate_compare_artifact(
            path,
            expected_suite="quick",
            expected_candidate_checkpoint=final_checkpoint,
        ),
    )
    if compare_quick_errors:
        return fail_cycle(
            status,
            status_path=cycle_status_path,
            log_file=cycle_log,
            phase="failed",
            reason=f"compare_quick_invalid:{compare_quick_errors}",
        )
    if not compare_quick_md.exists() or not compare_quick_md.read_text(encoding="utf-8").strip():
        return fail_cycle(
            status,
            status_path=cycle_status_path,
            log_file=cycle_log,
            phase="failed",
            reason=f"missing_compare_quick_report:{compare_quick_md}",
        )
    status["compare_quick"] = compare_quick_payload

    _, compare_main_payload, compare_main_errors = run_command_until_valid_artifact(
        compare_main_cmd,
        cwd=PROJECT_ROOT,
        artifact_path=compare_main_json,
        attempts=3,
        sleep_seconds=20,
        log_file=cycle_log,
        validator=lambda path: validate_compare_artifact(
            path,
            expected_suite="main",
            expected_candidate_checkpoint=final_checkpoint,
        ),
    )
    if compare_main_errors:
        return fail_cycle(
            status,
            status_path=cycle_status_path,
            log_file=cycle_log,
            phase="failed",
            reason=f"compare_main_invalid:{compare_main_errors}",
        )
    if not compare_main_md.exists() or not compare_main_md.read_text(encoding="utf-8").strip():
        return fail_cycle(
            status,
            status_path=cycle_status_path,
            log_file=cycle_log,
            phase="failed",
            reason=f"missing_compare_main_report:{compare_main_md}",
        )
    status["compare_main"] = compare_main_payload

    status.update(
        {
            "phase": "completed",
            "completed": True,
            "completed_at": now_iso(),
            "eval_artifacts": {
                "quick_json": str(quick_json),
                "main_json": str(main_json),
                "compare_quick_json": str(compare_quick_json),
                "compare_main_json": str(compare_main_json),
                "compare_quick_md": str(compare_quick_md),
                "compare_main_md": str(compare_main_md),
            },
        }
    )
    write_json(cycle_status_path, status)
    append_log(cycle_log, "cycle completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
