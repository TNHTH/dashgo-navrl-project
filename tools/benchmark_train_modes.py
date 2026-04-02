#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dashgo_rl.project_paths import ARTIFACTS_ROOT, ISAAC_PYTHON


OFFICIAL_DECISION_CANDIDATES = [
    "official_e8_camoff",
    "official_e24_camoff",
    "official_e64_camoff",
    "official_e96_camoff",
    "official_e128_camoff",
]
NON_FINITE_PATTERN = re.compile(r"\b(?:nan|inf)\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    profile: str
    map_source: str
    num_envs: int
    enable_cameras: bool


DEFAULT_CANDIDATES = [
    CandidateConfig(name="official_e8_camoff", profile="pilot", map_source="dashgo_official", num_envs=8, enable_cameras=False),
    CandidateConfig(name="official_e24_camoff", profile="pilot", map_source="dashgo_official", num_envs=24, enable_cameras=False),
    CandidateConfig(name="official_e64_camoff", profile="pilot", map_source="dashgo_official", num_envs=64, enable_cameras=False),
    CandidateConfig(name="official_e96_camoff", profile="pilot", map_source="dashgo_official", num_envs=96, enable_cameras=False),
    CandidateConfig(name="official_e128_camoff", profile="pilot", map_source="dashgo_official", num_envs=128, enable_cameras=False),
    CandidateConfig(name="official_e8_camon", profile="pilot", map_source="dashgo_official", num_envs=8, enable_cameras=True),
    CandidateConfig(name="upstream_e16_camoff", profile="pilot", map_source="navrl_upstream", num_envs=16, enable_cameras=False),
]


def now_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def gpu_snapshot() -> dict[str, float] | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    line = result.stdout.strip().splitlines()
    if result.returncode != 0 or not line:
        return None
    try:
        total, used, util = [float(item.strip()) for item in line[0].split(",")]
    except ValueError:
        return None
    return {
        "memory_total_mb": total,
        "memory_used_mb": used,
        "utilization_gpu": util,
    }


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


def extract_latest_value(text: str, marker: str) -> str | None:
    value = None
    for line in text.splitlines():
        if marker in line:
            value = line.split(marker, 1)[1].strip()
    return value


def has_non_finite_marker(log_text: str) -> bool:
    for line in log_text.splitlines():
        if "actor_loss=" in line or "critic_loss=" in line or "entropy=" in line or "explained_var=" in line:
            if NON_FINITE_PATTERN.search(line):
                return True
        if "failure_reason=non_finite_training_state" in line:
            return True
    return False


def build_command(candidate: CandidateConfig, max_frame_num: int) -> list[str]:
    return [
        str(ISAAC_PYTHON),
        str(PROJECT_ROOT / "apps" / "isaac" / "train_navrl.py"),
        "--headless",
        f"profiles={candidate.profile}",
        f"max_frame_num={max_frame_num}",
        "save_interval_batches=9999",
        "eval_interval_batches=9999",
        "logging.print_interval_batches=9999",
        f"env.num_envs={candidate.num_envs}",
        f"env.map_source={candidate.map_source}",
        f"enable_cameras={'true' if candidate.enable_cameras else 'false'}",
    ]


def run_candidate(
    candidate: CandidateConfig,
    *,
    max_frame_num: int,
    timeout_seconds: int,
    sample_interval_seconds: float,
    repeat_index: int,
    run_root: Path,
) -> dict[str, Any]:
    candidate_root = run_root / candidate.name / f"repeat_{repeat_index:02d}"
    candidate_root.mkdir(parents=True, exist_ok=True)
    log_path = candidate_root / "train.log"
    command = build_command(candidate, max_frame_num)
    started_at = now_iso()
    baseline_gpu = gpu_snapshot()
    peak_memory_mb = (baseline_gpu or {}).get("memory_used_mb", 0.0)
    peak_utilization_gpu = (baseline_gpu or {}).get("utilization_gpu", 0.0)

    with log_path.open("wb") as log_handle:
        banner = f"[Benchmark] started_at={started_at} candidate={candidate.name} repeat_index={repeat_index}\n"
        log_handle.write(banner.encode("utf-8"))
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    started_monotonic = time.monotonic()
    timed_out = False
    while process.poll() is None:
        snapshot = gpu_snapshot()
        if snapshot is not None:
            peak_memory_mb = max(peak_memory_mb, snapshot["memory_used_mb"])
            peak_utilization_gpu = max(peak_utilization_gpu, snapshot["utilization_gpu"])
        elapsed = time.monotonic() - started_monotonic
        if elapsed > timeout_seconds:
            timed_out = True
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            break
        time.sleep(sample_interval_seconds)

    return_code = process.wait(timeout=30)
    elapsed_seconds = time.monotonic() - started_monotonic
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    final_checkpoint = extract_latest_value(log_text, "final_checkpoint=")
    run_path = extract_latest_value(log_text, "run_root=")
    has_nan = has_non_finite_marker(log_text)
    finite = not has_nan
    success = (return_code == 0) and (final_checkpoint is not None) and (not timed_out) and finite
    memory_total_mb = (baseline_gpu or gpu_snapshot() or {}).get("memory_total_mb", 0.0)
    frames_per_second = float(max_frame_num) / max(elapsed_seconds, 1.0e-6) if success else 0.0

    payload = {
        **asdict(candidate),
        "started_at": started_at,
        "finished_at": now_iso(),
        "repeat_index": int(repeat_index),
        "max_frame_num": int(max_frame_num),
        "command": command,
        "command_shell": " ".join(shlex.quote(item) for item in command),
        "return_code": return_code,
        "timed_out": timed_out,
        "success": success,
        "finite": finite,
        "has_nan": has_nan,
        "elapsed_seconds": elapsed_seconds,
        "frames_per_second": frames_per_second,
        "peak_memory_mb": peak_memory_mb,
        "memory_total_mb": memory_total_mb,
        "peak_memory_ratio": (peak_memory_mb / memory_total_mb) if memory_total_mb else None,
        "peak_utilization_gpu": peak_utilization_gpu,
        "run_root": run_path,
        "final_checkpoint": final_checkpoint,
        "log_path": str(log_path),
        "probe_metrics": {
            "elapsed_seconds": elapsed_seconds,
            "frames_per_second": frames_per_second,
            "peak_memory_ratio": (peak_memory_mb / memory_total_mb) if memory_total_mb else None,
            "peak_utilization_gpu": peak_utilization_gpu,
            "timed_out": timed_out,
            "has_nan": has_nan,
            "return_code": return_code,
        },
    }
    (candidate_root / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def median_or_none(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(statistics.median(numeric))


def aggregate_candidate_results(results: list[dict[str, Any]], required_repeats: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(str(item["name"]), []).append(item)

    aggregates: list[dict[str, Any]] = []
    for name, runs in grouped.items():
        first = runs[0]
        complete_repeats = len(runs) == required_repeats
        stability_ok = all(
            bool(run.get("success")) and bool(run.get("finite")) and (not bool(run.get("timed_out"))) and (not bool(run.get("has_nan")))
            for run in runs
        )
        selection_reason = "stable_probe_ready" if complete_repeats and stability_ok else "insufficient_repeats_or_instability"
        aggregates.append(
            {
                "name": name,
                "profile": first["profile"],
                "map_source": first["map_source"],
                "num_envs": first["num_envs"],
                "enable_cameras": first["enable_cameras"],
                "repeat_count": len(runs),
                "repeat_indices": [run["repeat_index"] for run in runs],
                "successful_runs": sum(1 for run in runs if run.get("success")),
                "complete_repeats": complete_repeats,
                "stable": stability_ok,
                "decision_eligible": complete_repeats and stability_ok,
                "median_frames_per_second": median_or_none([run.get("frames_per_second") for run in runs]),
                "median_peak_memory_ratio": median_or_none([run.get("peak_memory_ratio") for run in runs]),
                "median_peak_utilization_gpu": median_or_none([run.get("peak_utilization_gpu") for run in runs]),
                "has_nan": any(bool(run.get("has_nan")) for run in runs),
                "timed_out": any(bool(run.get("timed_out")) for run in runs),
                "selection_reason": selection_reason,
                "probe_metrics": {
                    "median_frames_per_second": median_or_none([run.get("frames_per_second") for run in runs]),
                    "median_peak_memory_ratio": median_or_none([run.get("peak_memory_ratio") for run in runs]),
                    "median_peak_utilization_gpu": median_or_none([run.get("peak_utilization_gpu") for run in runs]),
                    "has_nan": any(bool(run.get("has_nan")) for run in runs),
                    "timed_out": any(bool(run.get("timed_out")) for run in runs),
                    "return_codes": [run.get("return_code") for run in runs],
                },
            }
        )
    aggregates.sort(key=lambda item: (item["num_envs"], item["name"]))
    return aggregates


def choose_best_official(aggregates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    official = [
        item
        for item in aggregates
        if item.get("decision_eligible")
        and item.get("map_source") == "dashgo_official"
        and item.get("enable_cameras") is False
        and item.get("name") in OFFICIAL_DECISION_CANDIDATES
    ]
    if not official:
        return None, "没有满足稳定性前置条件的正式口径候选"
    official.sort(
        key=lambda item: (
            float(item.get("median_frames_per_second") or 0.0),
            -float(item.get("median_peak_memory_ratio") or 1.0),
            int(item.get("num_envs", 0)),
        ),
        reverse=True,
    )
    best = official[0]
    reason = (
        f"selected_by_median_fps_with_stability:name={best['name']}:"
        f"fps={best.get('median_frames_per_second')}:memory_ratio={best.get('median_peak_memory_ratio')}"
    )
    return best, reason


def decision_ready_reason(aggregates: list[dict[str, Any]], repeats: int) -> tuple[bool, str]:
    aggregate_map = {item["name"]: item for item in aggregates}
    missing = [name for name in OFFICIAL_DECISION_CANDIDATES if name not in aggregate_map]
    if missing:
        return False, f"missing_required_candidates:{missing}"
    if repeats < 2:
        return False, f"repeats_too_low:{repeats}"
    unstable = [
        name
        for name in OFFICIAL_DECISION_CANDIDATES
        if not bool(aggregate_map[name].get("decision_eligible"))
    ]
    if unstable:
        return False, f"official_candidates_not_stable:{unstable}"
    return True, "complete_official_matrix_with_repeated_stable_runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo NavRL 本机训练模式短跑摸索")
    parser.add_argument("--max-frame-num", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-concurrent", action="store_true", help="允许与当前训练并发运行 probe")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    benchmark_root = ARTIFACTS_ROOT / "benchmarks" / now_slug()
    benchmark_root.mkdir(parents=True, exist_ok=True)
    json_out = args.json_out.resolve() if args.json_out is not None else benchmark_root / "summary.json"

    active_training_pids = detect_active_training()
    if active_training_pids and not args.allow_concurrent:
        payload = {
            "created_at": now_iso(),
            "benchmark_root": str(benchmark_root),
            "status": "failed",
            "results": [],
            "aggregated_results": [],
            "decision_ready": False,
            "selection_reason": f"blocked_by_active_training:{active_training_pids}",
            "selected_candidate": None,
            "active_training_pids": active_training_pids,
            "repeats": int(args.repeats),
        }
        write_summary(json_out, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    results: list[dict[str, Any]] = []
    for repeat_index in range(1, int(args.repeats) + 1):
        for candidate in DEFAULT_CANDIDATES:
            result = run_candidate(
                candidate,
                max_frame_num=int(args.max_frame_num),
                timeout_seconds=int(args.timeout_seconds),
                sample_interval_seconds=float(args.sample_interval_seconds),
                repeat_index=repeat_index,
                run_root=benchmark_root,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "candidate": candidate.name,
                        "repeat_index": repeat_index,
                        "success": result["success"],
                        "finite": result["finite"],
                        "has_nan": result["has_nan"],
                        "frames_per_second": result["frames_per_second"],
                        "peak_memory_ratio": result["peak_memory_ratio"],
                        "return_code": result["return_code"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    aggregates = aggregate_candidate_results(results, required_repeats=int(args.repeats))
    ready, ready_reason = decision_ready_reason(aggregates, repeats=int(args.repeats))
    selected_candidate = None
    selection_reason = ready_reason
    if ready:
        selected_candidate, selection_reason = choose_best_official(aggregates)
        ready = selected_candidate is not None
        if not ready:
            selection_reason = selection_reason

    payload = {
        "created_at": now_iso(),
        "benchmark_root": str(benchmark_root),
        "status": "completed" if results else "failed",
        "results": results,
        "aggregated_results": aggregates,
        "decision_ready": ready,
        "selection_reason": selection_reason,
        "selected_candidate": selected_candidate,
        "best_official": selected_candidate,
        "repeats": int(args.repeats),
        "active_training_pids": active_training_pids,
    }
    write_summary(json_out, payload)

    print(
        json.dumps(
            {
                "selected_candidate": None if selected_candidate is None else selected_candidate["name"],
                "decision_ready": ready,
                "selection_reason": selection_reason,
                "summary_json": str(json_out),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
