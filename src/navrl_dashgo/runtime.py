from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dashgo_rl.project_paths import ARTIFACTS_ROOT, CHECKPOINTS_ROOT, EVAL_ROOT, LOGS_ROOT, PROJECT_ROOT


@dataclass
class RunLayout:
    project_root: Path
    run_root: Path
    checkpoint_root: Path
    log_root: Path
    tensorboard_root: Path
    config_snapshot: Path


def timestamp_slug(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("%Y%m%d_%H%M%S")


def build_run_layout(profile: str, now: datetime | None = None) -> RunLayout:
    slug = f"{profile}_{timestamp_slug(now)}"
    run_root = ARTIFACTS_ROOT / "runs" / slug
    checkpoint_root = run_root / "checkpoints"
    log_root = run_root / "logs"
    tensorboard_root = log_root / "tensorboard"
    for path in (
        ARTIFACTS_ROOT,
        CHECKPOINTS_ROOT,
        LOGS_ROOT,
        EVAL_ROOT,
        run_root,
        checkpoint_root,
        log_root,
        tensorboard_root,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return RunLayout(
        project_root=PROJECT_ROOT,
        run_root=run_root,
        checkpoint_root=checkpoint_root,
        log_root=log_root,
        tensorboard_root=tensorboard_root,
        config_snapshot=run_root / "config_snapshot.json",
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
