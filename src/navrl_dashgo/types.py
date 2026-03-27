from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


@dataclass
class JsonModel:
    def to_dict(self) -> dict[str, Any]:
        return _normalize(asdict(self))


@dataclass
class EvalMetrics(JsonModel):
    success_rate: float = 0.0
    collision_rate: float = 0.0
    hard_stop_rate: float = 0.0
    cmd_saturation_rate: float = 0.0
    heading_guard_trigger_rate: float = 0.0
    recovery_trigger_rate: float = 0.0
    plan_invalid_ratio: float = 0.0
    time_to_goal: float = 0.0
    timeout_rate: float = 0.0
    mean_steps: float = 0.0
    reverse_case_success_rate: float = 0.0
    spin_proxy_rate: float = 0.0
    progress_stall_rate: float = 0.0
    high_clip_ratio: float = 0.0
    path_efficiency: float = 0.0
    net_progress_ratio: float = 0.0
    orbit_score: float = 0.0
    near_obstacle_dwell: float = 0.0
    sensor_health_score: float = 0.0
    log_anomaly_count: float = 0.0
    score: float = 0.0
    total_episodes: int = 0
    completed_episodes: int = 0


@dataclass
class EvalRequest(JsonModel):
    checkpoint: Path
    suite: str
    project_root: Path
    notes: list[str] = field(default_factory=list)
    requested_episodes: int | None = None
    created_at: str = field(default_factory=iso_now)


@dataclass
class EvalResult(JsonModel):
    status: str
    request: EvalRequest
    metrics: EvalMetrics | None = None
    scenes: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=iso_now)
