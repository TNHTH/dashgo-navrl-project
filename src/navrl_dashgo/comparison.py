from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any


DEFAULT_METRIC_NAMES = [
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "time_to_goal",
    "path_efficiency",
    "spin_proxy_rate",
    "progress_stall_rate",
    "score",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_path_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(Path(str(value)).expanduser().resolve())


def resolve_online_baseline_checkpoint(manifest_path: Path) -> Path:
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到在线基线 manifest: {manifest_path}")
    payload = read_json(manifest_path)
    checkpoint = Path(str(payload.get("checkpoint_path", ""))).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"manifest 指向的基线 checkpoint 不存在: {checkpoint}")
    return checkpoint


def extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload.get("metrics") or {}
    return metrics if isinstance(metrics, dict) else {}


def validate_eval_payload(
    payload: dict[str, Any],
    *,
    expected_suite: str | None = None,
    expected_checkpoint: str | Path | None = None,
    required_metric_names: list[str] | None = None,
    allowed_statuses: tuple[str, ...] | None = ("completed",),
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload 不是 JSON object"]
    allowed = tuple(allowed_statuses or ("completed",))
    status = str(payload.get("status"))
    if status not in allowed:
        if allowed == ("completed",):
            errors.append(f"status={payload.get('status')!r} 不是 completed")
        else:
            errors.append(f"status={payload.get('status')!r} 不在允许列表 {list(allowed)}")
    request = payload.get("request")
    if not isinstance(request, dict):
        errors.append("缺少 request")
        request = {}
    if expected_suite is not None:
        actual_suite = request.get("suite") or payload.get("metadata", {}).get("suite")
        if actual_suite != expected_suite:
            errors.append(f"suite 不匹配: expected={expected_suite}, actual={actual_suite}")
    if expected_checkpoint is not None:
        actual_checkpoint = _normalize_path_str(request.get("checkpoint"))
        target_checkpoint = _normalize_path_str(expected_checkpoint)
        if actual_checkpoint != target_checkpoint:
            errors.append(f"checkpoint 不匹配: expected={target_checkpoint}, actual={actual_checkpoint}")
    metrics = extract_metrics(payload)
    if not metrics:
        errors.append("缺少 metrics")
    required_names = required_metric_names or DEFAULT_METRIC_NAMES
    missing_metrics = [name for name in required_names if name not in metrics]
    if missing_metrics:
        errors.append(f"metrics 缺少关键字段: {missing_metrics}")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        errors.append("scenes 不是 list")
    return errors


def metric_row(name: str, baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, Any]:
    base_val = baseline.get(name)
    cand_val = candidate.get(name)
    delta = None if base_val is None or cand_val is None else float(cand_val) - float(base_val)
    return {"metric": name, "baseline": base_val, "candidate": cand_val, "delta": delta}


def build_metric_rows(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    metric_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    metrics = metric_names or DEFAULT_METRIC_NAMES
    baseline_metrics = extract_metrics(baseline_payload)
    candidate_metrics = extract_metrics(candidate_payload)
    return [metric_row(name, baseline_metrics, candidate_metrics) for name in metrics]


def validate_comparison_payload(
    payload: dict[str, Any],
    *,
    expected_suite: str | None = None,
    expected_candidate_checkpoint: str | Path | None = None,
    allowed_eval_statuses: tuple[str, ...] | None = ("completed",),
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload 不是 JSON object"]
    if str(payload.get("status", "completed")) != "completed":
        errors.append(f"comparison status={payload.get('status')!r} 不是 completed")
    if expected_suite is not None and payload.get("suite") != expected_suite:
        errors.append(f"comparison suite 不匹配: expected={expected_suite}, actual={payload.get('suite')}")
    baseline_payload = payload.get("baseline")
    candidate_payload = payload.get("candidate")
    if not isinstance(baseline_payload, dict):
        errors.append("baseline payload 缺失")
    else:
        errors.extend(
            f"baseline: {item}"
            for item in validate_eval_payload(
                baseline_payload,
                expected_suite=expected_suite,
                allowed_statuses=allowed_eval_statuses,
            )
        )
    if not isinstance(candidate_payload, dict):
        errors.append("candidate payload 缺失")
    else:
        errors.extend(
            f"candidate: {item}"
            for item in validate_eval_payload(
                candidate_payload,
                expected_suite=expected_suite,
                expected_checkpoint=expected_candidate_checkpoint,
                allowed_statuses=allowed_eval_statuses,
            )
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("rows 缺失或为空")
    return errors


def termination_counts(payload: dict[str, Any]) -> dict[str, int]:
    scenes = payload.get("scenes") or []
    counter = Counter(str(scene.get("termination_reason", "unknown")) for scene in scenes if isinstance(scene, dict))
    return dict(sorted(counter.items()))


def mean_scene_value(payload: dict[str, Any], key: str) -> float | None:
    scenes = [scene for scene in (payload.get("scenes") or []) if isinstance(scene, dict) and key in scene]
    if not scenes:
        return None
    return sum(float(scene[key]) for scene in scenes) / len(scenes)


def infer_failure_modes(payload: dict[str, Any]) -> list[str]:
    metrics = extract_metrics(payload)
    failures: list[str] = []
    if metrics.get("success_rate", 0.0) <= 0.0:
        failures.append("没有成功到达 episode")
    if metrics.get("collision_rate", 0.0) >= 0.5:
        failures.append("碰撞占主导")
    if metrics.get("timeout_rate", 0.0) >= 0.3:
        failures.append("超时占比较高")
    if metrics.get("progress_stall_rate", 0.0) >= 0.5:
        failures.append("推进停滞明显")
    if metrics.get("orbit_score", 0.0) >= 0.15:
        failures.append("存在绕圈或局部极小值迹象")
    mean_end_distance = mean_scene_value(payload, "end_distance")
    if mean_end_distance is not None and mean_end_distance > 1.0:
        failures.append(f"平均终点剩余距离偏大 ({mean_end_distance:.3f}m)")
    if not failures and str(payload.get("status")) != "completed":
        failures.append("状态未达 completed，但未提取到明确主导失败模式")
    return failures


def overall_conclusion(rows: list[dict[str, Any]]) -> str:
    row_map = {row["metric"]: row for row in rows}
    success_delta = row_map.get("success_rate", {}).get("delta")
    collision_delta = row_map.get("collision_rate", {}).get("delta")
    score_delta = row_map.get("score", {}).get("delta")
    if success_delta is None or collision_delta is None or score_delta is None:
        return "缺少关键指标，无法形成自动结论。"
    if success_delta > 0.0 and collision_delta <= 0.0 and score_delta > 0.0:
        return "NavRL-style 候选整体优于基线。"
    if success_delta < 0.0 and collision_delta >= 0.0 and score_delta < 0.0:
        return "NavRL-style 候选整体劣于基线。"
    if score_delta > 0.0:
        return "NavRL-style 候选在部分指标上改善，但未形成全面优势。"
    return "NavRL-style 候选未形成稳定优势。"


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def render_markdown_report(
    *,
    suite: str,
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    baseline_source: str,
    candidate_source: str,
    generated_on: str | None = None,
) -> str:
    baseline_metrics = extract_metrics(baseline_payload)
    candidate_metrics = extract_metrics(candidate_payload)
    baseline_failures = infer_failure_modes(baseline_payload)
    candidate_failures = infer_failure_modes(candidate_payload)
    report_date = generated_on or datetime.now().astimezone().strftime("%Y-%m-%d")

    lines = [
        f"# DashGo NavRL 对比报告 - {suite}",
        "",
        f"生成时间: {report_date}",
        "",
        "## 结论",
        f"- {overall_conclusion(rows)}",
        f"- 基线状态: `{baseline_payload.get('status', 'unknown')}`，候选状态: `{candidate_payload.get('status', 'unknown')}`。",
        f"- 基线来源: `{baseline_source}`",
        f"- 候选来源: `{candidate_source}`",
        "",
        "## 指标对比",
        "| metric | baseline | candidate | delta |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {format_metric(row['baseline'])} | {format_metric(row['candidate'])} | {format_metric(row['delta'])} |"
        )

    lines.extend(
        [
            "",
            "## 终止原因分布",
            f"- 基线: `{termination_counts(baseline_payload)}`",
            f"- 候选: `{termination_counts(candidate_payload)}`",
            "",
            "## 失败模式",
            f"- 基线: `{'；'.join(baseline_failures) if baseline_failures else '无明显失败模式'}`",
            f"- 候选: `{'；'.join(candidate_failures) if candidate_failures else '无明显失败模式'}`",
            "",
            "## 辅助观察",
            f"- 基线平均终点距离: `{format_metric(mean_scene_value(baseline_payload, 'end_distance'))}` m",
            f"- 候选平均终点距离: `{format_metric(mean_scene_value(candidate_payload, 'end_distance'))}` m",
            f"- 基线平均路径效率: `{format_metric(baseline_metrics.get('path_efficiency'))}`",
            f"- 候选平均路径效率: `{format_metric(candidate_metrics.get('path_efficiency'))}`",
            "",
            "## 口径说明",
            "- 这是第一阶段仿真闭环对比，只比较 DashGo 平台下的 quick/main 协议，不包含 ROS2/TorchScript 实机部署。",
            "- 上游 `onboard_detector` 与 `safe_action` 只作为第二阶段参考，不纳入本次可部署链对比。",
        ]
    )
    return "\n".join(lines) + "\n"
