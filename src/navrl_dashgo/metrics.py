from __future__ import annotations

from .types import EvalMetrics


def summarize_eval_episodes(episodes: list[dict], *, suite: str, log_anomaly_count: int = 0) -> EvalMetrics:
    total = len(episodes)
    if total == 0:
        return EvalMetrics(log_anomaly_count=float(log_anomaly_count))

    successes = sum(1 for item in episodes if item.get("termination_reason") == "reach_goal")
    collisions = sum(1 for item in episodes if item.get("termination_reason") == "object_collision")
    timeouts = sum(1 for item in episodes if item.get("termination_reason") == "time_out")
    reverse_total = max(1, sum(1 for item in episodes if item.get("reverse_case")))
    reverse_successes = sum(
        1 for item in episodes if item.get("reverse_case") and item.get("termination_reason") == "reach_goal"
    )

    mean_steps = sum(float(item.get("steps", 0.0)) for item in episodes) / total
    spin_proxy_rate = sum(float(item.get("spin_proxy_ratio", 0.0)) for item in episodes) / total
    progress_stall_rate = sum(1.0 for item in episodes if item.get("progress_stall")) / total
    high_clip_ratio = sum(float(item.get("high_clip_ratio", 0.0)) for item in episodes) / total
    path_efficiency = sum(float(item.get("path_efficiency", 0.0)) for item in episodes) / total
    net_progress_ratio = sum(float(item.get("net_progress_ratio", 0.0)) for item in episodes) / total
    orbit_score = sum(1.0 for item in episodes if item.get("orbit_detected")) / total
    near_obstacle_dwell = sum(float(item.get("near_obstacle_dwell_ratio", 0.0)) for item in episodes) / total
    sensor_health_score = sum(float(item.get("sensor_health_score", 1.0)) for item in episodes) / total
    heading_guard_trigger_rate = sum(float(item.get("heading_guard_trigger_rate", 0.0)) for item in episodes) / total
    recovery_trigger_rate = sum(float(item.get("recovery_trigger_rate", 0.0)) for item in episodes) / total
    plan_invalid_ratio = sum(float(item.get("plan_invalid_ratio", 0.0)) for item in episodes) / total
    successful_times = [
        float(item.get("elapsed_time", 0.0)) for item in episodes if item.get("termination_reason") == "reach_goal"
    ]
    time_to_goal = sum(successful_times) / len(successful_times) if successful_times else 0.0
    hard_stop_rate = collisions / total
    cmd_saturation_rate = high_clip_ratio

    score = (
        successes / total * 100.0
        - collisions / total * 40.0
        - timeouts / total * 25.0
        - orbit_score * 25.0
        - progress_stall_rate * 20.0
        + path_efficiency * 15.0
        + net_progress_ratio * 10.0
        - high_clip_ratio * 10.0
    )
    if suite == "main":
        score -= near_obstacle_dwell * 5.0

    return EvalMetrics(
        success_rate=successes / total,
        collision_rate=collisions / total,
        hard_stop_rate=hard_stop_rate,
        cmd_saturation_rate=cmd_saturation_rate,
        heading_guard_trigger_rate=heading_guard_trigger_rate,
        recovery_trigger_rate=recovery_trigger_rate,
        plan_invalid_ratio=plan_invalid_ratio,
        time_to_goal=time_to_goal,
        timeout_rate=timeouts / total,
        mean_steps=mean_steps,
        reverse_case_success_rate=reverse_successes / reverse_total,
        spin_proxy_rate=spin_proxy_rate,
        progress_stall_rate=progress_stall_rate,
        high_clip_ratio=high_clip_ratio,
        path_efficiency=path_efficiency,
        net_progress_ratio=net_progress_ratio,
        orbit_score=orbit_score,
        near_obstacle_dwell=near_obstacle_dwell,
        sensor_health_score=sensor_health_score,
        log_anomaly_count=float(log_anomaly_count),
        score=score,
        total_episodes=total,
        completed_episodes=total,
    )


def behavior_gate_violations(metrics: EvalMetrics, *, suite: str) -> list[str]:
    violations: list[str] = []
    if suite == "quick":
        if metrics.success_rate < 0.75:
            violations.append("success_rate<0.75")
        if metrics.collision_rate > 0.10:
            violations.append("collision_rate>0.10")
        if metrics.orbit_score > 0.10:
            violations.append("orbit_score>0.10")
        if metrics.progress_stall_rate > 0.25:
            violations.append("progress_stall_rate>0.25")
    else:
        if metrics.success_rate < 0.85:
            violations.append("success_rate<0.85")
        if metrics.collision_rate > 0.05:
            violations.append("collision_rate>0.05")
        if metrics.hard_stop_rate > 0.05:
            violations.append("hard_stop_rate>0.05")
        if metrics.orbit_score > 0.05:
            violations.append("orbit_score>0.05")
        if metrics.spin_proxy_rate > 0.35 and metrics.net_progress_ratio < 0.25:
            violations.append("spin_proxy_rate>0.35_and_net_progress_ratio<0.25")
        if metrics.cmd_saturation_rate > 0.60 and metrics.path_efficiency < 0.45:
            violations.append("cmd_saturation_rate>0.60_and_path_efficiency<0.45")
        if metrics.progress_stall_rate > 0.20:
            violations.append("progress_stall_rate>0.20")
        if metrics.plan_invalid_ratio > 0.0:
            violations.append("plan_invalid_ratio>0")
    if metrics.sensor_health_score < 0.80:
        violations.append("sensor_health_score<0.80")
    if metrics.log_anomaly_count > 0.0:
        violations.append("log_anomaly_count>0")
    return violations
