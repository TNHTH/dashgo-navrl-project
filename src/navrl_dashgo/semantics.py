from __future__ import annotations

import torch


def build_reference_path_progress(max_path_points: int, steps: torch.Tensor) -> torch.Tensor:
    """为前 steps 个有效路径点生成 0..1 的进度，最后一个有效点严格落在终点。"""
    if steps.ndim != 1:
        raise ValueError("steps 必须是一维张量。")
    if max_path_points <= 1:
        raise ValueError("max_path_points 必须大于 1。")

    clamped_steps = torch.clamp(steps.to(dtype=torch.long), min=2, max=max_path_points)
    denom = (clamped_steps - 1).unsqueeze(-1).to(dtype=torch.float32)
    sample_idx = torch.arange(max_path_points, device=steps.device, dtype=torch.float32).unsqueeze(0)
    return torch.clamp(sample_idx / denom, max=1.0)


def compute_waypoint_lookahead_indices(
    reference_path_cursor: torch.Tensor,
    reference_path_len: torch.Tensor,
    lookahead_steps: int,
) -> torch.Tensor:
    """基于当前路径游标选择一个真正前瞻的 waypoint 索引。"""
    if reference_path_cursor.shape != reference_path_len.shape:
        raise ValueError("reference_path_cursor 与 reference_path_len 形状必须一致。")

    effective_lookahead = max(1, int(lookahead_steps))
    cursor = reference_path_cursor.to(dtype=torch.long)
    final_idx = torch.clamp(reference_path_len.to(dtype=torch.long) - 1, min=0)
    return torch.minimum(cursor + effective_lookahead, final_idx)


def restore_flat_history(flat_values: torch.Tensor, history_length: int, feature_dim: int) -> torch.Tensor:
    """按 Isaac Lab 的 flatten_history_dim=True 语义恢复 [history, feature]。"""
    expected_dim = int(history_length) * int(feature_dim)
    if flat_values.shape[-1] != expected_dim:
        raise ValueError(f"历史向量维度不匹配: expected={expected_dim}, actual={flat_values.shape[-1]}")
    return flat_values.reshape(flat_values.shape[0], int(history_length), int(feature_dim))


def restore_lidar_history(flat_lidar: torch.Tensor, history_length: int, num_sectors: int) -> torch.Tensor:
    """把 history-major 的扁平 LiDAR 恢复为 [N, 1, sectors, history] 供 CNN 使用。"""
    history_major = restore_flat_history(flat_lidar, history_length=history_length, feature_dim=num_sectors)
    return history_major.transpose(1, 2).unsqueeze(1)
