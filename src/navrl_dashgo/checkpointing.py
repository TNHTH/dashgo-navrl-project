from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def normalize_checkpoint_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return {"model_state_dict": payload}


def load_checkpoint_payload(checkpoint: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return normalize_checkpoint_payload(torch.load(Path(checkpoint), map_location=map_location))


def build_inference_state_dict(algo: Any) -> dict[str, dict[str, Any]]:
    return {
        "feature_extractor": algo.feature_extractor.state_dict(),
        "actor": algo.actor.state_dict(),
        "critic": algo.critic.state_dict(),
        "value_norm": algo.value_norm.state_dict(),
    }


def build_optimizer_state_dict(algo: Any) -> dict[str, dict[str, Any]]:
    return {
        "feature_extractor": algo.feature_extractor_optim.state_dict(),
        "actor": algo.actor_optim.state_dict(),
        "critic": algo.critic_optim.state_dict(),
    }


def build_checkpoint_payload(algo: Any, *, config: dict[str, Any], frame_count: int, profile: str) -> dict[str, Any]:
    return {
        "checkpoint_version": 2,
        "model_state_dict": algo.state_dict(),
        "inference_state_dict": build_inference_state_dict(algo),
        "optimizer_state_dict": build_optimizer_state_dict(algo),
        "config": config,
        "frame_count": int(frame_count),
        "profile": str(profile),
    }


def resolve_frame_count(payload: Any) -> int:
    normalized = normalize_checkpoint_payload(payload)
    try:
        frame_count = int(normalized.get("frame_count", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint.frame_count 非法: {normalized.get('frame_count')!r}") from exc
    return max(frame_count, 0)


def resolve_remaining_frames(target_frame_count: int, start_frame_count: int) -> int:
    target = int(target_frame_count)
    start = int(start_frame_count)
    remaining = target - start
    if remaining <= 0:
        raise ValueError(
            f"目标 max_frame_num={target} 必须大于 checkpoint.frame_count={start}，否则没有可继续训练的帧。"
        )
    return remaining


def load_training_checkpoint(algo: Any, payload: Any, *, load_optimizer_state: bool = True) -> list[str]:
    normalized = normalize_checkpoint_payload(payload)
    notes: list[str] = []

    inference_state = normalized.get("inference_state_dict")
    if isinstance(inference_state, dict):
        algo.feature_extractor.load_state_dict(inference_state["feature_extractor"], strict=True)
        algo.actor.load_state_dict(inference_state["actor"], strict=True)
        algo.critic.load_state_dict(inference_state["critic"], strict=True)
        value_norm_state = inference_state.get("value_norm")
        if isinstance(value_norm_state, dict):
            load_result = algo.value_norm.load_state_dict(value_norm_state, strict=False)
            if load_result.missing_keys:
                notes.append(f"value_norm_missing_keys={load_result.missing_keys}")
            if load_result.unexpected_keys:
                notes.append(f"value_norm_unexpected_keys={load_result.unexpected_keys}")
    else:
        state_dict = normalized["model_state_dict"]
        load_result = algo.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = ("value_norm.", "gae.")
        critical_missing = [key for key in load_result.missing_keys if not key.startswith(allowed_missing_prefixes)]
        if critical_missing or load_result.unexpected_keys:
            raise RuntimeError(
                "checkpoint 与训练模型结构不兼容: "
                f"critical_missing={critical_missing}, unexpected={load_result.unexpected_keys}"
            )
        if load_result.missing_keys:
            notes.append(f"legacy_checkpoint_missing_keys={load_result.missing_keys}")

    optimizer_state = normalized.get("optimizer_state_dict")
    if not load_optimizer_state:
        if isinstance(optimizer_state, dict):
            notes.append("optimizer_state_skipped=disabled")
        else:
            notes.append("optimizer_state_missing=all")
        return notes
    if not isinstance(optimizer_state, dict):
        notes.append("optimizer_state_missing=all")
        return notes

    optimizer_specs = (
        ("feature_extractor", algo.feature_extractor_optim),
        ("actor", algo.actor_optim),
        ("critic", algo.critic_optim),
    )
    for key, optimizer in optimizer_specs:
        state = optimizer_state.get(key)
        if state is None:
            notes.append(f"optimizer_state_missing={key}")
            continue
        try:
            optimizer.load_state_dict(state)
        except ValueError as exc:
            notes.append(f"optimizer_state_incompatible={key}:{exc}")
    return notes
