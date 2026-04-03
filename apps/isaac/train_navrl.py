#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dashgo_rl.project_paths import TRAIN_CONFIG_ROOT

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo NavRL 风格训练入口")
    parser.add_argument("--config-name", default="train", help="Hydra 配置名，不带 .yaml")
    original_argv = sys.argv[:]
    try:
        # AppLauncher.add_app_launcher_args() 内部会立即 parse_known_args()；
        # 先净化掉 Hydra override，避免 Isaac Kit 在参数检查阶段误触发提前退出。
        sys.argv = [sys.argv[0]]
        AppLauncher.add_app_launcher_args(parser)
    finally:
        sys.argv = original_argv
    return parser


def _maybe_apply_hydra_launcher_overrides(args_cli, hydra_overrides: list[str]) -> None:
    for item in hydra_overrides:
        if item.startswith("headless="):
            args_cli.headless = item.split("=", 1)[1].strip().lower() == "true"
        if item.startswith("enable_cameras="):
            args_cli.enable_cameras = item.split("=", 1)[1].strip().lower() == "true"


def _strip_hydra_overrides_from_sys_argv(hydra_overrides: list[str]) -> None:
    """移除 Hydra 覆盖参数，避免 Kit/SimulationApp 误读后静默提前退出。"""
    if not hydra_overrides:
        return
    override_set = set(hydra_overrides)
    sys.argv = [sys.argv[0], *[arg for arg in sys.argv[1:] if arg not in override_set]]


def load_cfg(config_name: str, overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(TRAIN_CONFIG_ROOT), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def main() -> int:
    parser = build_parser()
    args_cli, hydra_overrides = parser.parse_known_args()
    _maybe_apply_hydra_launcher_overrides(args_cli, hydra_overrides)
    cfg = load_cfg(args_cli.config_name, hydra_overrides)
    resolved_headless = bool(getattr(args_cli, "headless", False) or bool(cfg.headless))
    resolved_enable_cameras = bool(getattr(args_cli, "enable_cameras", False) or bool(cfg.enable_cameras))
    args_cli.headless = resolved_headless
    args_cli.enable_cameras = resolved_enable_cameras
    _strip_hydra_overrides_from_sys_argv(hydra_overrides)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        import torch
        from omegaconf import OmegaConf
        from torch.utils.tensorboard import SummaryWriter
        from torchrl.collectors import SyncDataCollector
        from torchrl.envs.utils import ExplorationType

        from navrl_dashgo.checkpointing import (
            build_checkpoint_payload,
            load_checkpoint_payload,
            load_training_checkpoint,
            resolve_frame_count,
            resolve_remaining_frames,
        )
        from navrl_dashgo.env_adapter import TorchRLDashgoEnv
        from navrl_dashgo.ppo import NonFiniteTrainingStateError, PPO
        from navrl_dashgo.runtime import build_run_layout, write_json

        cfg.headless = resolved_headless
        cfg.enable_cameras = resolved_enable_cameras
        cfg.env.map_source = str(getattr(cfg.env, "map_source", "dashgo_official")).strip().lower()
        resume_optimizer_state = bool(getattr(cfg, "resume_optimizer_state", False))
        resume_from_value = getattr(cfg, "resume_from", None)
        resume_checkpoint = None
        if resume_from_value not in (None, "", "null"):
            resume_checkpoint = Path(str(resume_from_value)).expanduser().resolve()
            if not resume_checkpoint.exists():
                raise FileNotFoundError(f"resume_from 指向的 checkpoint 不存在: {resume_checkpoint}")

        print(
            "[DashGo-NavRL] resolved_config "
            f"profile={cfg.profile} headless={cfg.headless} enable_cameras={cfg.enable_cameras} "
            f"map_source={cfg.env.map_source} num_envs={cfg.env.num_envs}",
            flush=True,
        )

        env = TorchRLDashgoEnv(cfg)
        algo = PPO(cfg.algo, env.observation_spec, env.agent_action_spec, env.device)
        algo.train()
        resume_notes: list[str] = []
        start_frame_count = 0
        if resume_checkpoint is not None:
            resume_payload = load_checkpoint_payload(resume_checkpoint, map_location=env.device)
            resume_notes = load_training_checkpoint(
                algo,
                resume_payload,
                load_optimizer_state=resume_optimizer_state,
            )
            start_frame_count = resolve_frame_count(resume_payload)

        layout = build_run_layout(str(cfg.profile))
        write_json(layout.config_snapshot, OmegaConf.to_container(cfg, resolve=True))
        if resume_checkpoint is not None:
            write_json(
                layout.run_root / "resume_state.json",
                {
                    "resume_from": str(resume_checkpoint),
                    "resume_optimizer_state": resume_optimizer_state,
                    "start_frame_count": start_frame_count,
                    "resume_notes": resume_notes,
                },
            )
        tb_writer = SummaryWriter(log_dir=str(layout.tensorboard_root), flush_secs=30, max_queue=20)
        tb_writer.add_text("run/profile", str(cfg.profile), 0)
        tb_writer.add_text("run/config_yaml", OmegaConf.to_yaml(cfg), 0)
        if resume_checkpoint is not None:
            tb_writer.add_text("run/resume_from", str(resume_checkpoint), start_frame_count)
            tb_writer.add_text("run/resume_notes", "\n".join(resume_notes) or "none", start_frame_count)

        wandb_run = None
        if str(cfg.wandb.mode).lower() != "disabled":
            import wandb

            wandb_run = wandb.init(
                project=str(cfg.wandb.project),
                name=layout.run_root.name,
                entity=None if cfg.wandb.entity in (None, "null") else str(cfg.wandb.entity),
                mode=str(cfg.wandb.mode),
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        frames_per_batch = int(cfg.env.num_envs) * int(cfg.algo.training_frame_num)
        target_frame_count = int(cfg.max_frame_num)
        collector_total_frames = (
            resolve_remaining_frames(target_frame_count, start_frame_count)
            if resume_checkpoint is not None
            else target_frame_count
        )
        starting_batch_idx = start_frame_count // frames_per_batch
        collector = SyncDataCollector(
            env,
            policy=algo,
            frames_per_batch=frames_per_batch,
            total_frames=collector_total_frames,
            device=env.device,
            storing_device=env.device,
            policy_device=env.device,
            env_device=env.device,
            return_same_td=True,
            reset_when_done=True,
            exploration_type=ExplorationType.RANDOM,
        )

        def save_checkpoint(tag: str, frame_count: int) -> Path:
            checkpoint_path = layout.checkpoint_root / f"{tag}.pt"
            torch.save(
                build_checkpoint_payload(
                    algo,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    frame_count=frame_count,
                    profile=str(cfg.profile),
                ),
                checkpoint_path,
            )
            return checkpoint_path

        print(f"[DashGo-NavRL] run_root={layout.run_root}", flush=True)
        print(f"[DashGo-NavRL] tensorboard_root={layout.tensorboard_root}", flush=True)
        print(
            f"[DashGo-NavRL] profile={cfg.profile} num_envs={cfg.env.num_envs} "
            f"frames_per_batch={frames_per_batch} total_frames={target_frame_count}",
            flush=True,
        )
        if resume_checkpoint is not None:
            print(
                f"[DashGo-NavRL] resume_from={resume_checkpoint} start_frame_count={start_frame_count} "
                f"remaining_frames={collector_total_frames} "
                f"resume_optimizer_state={resume_optimizer_state}",
                flush=True,
            )
            if resume_notes:
                print(f"[DashGo-NavRL] resume_notes={' | '.join(resume_notes)}", flush=True)

        for batch_idx, data in enumerate(collector):
            global_batch_idx = starting_batch_idx + batch_idx
            stats = algo.update(data)
            frame_count = start_frame_count + int((batch_idx + 1) * frames_per_batch)
            stats["env_frames"] = frame_count
            stats["batch_idx"] = global_batch_idx

            tb_writer.add_scalar("train/actor_loss", float(stats["actor_loss"]), frame_count)
            tb_writer.add_scalar("train/critic_loss", float(stats["critic_loss"]), frame_count)
            tb_writer.add_scalar("train/entropy", float(stats["entropy"]), frame_count)
            tb_writer.add_scalar("train/explained_var", float(stats["explained_var"]), frame_count)
            tb_writer.add_scalar("train/batch_idx", float(global_batch_idx), frame_count)
            tb_writer.add_scalar("train/num_envs", float(cfg.env.num_envs), frame_count)
            tb_writer.add_scalar("train/static_obstacles", float(cfg.env.static_obstacles), frame_count)
            tb_writer.add_scalar("train/dynamic_obstacles", float(cfg.env.dynamic_obstacles), frame_count)

            if global_batch_idx % int(cfg.logging.print_interval_batches) == 0:
                print(
                    f"[DashGo-NavRL] batch={global_batch_idx} frames={frame_count} "
                    f"actor_loss={stats['actor_loss']:.4f} critic_loss={stats['critic_loss']:.4f} "
                    f"entropy={stats['entropy']:.4f} explained_var={stats['explained_var']:.4f}",
                    flush=True,
                )
            if global_batch_idx % int(cfg.save_interval_batches) == 0:
                checkpoint = save_checkpoint(f"checkpoint_{frame_count}", frame_count)
                print(f"[DashGo-NavRL] checkpoint={checkpoint}", flush=True)
                tb_writer.flush()
            if wandb_run is not None:
                wandb_run.log(stats)

        final_checkpoint = save_checkpoint("checkpoint_final", target_frame_count)
        print(f"[DashGo-NavRL] final_checkpoint={final_checkpoint}", flush=True)
        tb_writer.add_text("run/final_checkpoint", str(final_checkpoint), target_frame_count)
        tb_writer.flush()
        tb_writer.close()

        if wandb_run is not None:
            wandb_run.finish()
        collector.shutdown()
        env.close()
        return 0
    except NonFiniteTrainingStateError as exc:
        print("[DashGo-NavRL] failure_reason=non_finite_training_state", flush=True)
        print(f"[DashGo-NavRL] non_finite_detail={exc}", flush=True)
        traceback.print_exc()
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[DashGo-NavRL] failure_reason=training_runtime_error:{type(exc).__name__}", flush=True)
        print(f"[DashGo-NavRL] runtime_error_detail={exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
