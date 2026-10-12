"""Task-independent entry point for AnyTouch2 + Pi0.5 prefix concatenation."""

import dataclasses
import logging
import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import tyro

from openpi.training import config as training_config


@dataclasses.dataclass
class Args:
    mode: Literal["stats", "train", "serve"]
    # LeRobot ID, also used to locate checkpoint normalization assets when serving.
    repo_id: str
    anytouch_checkpoint: str = "assets/pretrained/anytouch2/checkpoint-4frames.pth"
    # Directory containing model.safetensors, not the file itself.
    base_checkpoint: str = "assets/pretrained/pi05_base_pytorch"
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    exp_name: str = "run1"
    batch_size: int = 64
    num_workers: int = 2
    num_train_steps: int = 10_000
    save_interval: int = 5_000
    keep_period: int = 5_000
    resume: bool = False
    wandb_enabled: bool = False
    # Stats only: limit dataset frames for a quick check; omit for final statistics.
    max_frames: int | None = None
    # Serve only: directory containing model.safetensors and assets/.
    checkpoint: str | None = None
    port: int = 9000
    host: str = "0.0.0.0"
    dry_run: bool = False


def build_config(args: Args) -> training_config.TrainConfig:
    if args.batch_size < 1 or args.num_workers < 0 or args.num_train_steps < 1:
        raise ValueError("batch_size and num_train_steps must be positive; num_workers must be nonnegative")
    if args.save_interval < 1 or args.keep_period < 1:
        raise ValueError("save_interval and keep_period must be positive")
    if args.max_frames is not None and args.max_frames < args.batch_size:
        raise ValueError("max_frames must be at least batch_size")
    if not args.exp_name or Path(args.exp_name).name != args.exp_name or args.exp_name in {".", ".."}:
        raise ValueError("exp_name must be a single directory name")
    config = training_config.make_anytouch2_config(
        args.repo_id,
        anytouch_checkpoint_path=args.anytouch_checkpoint,
        pytorch_weight_path=args.base_checkpoint,
    )
    return dataclasses.replace(
        config,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        resume=args.resume,
        wandb_enabled=args.wandb_enabled,
    )


def main(args: Args) -> None:
    config = build_config(args)
    logging.info("Mode=%s repo_id=%s config=%s", args.mode, args.repo_id, config.name)
    if args.dry_run:
        print(config)
        return
    if args.mode == "stats":
        from compute_norm_stats import compute

        compute(config, args.max_frames)
    elif args.mode == "train":
        from train_pytorch import train_loop

        for file in (
            Path(args.anytouch_checkpoint),
            config.assets_dirs / args.repo_id / "norm_stats.json",
        ):
            if not file.is_file():
                raise FileNotFoundError(file)
        if not args.resume and not (Path(args.base_checkpoint) / "model.safetensors").is_file():
            raise FileNotFoundError(Path(args.base_checkpoint) / "model.safetensors")
        train_loop(config)
    else:
        from openpi.policies.policy_config import create_trained_policy
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer

        if args.checkpoint is None:
            raise ValueError("serve requires --checkpoint")
        policy = create_trained_policy(config, args.checkpoint)
        WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
