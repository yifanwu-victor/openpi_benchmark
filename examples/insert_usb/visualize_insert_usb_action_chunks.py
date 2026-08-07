#!/usr/bin/env python3
"""Visualize OpenPI action chunks against Insert USB LeRobot training data.

Recommended usage from the OpenPI repo root:

    cd /shareNFS_40/wyf/workplace/openpi

    CUDA_VISIBLE_DEVICES=0 XLA_FLAGS=--xla_gpu_autotune_level=0 \
    uv run examples/insert_usb/visualize_insert_usb_action_chunks.py \
        --episode 0 \
        --stride 10 \
        --horizon 10 \
        --output-dir outputs/insert_usb_vis/lora_20000_ep0

The script loads a trained policy directly from checkpoint. It does not use the
websocket policy server. For each selected frame t = 0, stride, 2*stride, ...,
it feeds the observation at t to the policy, receives an action chunk, and
concatenates the chunk xyz targets into one long predicted trajectory.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import polars as pl

from openpi.policies import policy_config
from openpi.training import config as training_config


DEFAULT_CONFIG_NAME = "pi05_insert_usb_real_abs_eef_vision_lora"
DEFAULT_CHECKPOINT_DIR = Path(
    "/shareNFS_40/wyf/workplace/openpi/checkpoints/"
    "pi05_insert_usb_real_abs_eef_vision_lora/"
    "insert_usb_vision_lora_v1/29999"
)
DEFAULT_DATASET_DIR = Path("/shareNFS_40/wyf/data/lerobot/wyf/insert_usb_real_abs_eef")
DEFAULT_OUTPUT_DIR = Path("outputs/insert_usb_vis/lora_29999_ep0")
DEFAULT_PROMPT = "pick and insert the USB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OpenPI predicted action chunks with Insert USB LeRobot training actions."
    )
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME, help=f"Train config name. Default: {DEFAULT_CONFIG_NAME}")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Checkpoint step directory. Default: {DEFAULT_CHECKPOINT_DIR}",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"LeRobot dataset root. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize. Default: 0.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to sample. Default: 0.")
    parser.add_argument("--stride", type=int, default=10, help="Observation stride in frames. Default: 10.")
    parser.add_argument("--horizon", type=int, default=10, help="Number of actions to keep from each chunk. Default: 10.")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional limit on sampled chunks for quick checks. Default: use the full episode.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent with every observation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_frame < 0:
        raise ValueError(f"--start-frame must be non-negative, got {args.start_frame}")
    if args.stride <= 0:
        raise ValueError(f"--stride must be positive, got {args.stride}")
    if args.horizon <= 0:
        raise ValueError(f"--horizon must be positive, got {args.horizon}")
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError(f"--max-chunks must be positive if set, got {args.max_chunks}")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = load_dataset_info(dataset_dir)
    episode_path = episode_parquet_path(dataset_dir, info, args.episode)
    df = pl.read_parquet(episode_path)
    if len(df) < args.horizon:
        raise ValueError(f"Episode {args.episode} has {len(df)} frame(s), shorter than horizon {args.horizon}")

    starts = np.arange(args.start_frame, len(df) - args.horizon + 1, args.stride, dtype=np.int64)
    if args.max_chunks is not None:
        starts = starts[: args.max_chunks]
    if len(starts) == 0:
        raise ValueError(
            f"No chunks selected: len(df)={len(df)}, start_frame={args.start_frame}, horizon={args.horizon}, stride={args.stride}"
        )

    print(f"Dataset: {dataset_dir}")
    print(f"Episode parquet: {episode_path}")
    print(f"Frames: {len(df)}")
    print(f"FPS: {info.get('fps')}")
    print(f"Config: {args.config_name}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Selected chunks: {len(starts)}")
    print(f"Stride/horizon: {args.stride}/{args.horizon}")

    train_config = training_config.get_config(args.config_name)
    model_horizon = int(train_config.model.action_horizon)
    model_action_dim = int(train_config.model.action_dim)
    if args.horizon > model_horizon:
        raise ValueError(f"--horizon {args.horizon} exceeds model action_horizon {model_horizon}")
    if model_action_dim < 10:
        raise ValueError(f"Expected model action_dim >= 10, got {model_action_dim}")

    print("Loading policy...")
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)

    pred_xyz_chunks = []
    gt_xyz_chunks = []
    state_xyz_chunks = []
    pred_gripper_chunks = []
    gt_gripper_chunks = []
    state_gripper_chunks = []
    all_start_indices = []
    for chunk_idx, start in enumerate(starts):
        start = int(start)
        row = df.row(start, named=True)
        obs = make_observation(row, args.prompt)
        result = policy.infer(obs)
        if "actions" not in result:
            raise RuntimeError(f"Policy result has no 'actions' key: {list(result.keys())}")
        pred_actions = np.asarray(result["actions"], dtype=np.float32)
        if pred_actions.ndim != 2 or pred_actions.shape[1] < 10:
            raise ValueError(f"Expected predicted actions [T, >=10], got {pred_actions.shape}")
        if pred_actions.shape[0] < args.horizon:
            raise ValueError(f"Predicted chunk has length {pred_actions.shape[0]}, shorter than horizon {args.horizon}")

        gt_actions = np.asarray(df["actions"][start : start + args.horizon].to_list(), dtype=np.float32)
        states = np.asarray(df["state"][start : start + args.horizon].to_list(), dtype=np.float32)
        if gt_actions.shape != (args.horizon, 10):
            raise ValueError(f"Expected gt actions [{args.horizon}, 10], got {gt_actions.shape} at start {start}")
        if states.shape != (args.horizon, 10):
            raise ValueError(f"Expected states [{args.horizon}, 10], got {states.shape} at start {start}")

        pred_xyz_chunks.append(pred_actions[: args.horizon, :3])
        gt_xyz_chunks.append(gt_actions[:, :3])
        state_xyz_chunks.append(states[:, :3])
        pred_gripper_chunks.append(pred_actions[: args.horizon, 9])
        gt_gripper_chunks.append(gt_actions[:, 9])
        state_gripper_chunks.append(states[:, 9])
        all_start_indices.extend(range(start, start + args.horizon))

        if chunk_idx % 10 == 0 or chunk_idx == len(starts) - 1:
            print(f"  chunk {chunk_idx + 1}/{len(starts)} start={start}")

    pred_xyz = np.concatenate(pred_xyz_chunks, axis=0)
    gt_xyz = np.concatenate(gt_xyz_chunks, axis=0)
    state_xyz = np.concatenate(state_xyz_chunks, axis=0)
    pred_gripper = np.concatenate(pred_gripper_chunks, axis=0)
    gt_gripper = np.concatenate(gt_gripper_chunks, axis=0)
    state_gripper = np.concatenate(state_gripper_chunks, axis=0)
    frame_indices = np.asarray(all_start_indices, dtype=np.int64)
    if pred_xyz.shape != gt_xyz.shape:
        raise RuntimeError(f"pred_xyz shape {pred_xyz.shape} does not match gt_xyz {gt_xyz.shape}")

    xyz_error = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
    delta_pred = pred_xyz - state_xyz
    delta_gt = gt_xyz - state_xyz
    delta_error = np.linalg.norm(delta_pred - delta_gt, axis=1)
    gripper_error = np.abs(pred_gripper - gt_gripper)

    summary = {
        "dataset_dir": str(dataset_dir),
        "episode": int(args.episode),
        "episode_path": str(episode_path),
        "fps": info.get("fps"),
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "stride": int(args.stride),
        "horizon": int(args.horizon),
        "num_chunks": int(len(starts)),
        "num_points": int(len(pred_xyz)),
        "xyz_error_mean_m": float(np.mean(xyz_error)),
        "xyz_error_max_m": float(np.max(xyz_error)),
        "delta_xyz_error_mean_m": float(np.mean(delta_error)),
        "delta_xyz_error_max_m": float(np.max(delta_error)),
        "gripper_error_mean": float(np.mean(gripper_error)),
        "gripper_error_max": float(np.max(gripper_error)),
    }

    save_arrays(
        output_dir,
        frame_indices,
        pred_xyz,
        gt_xyz,
        state_xyz,
        pred_gripper,
        gt_gripper,
        state_gripper,
        xyz_error,
        delta_error,
        gripper_error,
    )
    plot_xyz_over_time(
        output_dir / "xyz_over_time.png",
        frame_indices,
        pred_xyz,
        gt_xyz,
        state_xyz,
        pred_gripper,
        gt_gripper,
        state_gripper,
    )
    plot_xyz_3d(output_dir / "xyz_3d.png", pred_xyz, gt_xyz, state_xyz)
    plot_xyz_error(output_dir / "xyz_error.png", frame_indices, xyz_error, delta_error)
    save_input_preview(output_dir / "input_preview.png", df, starts[: min(8, len(starts))])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nWrote visualizations to: {output_dir}")


def load_dataset_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot info.json: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def episode_parquet_path(dataset_dir: Path, info: dict, episode_index: int) -> Path:
    if episode_index < 0:
        raise ValueError(f"--episode must be non-negative, got {episode_index}")
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    data_path = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    path = dataset_dir / data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
    if not path.is_file():
        raise FileNotFoundError(f"Missing episode parquet: {path}")
    return path


def make_observation(row: dict, prompt: str) -> dict:
    return {
        "observation/image": decode_lerobot_image(row["image"]),
        "observation/wrist_image": decode_lerobot_image(row["wrist_image"]),
        "observation/state": np.asarray(row["state"], dtype=np.float32),
        "prompt": prompt,
    }


def decode_lerobot_image(value) -> np.ndarray:
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes is None:
            raise ValueError(f"Image entry has no bytes field: {value}")
        with Image.open(io.BytesIO(image_bytes)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    raise TypeError(f"Expected LeRobot image dict with bytes/path fields, got {type(value)}")


def save_arrays(
    output_dir: Path,
    frame_indices: np.ndarray,
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    state_xyz: np.ndarray,
    pred_gripper: np.ndarray,
    gt_gripper: np.ndarray,
    state_gripper: np.ndarray,
    xyz_error: np.ndarray,
    delta_error: np.ndarray,
    gripper_error: np.ndarray,
) -> None:
    np.savez(
        output_dir / "action_chunk_xyz_compare.npz",
        frame_indices=frame_indices,
        pred_xyz=pred_xyz,
        gt_xyz=gt_xyz,
        state_xyz=state_xyz,
        pred_gripper=pred_gripper,
        gt_gripper=gt_gripper,
        state_gripper=state_gripper,
        xyz_error=xyz_error,
        delta_xyz_error=delta_error,
        gripper_error=gripper_error,
    )


def plot_xyz_over_time(
    output_path: Path,
    frame_indices: np.ndarray,
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    state_xyz: np.ndarray,
    pred_gripper: np.ndarray,
    gt_gripper: np.ndarray,
    state_gripper: np.ndarray,
) -> None:
    labels = ("x", "y", "z")
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for dim, ax in enumerate(axes[:3]):
        ax.plot(frame_indices, gt_xyz[:, dim], label="gt action xyz", linewidth=1.4)
        ax.plot(frame_indices, pred_xyz[:, dim], label="pred action xyz", linewidth=1.2, alpha=0.9)
        ax.plot(frame_indices, state_xyz[:, dim], label="state xyz", linewidth=0.9, alpha=0.6, linestyle="--")
        ax.set_ylabel(f"{labels[dim]} (m)")
        ax.grid(True, alpha=0.25)
    axes[3].plot(frame_indices, gt_gripper, label="gt action gripper", linewidth=1.4)
    axes[3].plot(frame_indices, pred_gripper, label="pred action gripper", linewidth=1.2, alpha=0.9)
    axes[3].plot(frame_indices, state_gripper, label="state gripper", linewidth=0.9, alpha=0.6, linestyle="--")
    axes[3].set_ylabel("gripper")
    axes[3].grid(True, alpha=0.25)
    axes[-1].set_xlabel("frame index")
    axes[0].legend(loc="best")
    axes[3].legend(loc="best")
    fig.suptitle("Insert USB action chunk xyz and gripper targets")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_xyz_3d(output_path: Path, pred_xyz: np.ndarray, gt_xyz: np.ndarray, state_xyz: np.ndarray) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="gt action xyz", linewidth=1.5)
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], label="pred action xyz", linewidth=1.2)
    ax.plot(state_xyz[:, 0], state_xyz[:, 1], state_xyz[:, 2], label="state xyz", linewidth=0.9, alpha=0.55)
    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], marker="o", s=36, label="start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Insert USB 3D EEF target trajectory")
    ax.legend(loc="best")
    set_axes_equal_3d(ax, np.concatenate([pred_xyz, gt_xyz, state_xyz], axis=0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_xyz_error(
    output_path: Path,
    frame_indices: np.ndarray,
    xyz_error: np.ndarray,
    delta_error: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(frame_indices, xyz_error, label="||pred_xyz - gt_xyz||", linewidth=1.4)
    ax.plot(frame_indices, delta_error, label="||pred_delta_xyz - gt_delta_xyz||", linewidth=1.2)
    ax.set_xlabel("frame index")
    ax.set_ylabel("error (m)")
    ax.set_title("Insert USB xyz target error")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_input_preview(output_path: Path, df: pl.DataFrame, starts: np.ndarray) -> None:
    if len(starts) == 0:
        return
    rows = []
    for start in starts:
        row = df.row(int(start), named=True)
        image = decode_lerobot_image(row["image"])
        wrist = decode_lerobot_image(row["wrist_image"])
        rows.append(np.concatenate([image, wrist], axis=1))
    preview = np.concatenate(rows, axis=0)
    Image.fromarray(preview).save(output_path)


def set_axes_equal_3d(ax, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.5)
    radius = max(radius, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


if __name__ == "__main__":
    main()
