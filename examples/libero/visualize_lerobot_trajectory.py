#!/usr/bin/env python3
"""Visualize image streams and state/action traces from a LIBERO LeRobot dataset."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import polars as pl

DEFAULT_DATASET_DIR = Path("/shareNFS_40/wyf/data/lerobot/wyf/libero_task0_debug")
DEFAULT_OUTPUT_DIR = Path("/shareNFS_40/wyf/data/lerobot/wyf/libero_task0_debug/visualizations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LIBERO LeRobot trajectory videos and state/action plots.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help=f"Dataset root. Default: {DEFAULT_DATASET_DIR}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Episode indices to visualize. Default: first N episodes.")
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of episodes to visualize when --episodes is omitted.")
    parser.add_argument("--video-skip", type=int, default=2, help="Write one frame every N dataset frames. Default: 2.")
    parser.add_argument("--video-fps", type=float, default=None, help="Output video FPS. Default: dataset fps / video-skip.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError(f"--num-episodes must be positive, got {args.num_episodes}")
    if args.video_skip <= 0:
        raise ValueError(f"--video-skip must be positive, got {args.video_skip}")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    info = load_json(dataset_dir / "meta" / "info.json")
    episodes = load_episodes(dataset_dir)
    selected = args.episodes if args.episodes is not None else [ep["episode_index"] for ep in episodes[: args.num_episodes]]
    if not selected:
        raise ValueError("No episodes selected")

    fps = float(args.video_fps or max(float(info.get("fps", 10)) / args.video_skip, 1.0))
    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "fps": fps,
        "video_skip": args.video_skip,
        "episodes": [],
    }

    for episode_index in selected:
        episode_path = episode_parquet_path(dataset_dir, info, episode_index)
        episode_df = pl.read_parquet(episode_path)
        if len(episode_df) == 0:
            raise ValueError(f"Episode {episode_index} has no frames: {episode_path}")

        video_path = output_dir / f"episode_{episode_index:06d}_views.mp4"
        plot_path = output_dir / f"episode_{episode_index:06d}_state_actions.png"
        export_video(episode_df, video_path, fps=fps, video_skip=args.video_skip)
        export_state_action_plot(episode_df, plot_path)

        actions = np.asarray(episode_df["actions"].to_list(), dtype=np.float32)
        states = np.asarray(episode_df["state"].to_list(), dtype=np.float32)
        if not np.isfinite(actions).all():
            raise ValueError(f"Episode {episode_index} actions contain NaN or inf")
        if not np.isfinite(states).all():
            raise ValueError(f"Episode {episode_index} states contain NaN or inf")
        if actions.shape[1:] != (7,):
            raise ValueError(f"Expected action shape (*, 7), got {actions.shape}")

        episode_summary = {
            "episode_index": int(episode_index),
            "frames": len(episode_df),
            "episode_path": str(episode_path),
            "video_path": str(video_path),
            "plot_path": str(plot_path),
            "action_min": actions.min(axis=0).tolist(),
            "action_max": actions.max(axis=0).tolist(),
            "state_min": states.min(axis=0).tolist(),
            "state_max": states.max(axis=0).tolist(),
        }
        summary["episodes"].append(episode_summary)
        print(f"Episode {episode_index}: wrote {video_path} and {plot_path}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_episodes(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing episodes metadata: {path}")
    episodes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(episodes, key=lambda item: int(item["episode_index"]))


def episode_parquet_path(dataset_dir: Path, info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    data_path = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    path = dataset_dir / data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
    if not path.is_file():
        raise FileNotFoundError(f"Missing episode parquet: {path}")
    return path


def export_video(df: pl.DataFrame, video_path: Path, *, fps: float, video_skip: int) -> None:
    with imageio.get_writer(video_path, fps=fps) as writer:
        for index, row in enumerate(df.iter_rows(named=True)):
            if index % video_skip != 0:
                continue
            image = decode_lerobot_image(row["image"])
            wrist = decode_lerobot_image(row["wrist_image"])
            frame = np.concatenate([image, wrist], axis=1)
            writer.append_data(frame)


def export_state_action_plot(df: pl.DataFrame, plot_path: Path) -> None:
    timestamps = np.asarray(df["timestamp"].to_list(), dtype=np.float32)
    states = np.asarray(df["state"].to_list(), dtype=np.float32)
    actions = np.asarray(df["actions"].to_list(), dtype=np.float32)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for dim in range(states.shape[1]):
        axes[0].plot(timestamps, states[:, dim], linewidth=1.0, label=f"s{dim}")
    axes[0].set_title("State")
    axes[0].grid(visible=True, alpha=0.3)
    axes[0].legend(ncol=4, fontsize=8)

    for dim in range(actions.shape[1]):
        axes[1].plot(timestamps, actions[:, dim], linewidth=1.0, label=f"a{dim}")
    axes[1].set_title("Actions")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(visible=True, alpha=0.3)
    axes[1].legend(ncol=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def decode_lerobot_image(value) -> np.ndarray:
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes is not None:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        image_path = value.get("path")
        if image_path is not None:
            with Image.open(image_path) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(value)
    if array.ndim == 3:
        if array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array, 0.0, 1.0) * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array
    raise TypeError(f"Unsupported image value type: {type(value)}")


if __name__ == "__main__":
    main()
