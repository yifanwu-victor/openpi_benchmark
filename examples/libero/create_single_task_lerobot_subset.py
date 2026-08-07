#!/usr/bin/env python3
"""Create a local single-task LIBERO LeRobot dataset subset.

Recommended usage from the OpenPI repo root:

    cd /shareNFS_40/wyf/workplace/openpi
    export HF_LEROBOT_HOME=/shareNFS_40/wyf/data/lerobot
    uv run examples/libero/create_single_task_lerobot_subset.py --overwrite

The default source is physical-intelligence/libero and the default output is:
    /shareNFS_40/wyf/data/lerobot/wyf/libero_task0_debug
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np

DEFAULT_SOURCE_REPO = "physical-intelligence/libero"
DEFAULT_REPO_ID = "wyf/libero_task0_debug"
DEFAULT_TASK_INDEX = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a local single-task LIBERO LeRobot subset.")
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO, help=f"Source LeRobot repo. Default: {DEFAULT_SOURCE_REPO}")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Output local LeRobot repo id. Default: {DEFAULT_REPO_ID}")
    parser.add_argument("--output-root", type=Path, default=None, help="Output dataset root. Default: $HF_LEROBOT_HOME/<repo-id>")
    parser.add_argument("--task-index", type=int, default=DEFAULT_TASK_INDEX, help=f"Source task index. Default: {DEFAULT_TASK_INDEX}")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap on converted episodes.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output dataset if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected episodes without writing a dataset.")
    parser.add_argument("--image-writer-processes", type=int, default=0, help="LeRobot image writer processes. Default: 0.")
    parser.add_argument("--image-writer-threads", type=int, default=4, help="LeRobot image writer threads. Default: 4.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task_index < 0:
        raise ValueError(f"--task-index must be non-negative, got {args.task_index}")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError(f"--max-episodes must be positive if set, got {args.max_episodes}")

    output_root = (args.output_root or (HF_LEROBOT_HOME / args.repo_id)).expanduser().resolve()
    source_meta = LeRobotDatasetMetadata(args.source_repo)
    if args.task_index not in source_meta.tasks:
        raise ValueError(f"Task index {args.task_index} not found. Available: {sorted(source_meta.tasks)}")

    task = source_meta.tasks[args.task_index]
    selected_episodes = [
        int(episode_index)
        for episode_index, episode in sorted(source_meta.episodes.items())
        if task in episode["tasks"]
    ]
    if args.max_episodes is not None:
        selected_episodes = selected_episodes[: args.max_episodes]
    if not selected_episodes:
        raise ValueError(f"No episodes found for task_index={args.task_index}: {task!r}")

    total_frames = sum(int(source_meta.episodes[episode_index]["length"]) for episode_index in selected_episodes)
    print(f"Source repo: {args.source_repo}")
    print(f"Task index: {args.task_index}")
    print(f"Task: {task}")
    print(f"Selected episodes: {len(selected_episodes)}")
    print(f"Selected frames: {total_frames}")
    print(f"First source episodes: {selected_episodes[:10]}")
    print(f"Output repo id: {args.repo_id}")
    print(f"Output root: {output_root}")

    if args.dry_run:
        print("Dry run only; no LeRobot dataset was written.")
        return

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        robot_type=source_meta.info.get("robot_type", "panda"),
        fps=int(source_meta.fps),
        features={
            "image": source_meta.features["image"],
            "wrist_image": source_meta.features["wrist_image"],
            "state": source_meta.features["state"],
            "actions": source_meta.features["actions"],
        },
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    converted_frames = 0
    try:
        for local_episode_index, source_episode_index in enumerate(selected_episodes):
            source_dataset = LeRobotDataset(args.source_repo, episodes=[source_episode_index], download_videos=False)
            source_frames = 0
            for row in source_dataset:
                row_task_index = int(to_numpy(row["task_index"]))
                if row_task_index != args.task_index:
                    raise ValueError(
                        f"Unexpected task_index={row_task_index} in source episode {source_episode_index}; "
                        f"expected {args.task_index}"
                    )
                dataset.add_frame(
                    {
                        "image": image_to_uint8_hwc(row["image"]),
                        "wrist_image": image_to_uint8_hwc(row["wrist_image"]),
                        "state": to_numpy(row["state"]).astype(np.float32),
                        "actions": to_numpy(row["actions"]).astype(np.float32),
                        "task": str(row["task"]),
                    }
                )
                source_frames += 1
            dataset.save_episode()
            converted_frames += source_frames
            print(
                f"Saved local episode {local_episode_index:04d} from source episode "
                f"{source_episode_index:06d} with {source_frames} frames"
            )
    finally:
        dataset.stop_image_writer()

    print(f"Done. Wrote {len(selected_episodes)} episodes and {converted_frames} frames to {output_root}")


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def image_to_uint8_hwc(value) -> np.ndarray:
    image = to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0) * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with final dim 3, got shape {image.shape}")
    return image


if __name__ == "__main__":
    main()
