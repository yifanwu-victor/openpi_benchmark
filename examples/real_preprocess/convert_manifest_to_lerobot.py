#!/usr/bin/env python3
"""Convert Insert USB real raw data to LeRobot using a preprocessing manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

from real_preprocess_utils import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPO_ID,
    IMAGE_STREAMS,
    delta_eef_actions,
    inspect_episode,
    parse_manifest_segments,
    progress,
    read_yaml,
    repo_id_to_path,
    resize_rgb_image,
    read_rgb_image,
    state_xyz_quat_gripper,
)


DEFAULT_MANIFEST = Path(
    "/shareNFS_40/wyf/workplace/openpi/examples/real_preprocess/manifests/insert_usb_real_gelsight_delta_eef_v1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Insert USB real manifest to LeRobot.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-writer-processes", type=int, default=5)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_features(image_shapes: dict[str, tuple[int, int, int]]) -> dict:
    features = {}
    for feature_name, shape in image_shapes.items():
        features[feature_name] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }
    features["state"] = {"dtype": "float32", "shape": (8,), "names": ["state"]}
    features["actions"] = {"dtype": "float32", "shape": (7,), "names": ["actions"]}
    return features


def validate_segment(segment, num_frames: int) -> np.ndarray:
    if segment.start_frame < 0 or segment.end_frame < 0:
        raise ValueError(f"{segment.raw_episode}/{segment.new_episode_id}: negative frame bounds")
    if segment.start_frame > segment.end_frame:
        raise ValueError(f"{segment.raw_episode}/{segment.new_episode_id}: start_frame > end_frame")
    if segment.end_frame >= num_frames:
        raise ValueError(
            f"{segment.raw_episode}/{segment.new_episode_id}: end_frame {segment.end_frame} >= num_frames {num_frames}"
        )
    if segment.frame_stride <= 0:
        raise ValueError(f"{segment.raw_episode}/{segment.new_episode_id}: frame_stride must be positive")
    indices = segment.raw_indices
    if len(indices) < 2:
        raise ValueError(f"{segment.raw_episode}/{segment.new_episode_id}: segment selects fewer than 2 frames")
    return indices


def convert_segment(
    dataset: LeRobotDataset,
    raw_dir: Path,
    segment,
    task_prompt: str,
    target_offset: int,
    image_size: tuple[int, int],
) -> int:
    ep_dir = raw_dir / segment.raw_episode
    episode = inspect_episode(ep_dir, image_size)
    indices = validate_segment(segment, episode["num_frames"])

    states_all = state_xyz_quat_gripper(episode["states"])
    states = states_all[indices]
    actions = delta_eef_actions(episode["states"], indices, target_offset)
    image_paths = episode["image_paths"]

    for out_idx, raw_idx in enumerate(indices):
        frame = {
            "state": states[out_idx],
            "actions": actions[out_idx],
            "task": task_prompt,
        }
        for feature_name in IMAGE_STREAMS:
            frame[feature_name] = resize_rgb_image(read_rgb_image(image_paths[feature_name][int(raw_idx)]), image_size)
        dataset.add_frame(frame)

    dataset.save_episode()
    return len(indices)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = read_yaml(manifest_path)
    dataset_cfg = manifest["dataset"]
    if dataset_cfg.get("control_mode") != "delta_eef":
        raise ValueError(f"Only control_mode=delta_eef is supported, got {dataset_cfg.get('control_mode')!r}")

    raw_dir = Path(dataset_cfg["raw_dir"]).expanduser().resolve()
    image_size = tuple(int(v) for v in dataset_cfg.get("image_size", [224, 224]))
    task_prompt = str(dataset_cfg.get("task_prompt", "pick and insert the USB"))
    target_offset = int(dataset_cfg.get("target_offset", 1))
    fps = int(dataset_cfg.get("target_fps", 15))
    action_horizon = int(dataset_cfg.get("action_horizon", 50))
    segments = parse_manifest_segments(manifest)

    output_root = args.output_root.expanduser().resolve()
    dataset_root = output_root / repo_id_to_path(args.repo_id)
    first_episode = inspect_episode(raw_dir / segments[0].raw_episode, image_size)
    features = build_features(first_episode["image_shapes"])

    print(f"Manifest: {manifest_path}")
    print(f"Raw dir: {raw_dir}")
    print(f"Segments: {len(segments)}")
    print(f"Repo id: {args.repo_id}")
    print(f"Output dataset root: {dataset_root}")
    print(f"Task prompt: {task_prompt!r}")
    print(f"FPS: {fps}")
    print(f"Action horizon metadata: {action_horizon}")
    print("State: [eef_xyz, eef_quat_xyzw, gripper_q] shape [8]")
    print("Actions: [delta_xyz, delta_rotvec, target_gripper_q] shape [7]")

    if args.dry_run:
        total = 0
        for segment in segments:
            episode = inspect_episode(raw_dir / segment.raw_episode, image_size)
            total += len(validate_segment(segment, episode["num_frames"]))
        print(f"Dry run only; planned frames: {total}")
        return

    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {dataset_root}. Pass --overwrite to replace it.")
        shutil.rmtree(dataset_root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dataset_root,
        robot_type="franka",
        fps=fps,
        features=features,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    total_frames = 0
    for segment in progress(segments, "Converting segments"):
        total_frames += convert_segment(
            dataset=dataset,
            raw_dir=raw_dir,
            segment=segment,
            task_prompt=task_prompt,
            target_offset=target_offset,
            image_size=image_size,
        )

    print(f"Done. Wrote {len(segments)} episodes and {total_frames} frames to {dataset_root}")


if __name__ == "__main__":
    main()
