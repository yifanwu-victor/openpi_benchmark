#!/usr/bin/env python3
"""
Convert UniVTAC sim HDF5 episodes to a LeRobot dataset for AnyTouch2 tactile clips.

This converter intentionally lives apart from `examples/UniVTAC/convert_sim_to_lerobot.py`.
It preserves the standard OpenPI fields and adds clip fields consumed by the
AnyTouch2 tactile encoder:

    image
    wrist_image
    left_tactile_clip   [4, 224, 224, 3]
    right_tactile_clip  [4, 224, 224, 3]
    state
    actions
    task / prompt

Clip rule for current raw frame t:

    [t - 6, t - 4, t - 2, t]

Frame indices before zero are clamped to zero, so the first frames repeat the
episode's first tactile image.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

BASE_CONVERTER = Path(__file__).resolve().parents[1] / "UniVTAC" / "convert_sim_to_lerobot.py"
spec = importlib.util.spec_from_file_location("univtac_convert_sim_to_lerobot", BASE_CONVERTER)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load base converter from {BASE_CONVERTER}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


DEFAULT_REPO_ID = "simulation/insert_usb_v3_gelsight_abs_joint_anytouch2_clip"
DEFAULT_OUTPUT_DIR = Path("/shareNFS_40/wyf/data/lerobot/simulation/insert_usb_v3_gelsight_abs_joint_anytouch2_clip")
DEFAULT_CLIP_OFFSETS = (-6, -4, -2, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert UniVTAC sim HDF5 to LeRobot with AnyTouch2 tactile clips.")
    parser.add_argument("--hdf5-dir", type=Path, required=True, help="Directory containing .hdf5 episodes.")
    parser.add_argument("--output-dir", type=Path, required=True, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--control-mode",
        choices=("abs_joint", "delta_eef"),
        default="abs_joint",
        help=(
            "Representation to store. For relative-joint training, use the default abs_joint: "
            "state/action are [joint7, gripper], and the OpenPI config converts action to relative joint + abs gripper."
        ),
    )
    parser.add_argument("--task-prompt", default=None, help="Override HDF5 instruction attr.")
    parser.add_argument("--episode-ids", nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--target-offset", type=int, default=1)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--image-size", nargs=2, type=int, default=base.DEFAULT_IMAGE_SIZE, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--gripper-mode", choices=("left", "right", "mean"), default="left")
    parser.add_argument("--clip-offsets", nargs=4, type=int, default=DEFAULT_CLIP_OFFSETS)
    parser.add_argument("--image-writer-processes", type=int, default=5)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--read-retries", type=int, default=3)
    parser.add_argument("--read-retry-delay", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_features(image_shapes: dict[str, tuple[int, int, int]], clip_frames: int, control_mode: str) -> dict:
    features = {
        "image": {"dtype": "image", "shape": image_shapes["image"], "names": ["height", "width", "channel"]},
        "wrist_image": {
            "dtype": "image",
            "shape": image_shapes["wrist_image"],
            "names": ["height", "width", "channel"],
        },
        "left_tactile_clip": {
            "dtype": "uint8",
            "shape": (clip_frames, *image_shapes["left_tactile_image"]),
            "names": ["frame", "height", "width", "channel"],
        },
        "right_tactile_clip": {
            "dtype": "uint8",
            "shape": (clip_frames, *image_shapes["right_tactile_image"]),
            "names": ["frame", "height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (base.CONTROL_MODE_DIMS[control_mode][0],), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (base.CONTROL_MODE_DIMS[control_mode][1],), "names": ["actions"]},
    }
    return features


def decode_resized(h5_file: h5py.File, h5_key: str, raw_idx: int, image_size: tuple[int, int], args) -> np.ndarray:
    return base.resize_image(
        base.decode_sim_rgb_jpeg(
            h5_file[h5_key],
            int(raw_idx),
            read_retries=args.read_retries,
            read_retry_delay=args.read_retry_delay,
        ),
        image_size,
    )


def make_tactile_clip(
    h5_file: h5py.File,
    h5_key: str,
    raw_idx: int,
    image_size: tuple[int, int],
    clip_offsets: tuple[int, ...],
    args,
) -> np.ndarray:
    frames = []
    for offset in clip_offsets:
        clip_idx = max(0, int(raw_idx) + int(offset))
        frames.append(decode_resized(h5_file, h5_key, clip_idx, image_size, args))
    return np.stack(frames, axis=0).astype(np.uint8)


def convert_episode(dataset: LeRobotDataset, hdf5_path: Path, args) -> int:
    image_size = tuple(args.image_size)
    with h5py.File(hdf5_path, "r") as h5_file:
        num_frames = base.validate_hdf5_episode(h5_file, hdf5_path, args.control_mode)
        episode_prompt = base.read_episode_prompt(h5_file, args.task_prompt, hdf5_path)
        indices = base.selected_indices(num_frames, args.frame_stride)
        states, actions = base.make_states_and_actions_for_mode(
            h5_file,
            args.control_mode,
            indices,
            args.target_offset,
            args.gripper_mode,
        )

        for out_idx, raw_idx in enumerate(indices):
            frame = {
                "image": decode_resized(h5_file, base.IMAGE_STREAMS["image"], int(raw_idx), image_size, args),
                "wrist_image": decode_resized(h5_file, base.IMAGE_STREAMS["wrist_image"], int(raw_idx), image_size, args),
                "left_tactile_clip": make_tactile_clip(
                    h5_file,
                    base.IMAGE_STREAMS["left_tactile_image"],
                    int(raw_idx),
                    image_size,
                    tuple(args.clip_offsets),
                    args,
                ),
                "right_tactile_clip": make_tactile_clip(
                    h5_file,
                    base.IMAGE_STREAMS["right_tactile_image"],
                    int(raw_idx),
                    image_size,
                    tuple(args.clip_offsets),
                    args,
                ),
                "state": states[out_idx],
                "actions": actions[out_idx],
                "task": episode_prompt,
            }
            dataset.add_frame(frame)

    dataset.save_episode()
    return len(indices)


def main() -> None:
    args = parse_args()
    hdf5_dir = args.hdf5_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    image_size = tuple(args.image_size)

    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"Missing HDF5 directory: {hdf5_dir}")

    episodes = base.sorted_hdf5_paths(hdf5_dir, args.episode_ids)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    first_episode = base.inspect_episode(episodes[0], image_size, args.control_mode, args.task_prompt)
    fps = args.fps or base.infer_fps(first_episode["step"], args.frame_stride, base.DEFAULT_FPS)
    features = build_features(first_episode["image_shapes"], clip_frames=len(args.clip_offsets), control_mode=args.control_mode)

    print(f"HDF5 dir: {hdf5_dir}")
    print(f"Episodes: {len(episodes)}")
    print(f"Repo id: {args.repo_id}")
    print(f"Output dataset root: {output_dir}")
    print(f"Task prompt source: {'override' if args.task_prompt else base.PROMPT_ATTR!r}")
    print(f"Control mode: {args.control_mode}")
    if args.control_mode == "abs_joint":
        print("State semantics: [joint7, gripper]")
        print("Action semantics: [target joint7, target gripper]")
        print("Training config may convert actions to [target joint7 - current joint7, abs gripper].")
    else:
        print("State semantics: [eef xyz, eef quat xyzw, gripper]")
        print("Action semantics: [delta xyz, world-frame delta rotvec, target gripper]")
    print(f"Clip offsets: {tuple(args.clip_offsets)}")
    print(f"Image size: {image_size[0]}x{image_size[1]} (width x height)")
    print("Features:")
    for key, value in features.items():
        print(f"  {key}: dtype={value['dtype']}, shape={value['shape']}")

    if args.dry_run:
        print("Dry run only; no LeRobot dataset was written.")
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_dir,
        robot_type="franka",
        fps=fps,
        features=features,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    total_frames = 0
    for hdf5_path in base.progress(episodes, desc="Converting AnyTouch2 episodes"):
        total_frames += convert_episode(dataset, hdf5_path, args)

    print(f"Done. Wrote {len(episodes)} episodes and {total_frames} frames to {output_dir}")


if __name__ == "__main__":
    main()
