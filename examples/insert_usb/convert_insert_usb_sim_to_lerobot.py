#!/usr/bin/env python3
"""
Convert UniVTAC simulated Insert USB HDF5 episodes to a LeRobot dataset.

Recommended usage from the OpenPI repo root:

    cd /shareNFS_40/wyf/workplace/openpi
    export HF_LEROBOT_HOME=/shareNFS_40/wyf/data/lerobot

    uv run examples/insert_usb/convert_insert_usb_sim_to_lerobot.py \
        --hdf5-dir /shareNFS_40/wyf/data/UniVTAC/sim/Insert_USB_Tactile \
        --repo-id wyf/insert_usb_sim_abs_eef \
        --output-root /shareNFS_40/wyf/data/lerobot \
        --image-size 224 224 \
        --overwrite

Default output:

    /shareNFS_40/wyf/data/lerobot/wyf/insert_usb_sim_abs_eef

This converter keeps the same LeRobot feature names and state/action format as
the real Insert USB converter:

    image                <- observation/head/rgb
    wrist_image          <- observation/wrist/rgb
    left_tactile_image   <- tactile/left_tactile/rgb_marker
    right_tactile_image  <- tactile/right_tactile/rgb_marker

    state   = [eef_pos(3), eef_rot6d(6), gripper_q(1)]      # shape [10]
    actions = [target_eef_pos(3), target_eef_rot6d(6), target_gripper_q(1)]

Important simulation-specific notes:

* HDF5 image datasets are JPEG byte streams. `cv2.imdecode` returns the
  visually correct channel order for this dataset; do not apply BGR->RGB.
* `embodiment/ee` stores quaternions as [qw, qx, qy, qz]. Convert to xyzw
  before calling scipy Rotation.
* `embodiment/joint` is [7 arm joints, left finger, right finger]. The two
  finger values match in the inspected data; this script keeps the right finger.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import shutil

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_HDF5_DIR = Path("/shareNFS_40/wyf/data/UniVTAC/sim/Insert_USB_Tactile")
DEFAULT_OUTPUT_ROOT = Path("/shareNFS_40/wyf/data/lerobot")
DEFAULT_REPO_ID = "wyf/insert_usb_sim_abs_eef"
DEFAULT_TASK_PROMPT = "pick and insert the USB"
DEFAULT_FPS = 60
DEFAULT_IMAGE_SIZE = (224, 224)

IMAGE_STREAMS = {
    "image": "observation/head/rgb",
    "wrist_image": "observation/wrist/rgb",
    "left_tactile_image": "tactile/left_tactile/rgb_marker",
    "right_tactile_image": "tactile/right_tactile/rgb_marker",
}

REQUIRED_DATASETS = (
    "step",
    "embodiment/joint",
    "embodiment/ee",
    *IMAGE_STREAMS.values(),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert UniVTAC simulated Insert USB HDF5 episodes to LeRobot format."
    )
    parser.add_argument(
        "--hdf5-dir",
        type=Path,
        default=DEFAULT_HDF5_DIR,
        help=f"Input directory containing one .hdf5 file per episode. Default: {DEFAULT_HDF5_DIR}",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"LeRobot repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Base LeRobot output directory. The dataset is written to "
            "<output-root>/<repo-id>. Default: /shareNFS_40/wyf/data/lerobot"
        ),
    )
    parser.add_argument(
        "--task-prompt",
        default=DEFAULT_TASK_PROMPT,
        help=f"Task string stored in every episode. Default: {DEFAULT_TASK_PROMPT!r}",
    )
    parser.add_argument(
        "--episode-ids",
        nargs="*",
        default=None,
        help="Optional HDF5 episode ids to convert, e.g. 2 3 4. The .hdf5 suffix is optional.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert only the first N sorted episodes.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth saved frame. Default: 1.",
    )
    parser.add_argument(
        "--target-offset",
        type=int,
        default=1,
        help=(
            "Absolute action target offset in selected frames. "
            "Default 1 means actions[t] = state[t + 1]."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help=(
            "FPS recorded in LeRobot metadata. Default: infer from saved step "
            f"deltas and frame stride, fallback {DEFAULT_FPS}."
        ),
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=DEFAULT_IMAGE_SIZE,
        help=f"Resize all image streams before writing. Default: {DEFAULT_IMAGE_SIZE[0]} {DEFAULT_IMAGE_SIZE[1]}.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=5,
        help="LeRobot image writer processes. Default: 5.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=10,
        help="LeRobot image writer threads. Default: 10.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output dataset directory before conversion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned dataset without writing.",
    )
    return parser.parse_args()


def progress(items: Iterable, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def repo_id_to_path(repo_id: str) -> Path:
    return Path(*repo_id.split("/"))


def sorted_hdf5_paths(hdf5_dir: Path, episode_ids: list[str] | None) -> list[Path]:
    if episode_ids:
        paths = []
        for episode_id in episode_ids:
            episode_path = hdf5_dir / episode_id
            if episode_path.suffix != ".hdf5":
                episode_path = episode_path.with_suffix(".hdf5")
            paths.append(episode_path)
    else:
        paths = sorted(
            hdf5_dir.glob("*.hdf5"),
            key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
        )

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing HDF5 episode files: {missing}")
    if not paths:
        raise FileNotFoundError(f"No .hdf5 episodes found in {hdf5_dir}")
    return paths


def selected_indices(num_frames: int, frame_stride: int) -> np.ndarray:
    if frame_stride <= 0:
        raise ValueError(f"--frame-stride must be positive, got {frame_stride}")
    indices = np.arange(0, num_frames, frame_stride, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError(
            f"Need at least 2 selected frames, got {len(indices)} from {num_frames} frames with stride {frame_stride}"
        )
    return indices


def decode_sim_rgb_jpeg(dataset: h5py.Dataset, frame_index: int) -> np.ndarray:
    """Decode one simulated RGB image.

    The simulated data was encoded with OpenCV from arrays that already looked
    correct in RGB order. Applying BGR->RGB here visibly swaps red and blue.
    """

    buffer = np.frombuffer(dataset[frame_index], dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV failed to decode image frame {frame_index} from {dataset.name}")
    return image


def resize_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {image_size}")
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def validate_hdf5_episode(h5_file: h5py.File, path: Path) -> int:
    missing = [key for key in REQUIRED_DATASETS if key not in h5_file]
    if missing:
        raise KeyError(f"{path}: missing required datasets {missing}")

    num_frames = int(h5_file["step"].shape[0])
    if num_frames < 2:
        raise ValueError(f"{path}: expected at least 2 frames, got {num_frames}")

    for key in REQUIRED_DATASETS:
        length = int(h5_file[key].shape[0])
        if length != num_frames:
            raise ValueError(f"{path}: {key} length {length} does not match step length {num_frames}")

    joint_shape = h5_file["embodiment/joint"].shape
    ee_shape = h5_file["embodiment/ee"].shape
    if len(joint_shape) != 2 or joint_shape[1] < 9:
        raise ValueError(f"{path}: expected embodiment/joint shape [T, >=9], got {joint_shape}")
    if len(ee_shape) != 2 or ee_shape[1] != 7:
        raise ValueError(f"{path}: expected embodiment/ee shape [T, 7], got {ee_shape}")

    return num_frames


def image_shape(h5_file: h5py.File, h5_key: str, image_size: tuple[int, int]) -> tuple[int, int, int]:
    image = resize_image(decode_sim_rgb_jpeg(h5_file[h5_key], 0), image_size)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image at {h5_key}, got shape {image.shape}")
    return tuple(image.shape)


def rot6d_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[1] != 4:
        raise ValueError(f"Expected quat_wxyz shape [T, 4], got {quat_wxyz.shape}")
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    matrices = Rotation.from_quat(quat_xyzw).as_matrix()
    return np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1).astype(np.float32)


def make_abs_eef_state(joint: np.ndarray, ee: np.ndarray) -> np.ndarray:
    joint = np.asarray(joint, dtype=np.float32)
    ee = np.asarray(ee, dtype=np.float32)
    if joint.ndim != 2 or joint.shape[1] < 9:
        raise ValueError(f"Expected joint shape [T, >=9], got {joint.shape}")
    if ee.ndim != 2 or ee.shape[1] != 7:
        raise ValueError(f"Expected ee shape [T, 7], got {ee.shape}")
    if joint.shape[0] != ee.shape[0]:
        raise ValueError(f"joint length {joint.shape[0]} does not match ee length {ee.shape[0]}")

    eef_pos = ee[:, :3]
    eef_quat_wxyz = ee[:, 3:7]
    eef_rot6d = rot6d_from_quat_wxyz(eef_quat_wxyz)
    gripper_q = joint[:, 8:9]
    return np.concatenate([eef_pos, eef_rot6d, gripper_q], axis=1).astype(np.float32)


def make_action_targets(abs_eef_state: np.ndarray, target_offset: int) -> np.ndarray:
    if target_offset <= 0:
        raise ValueError(f"--target-offset must be positive, got {target_offset}")
    indices = np.arange(len(abs_eef_state), dtype=np.int64) + target_offset
    indices = np.minimum(indices, len(abs_eef_state) - 1)
    return abs_eef_state[indices].astype(np.float32)


def inspect_episode(path: Path, image_size: tuple[int, int]) -> dict:
    with h5py.File(path, "r") as h5_file:
        num_frames = validate_hdf5_episode(h5_file, path)
        image_shapes = {
            feature_name: image_shape(h5_file, h5_key, image_size)
            for feature_name, h5_key in IMAGE_STREAMS.items()
        }
        step = h5_file["step"][()]

    return {
        "num_frames": num_frames,
        "image_shapes": image_shapes,
        "step": step,
    }


def infer_fps(step: np.ndarray, frame_stride: int, fallback: int) -> int:
    if len(step) < 2:
        return fallback
    deltas = np.diff(step)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return fallback

    # Base sim dt is 1/120. A saved-step delta of 2 therefore means 60 FPS.
    raw_fps = int(round(120.0 / float(np.median(deltas))))
    return max(1, raw_fps // max(1, frame_stride))


def build_features(image_shapes: dict[str, tuple[int, int, int]]) -> dict:
    features = {}
    for feature_name, shape in image_shapes.items():
        features[feature_name] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }

    features["state"] = {
        "dtype": "float32",
        "shape": (10,),
        "names": ["state"],
    }
    features["actions"] = {
        "dtype": "float32",
        "shape": (10,),
        "names": ["actions"],
    }
    return features


def convert_episode(
    dataset: LeRobotDataset,
    hdf5_path: Path,
    task_prompt: str,
    frame_stride: int,
    target_offset: int,
    image_size: tuple[int, int],
) -> int:
    with h5py.File(hdf5_path, "r") as h5_file:
        num_frames = validate_hdf5_episode(h5_file, hdf5_path)
        indices = selected_indices(num_frames, frame_stride)

        abs_eef_state_all = make_abs_eef_state(
            h5_file["embodiment/joint"][()],
            h5_file["embodiment/ee"][()],
        )
        states = abs_eef_state_all[indices]
        actions = make_action_targets(states, target_offset)

        for out_idx, raw_idx in enumerate(indices):
            frame = {
                "state": states[out_idx],
                "actions": actions[out_idx],
                "task": task_prompt,
            }
            for feature_name, h5_key in IMAGE_STREAMS.items():
                frame[feature_name] = resize_image(decode_sim_rgb_jpeg(h5_file[h5_key], int(raw_idx)), image_size)
            dataset.add_frame(frame)

    dataset.save_episode()
    return len(indices)


def main() -> None:
    args = parse_args()
    hdf5_dir = args.hdf5_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dataset_root = output_root / repo_id_to_path(args.repo_id)
    image_size = tuple(args.image_size)

    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"Missing HDF5 directory: {hdf5_dir}")

    episodes = sorted_hdf5_paths(hdf5_dir, args.episode_ids)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    first_episode = inspect_episode(episodes[0], image_size)
    indices = selected_indices(first_episode["num_frames"], args.frame_stride)
    fps = args.fps or infer_fps(first_episode["step"], args.frame_stride, DEFAULT_FPS)
    features = build_features(first_episode["image_shapes"])

    print(f"HDF5 dir: {hdf5_dir}")
    print(f"Episodes: {len(episodes)}")
    print(f"Repo id: {args.repo_id}")
    print(f"Output dataset root: {dataset_root}")
    print(f"FPS: {fps}")
    print(f"Frame stride: {args.frame_stride}")
    print(f"Selected frames in first episode: {len(indices)} / {first_episode['num_frames']}")
    print(f"Target offset: {args.target_offset} selected frame(s)")
    print(f"Image size: {image_size[0]}x{image_size[1]} (width x height)")
    print("Features:")
    for key, value in features.items():
        print(f"  {key}: dtype={value['dtype']}, shape={value['shape']}")

    if args.dry_run:
        print("Dry run only; no LeRobot dataset was written.")
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
    for hdf5_path in progress(episodes, desc="Converting episodes"):
        total_frames += convert_episode(
            dataset,
            hdf5_path,
            task_prompt=args.task_prompt,
            frame_stride=args.frame_stride,
            target_offset=args.target_offset,
            image_size=image_size,
        )

    print(f"Done. Wrote {len(episodes)} episodes and {total_frames} frames to {dataset_root}")


if __name__ == "__main__":
    main()
