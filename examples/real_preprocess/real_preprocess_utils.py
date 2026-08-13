from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_RAW_DIR = Path("/shareNFS_40/wyf/data/real/insert_usb_real_gelsight")
DEFAULT_OUTPUT_ROOT = Path("/shareNFS_40/wyf/data/lerobot")
DEFAULT_REPO_ID = "wyf/insert_usb_real_gelsight_delta_eef"
DEFAULT_TASK_PROMPT = "pick and insert the USB"
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_FPS = 15
DEFAULT_ACTION_HORIZON = 50
DEFAULT_TARGET_OFFSET = 1
DEFAULT_MAX_MATCH_DT_S = 0.08

IMAGE_STREAMS = {
    "image": Path("main/color"),
    "wrist_image": Path("wrist/color"),
    "left_tactile_image": Path("tactile/left/rectify"),
    "right_tactile_image": Path("tactile/right/rectify"),
}

TIMESTAMP_KEYS = (
    "main_timestamps",
    "wrist_timestamps",
    "xense_left_timestamps",
    "xense_right_timestamps",
    "robot_timestamps",
)

REQUIRED_STATE_KEYS = (
    "sample_timestamps",
    "q",
    "eef_pos",
    "eef_quat",
    "gripper_q",
)

OPTIONAL_ALIGNED_STATE_KEYS = (
    "action",
    "dq",
    "eef_pose",
)


@dataclass(frozen=True)
class Segment:
    raw_episode: str
    new_episode_id: str
    start_frame: int
    end_frame: int
    frame_stride: int
    reason: str

    @property
    def raw_indices(self) -> np.ndarray:
        return np.arange(self.start_frame, self.end_frame + 1, self.frame_stride, dtype=np.int64)


def progress(items: Iterable, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def repo_id_to_path(repo_id: str) -> Path:
    return Path(*repo_id.split("/"))


def sorted_episode_dirs(raw_dir: Path, episode_ids: list[str] | None = None) -> list[Path]:
    if episode_ids is not None:
        episodes = [raw_dir / episode_id for episode_id in episode_ids]
    else:
        episodes = sorted((path for path in raw_dir.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda p: int(p.name))

    missing = [str(path) for path in episodes if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing episode directories: {missing}")
    if not episodes:
        raise FileNotFoundError(f"No episode directories found in {raw_dir}")
    return episodes


def filter_episode_range(episodes: list[Path], episode_start: int | None, episode_end: int | None) -> list[Path]:
    if episode_start is None and episode_end is None:
        return episodes

    start = 0 if episode_start is None else episode_start
    end = 10**12 if episode_end is None else episode_end
    if start < 0 or end < 0:
        raise ValueError(f"Episode range must be non-negative, got {start}-{end}")
    if start > end:
        raise ValueError(f"--episode-start must be <= --episode-end, got {start}>{end}")

    filtered = [path for path in episodes if start <= int(path.name) <= end]
    if not filtered:
        raise FileNotFoundError(f"No episode directories found in requested range {start}-{end}")
    return filtered


def select_episode_dirs(
    raw_dir: Path,
    episode_ids: list[str] | None,
    episode_start: int | None,
    episode_end: int | None,
    max_episodes: int | None,
) -> list[Path]:
    episodes = sorted_episode_dirs(raw_dir, episode_ids)
    if episode_ids is None:
        episodes = filter_episode_range(episodes, episode_start, episode_end)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    return episodes


def sorted_jpgs(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    paths = sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if not paths:
        raise FileNotFoundError(f"No .jpg images found in {image_dir}")
    return paths


def validate_frame_names(paths: list[Path], expected_len: int, label: str) -> None:
    if len(paths) != expected_len:
        raise ValueError(f"{label}: expected {expected_len} images, found {len(paths)}")
    for idx, path in enumerate(paths):
        expected = f"{idx:06d}"
        if path.stem != expected:
            raise ValueError(f"{label}: expected {expected}.jpg at index {idx}, got {path.name}")


def load_states(ep_dir: Path) -> dict[str, np.ndarray]:
    states_path = ep_dir / "states.npz"
    if not states_path.is_file():
        raise FileNotFoundError(f"Missing states.npz: {states_path}")

    with np.load(states_path) as states:
        missing = [key for key in REQUIRED_STATE_KEYS if key not in states]
        if missing:
            raise KeyError(f"{states_path}: missing required keys {missing}")
        return {key: states[key] for key in states.files}


def load_image_paths(ep_dir: Path, expected_len: int) -> dict[str, list[Path]]:
    image_paths = {}
    for feature_name, rel_dir in IMAGE_STREAMS.items():
        paths = sorted_jpgs(ep_dir / rel_dir)
        validate_frame_names(paths, expected_len, f"{ep_dir.name}/{rel_dir}")
        image_paths[feature_name] = paths
    return image_paths


def read_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_rgb_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {image_size}")
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def image_shape(path: Path, image_size: tuple[int, int]) -> tuple[int, int, int]:
    image = resize_rgb_image(read_rgb_image(path), image_size)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image at {path}, got shape {image.shape}")
    return tuple(image.shape)


def inspect_episode(ep_dir: Path, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> dict:
    states = load_states(ep_dir)
    num_frames = int(np.asarray(states["eef_pos"]).shape[0])
    if num_frames < 2:
        raise ValueError(f"{ep_dir}: expected at least 2 frames, got {num_frames}")

    state_shape_warnings = []
    for key, value in states.items():
        arr = np.asarray(value)
        if key.endswith("_timestamps") or key in REQUIRED_STATE_KEYS or key in OPTIONAL_ALIGNED_STATE_KEYS:
            if arr.shape[0] != num_frames:
                message = f"{key}:{arr.shape[0]}!={num_frames}"
                if key.endswith("_timestamps") or key in REQUIRED_STATE_KEYS:
                    raise ValueError(f"{ep_dir}: state key {key} has first dim {arr.shape[0]}, expected {num_frames}")
                state_shape_warnings.append(message)

    image_paths = load_image_paths(ep_dir, num_frames)
    image_shapes = {feature_name: image_shape(paths[0], image_size) for feature_name, paths in image_paths.items()}

    return {
        "num_frames": num_frames,
        "states": states,
        "image_paths": image_paths,
        "image_shapes": image_shapes,
        "state_shape_warnings": state_shape_warnings,
    }


def estimate_fps_from_timestamps(timestamps: np.ndarray, fallback: int = DEFAULT_FPS) -> int:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.shape[0] < 2:
        return fallback
    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return fallback
    return max(1, int(round(1.0 / float(np.median(deltas)))))


def state_xyz_quat_gripper(states: dict[str, np.ndarray]) -> np.ndarray:
    eef_pos = np.asarray(states["eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(states["eef_quat"], dtype=np.float32)
    gripper_q = np.asarray(states["gripper_q"], dtype=np.float32).reshape(-1, 1)

    if eef_pos.ndim != 2 or eef_pos.shape[1] != 3:
        raise ValueError(f"Expected eef_pos shape [T, 3], got {eef_pos.shape}")
    if eef_quat.ndim != 2 or eef_quat.shape[1] != 4:
        raise ValueError(f"Expected eef_quat shape [T, 4], got {eef_quat.shape}")
    if gripper_q.shape[0] != eef_pos.shape[0]:
        raise ValueError(f"gripper_q length {gripper_q.shape[0]} does not match eef_pos length {eef_pos.shape[0]}")

    norms = np.linalg.norm(eef_quat, axis=1)
    if np.any(norms < 1e-6):
        raise ValueError("eef_quat contains near-zero quaternion")
    eef_quat = eef_quat / norms[:, None]
    return np.concatenate([eef_pos, eef_quat, gripper_q], axis=1).astype(np.float32)


def delta_eef_actions(states: dict[str, np.ndarray], selected_indices: np.ndarray, target_offset: int) -> np.ndarray:
    if target_offset <= 0:
        raise ValueError(f"target_offset must be positive, got {target_offset}")
    eef_pos = np.asarray(states["eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(states["eef_quat"], dtype=np.float32)
    gripper_q = np.asarray(states["gripper_q"], dtype=np.float32).reshape(-1, 1)
    quat_norms = np.linalg.norm(eef_quat, axis=1)
    if np.any(quat_norms < 1e-6):
        raise ValueError("eef_quat contains near-zero quaternion")
    eef_quat = eef_quat / quat_norms[:, None]

    target_indices = np.minimum(selected_indices + target_offset, len(eef_pos) - 1)
    delta_xyz = eef_pos[target_indices] - eef_pos[selected_indices]

    current_rot = Rotation.from_quat(eef_quat[selected_indices])
    target_rot = Rotation.from_quat(eef_quat[target_indices])
    delta_rotvec = (target_rot * current_rot.inv()).as_rotvec().astype(np.float32)
    target_gripper_q = gripper_q[target_indices]

    return np.concatenate([delta_xyz, delta_rotvec, target_gripper_q], axis=1).astype(np.float32)


def raw_delta_eef_actions_with_next_gripper(
    states: dict[str, np.ndarray], selected_indices: np.ndarray, target_offset: int
) -> np.ndarray:
    if target_offset <= 0:
        raise ValueError(f"target_offset must be positive, got {target_offset}")
    if "action" not in states:
        raise KeyError("states.npz is missing raw action; cannot build raw-action delta_eef targets")

    raw_action = np.asarray(states["action"], dtype=np.float32)
    gripper_q = np.asarray(states["gripper_q"], dtype=np.float32).reshape(-1, 1)
    if raw_action.ndim != 2 or raw_action.shape[1] < 6:
        raise ValueError(f"Expected raw action shape [T, >=6], got {raw_action.shape}")
    if raw_action.shape[0] != gripper_q.shape[0]:
        raise ValueError(f"raw action length {raw_action.shape[0]} does not match gripper_q length {gripper_q.shape[0]}")

    target_indices = np.minimum(selected_indices + target_offset, len(gripper_q) - 1)
    delta_eef = raw_action[selected_indices, :6]
    target_gripper_q = gripper_q[target_indices]
    return np.concatenate([delta_eef, target_gripper_q], axis=1).astype(np.float32)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data


def parse_manifest_segments(manifest: dict) -> list[Segment]:
    dataset_cfg = manifest.get("dataset", {})
    default_stride = int(dataset_cfg.get("frame_stride", 1))
    segments: list[Segment] = []
    for episode in manifest.get("episodes", []):
        if episode.get("status", "keep") != "keep":
            continue
        raw_episode = str(episode["raw_episode"])
        for segment in episode.get("segments", []):
            stride = int(segment.get("frame_stride", default_stride))
            segments.append(
                Segment(
                    raw_episode=raw_episode,
                    new_episode_id=str(segment["new_episode_id"]),
                    start_frame=int(segment["start_frame"]),
                    end_frame=int(segment["end_frame"]),
                    frame_stride=stride,
                    reason=str(segment.get("reason", "")),
                )
            )
    if not segments:
        raise ValueError("Manifest contains no keep segments")
    return segments
