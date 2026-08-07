#!/usr/bin/env python3
"""Generate a conservative preprocessing manifest for Insert USB real raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from real_preprocess_utils import (
    DEFAULT_ACTION_HORIZON,
    DEFAULT_FPS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RAW_DIR,
    DEFAULT_TARGET_OFFSET,
    DEFAULT_TASK_PROMPT,
    inspect_episode,
    progress,
    select_episode_dirs,
    write_yaml,
)


DEFAULT_MANIFEST = Path(
    "/shareNFS_40/wyf/workplace/openpi/examples/real_preprocess/manifests/insert_usb_real_gelsight_delta_eef_v1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Insert USB real preprocessing manifest.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--episode-ids", nargs="*", default=None)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--task-prompt", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--target-fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--target-offset", type=int, default=DEFAULT_TARGET_OFFSET)
    parser.add_argument("--image-size", nargs=2, type=int, default=DEFAULT_IMAGE_SIZE, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--trim-start-idle", action="store_true", help="Use detected leading idle as start_frame.")
    parser.add_argument("--trim-end-idle", action="store_true", help="Use detected trailing idle as end_frame.")
    parser.add_argument("--drop-episodes-file", type=Path, default=None, help="Text file with one raw episode id to drop per line.")
    parser.add_argument("--drop-episode-ids", nargs="*", default=None, help="Raw episode ids to mark as drop.")
    parser.add_argument("--drop-reason", default="drop gripper abnormal episode")
    parser.add_argument("--start-idle-dx-threshold", type=float, default=2e-4)
    parser.add_argument("--end-idle-dx-threshold", type=float, default=1e-4)
    parser.add_argument("--idle-min-frames", type=int, default=8)
    return parser.parse_args()


def leading_idle_frames(dx: np.ndarray, threshold: float) -> int:
    count = 0
    for value in dx:
        if value < threshold:
            count += 1
        else:
            break
    return count


def trailing_idle_frames(dx: np.ndarray, threshold: float) -> int:
    count = 0
    for value in dx[::-1]:
        if value < threshold:
            count += 1
        else:
            break
    return count


def read_drop_episode_ids(args: argparse.Namespace) -> set[str]:
    drop_episode_ids = set(args.drop_episode_ids or [])
    if args.drop_episodes_file is not None:
        path = args.drop_episodes_file.expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines():
            episode_id = line.split("#", 1)[0].strip()
            if episode_id:
                drop_episode_ids.add(episode_id)
    return drop_episode_ids


def make_episode_entry(ep_dir: Path, image_size: tuple[int, int], args: argparse.Namespace, drop_episode_ids: set[str]) -> dict:
    episode = inspect_episode(ep_dir, image_size)
    states = episode["states"]
    num_frames = int(episode["num_frames"])

    if ep_dir.name in drop_episode_ids:
        return {
            "raw_episode": ep_dir.name,
            "status": "drop",
            "num_raw_frames": num_frames,
            "reason": args.drop_reason,
            "segments": [],
        }

    eef_pos = np.asarray(states["eef_pos"], dtype=np.float32)
    eef_delta = np.r_[0.0, np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)]

    leading_idle = leading_idle_frames(eef_delta, args.start_idle_dx_threshold)
    trailing_idle = trailing_idle_frames(eef_delta, args.end_idle_dx_threshold)
    start_frame = int(max(0, leading_idle - 1)) if args.trim_start_idle and leading_idle >= args.idle_min_frames else 0
    end_frame = int(num_frames - trailing_idle) if args.trim_end_idle and trailing_idle >= args.idle_min_frames else num_frames - 1
    if end_frame - start_frame + 1 < 2:
        start_frame = 0
        end_frame = num_frames - 1

    reasons = []
    if start_frame > 0:
        reasons.append(f"trim leading idle frames 0-{start_frame - 1}")
    if end_frame < num_frames - 1:
        reasons.append(f"trim terminal idle frames {end_frame + 1}-{num_frames - 1}")
    if not reasons:
        reasons.append("keep full episode")

    return {
        "raw_episode": ep_dir.name,
        "status": "keep",
        "num_raw_frames": num_frames,
        "segments": [
            {
                "new_episode_id": ep_dir.name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_stride": args.frame_stride,
                "reason": "; ".join(reasons),
            }
        ],
    }


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    image_size = tuple(args.image_size)
    episodes = select_episode_dirs(raw_dir, args.episode_ids, args.episode_start, args.episode_end, args.max_episodes)
    drop_episode_ids = read_drop_episode_ids(args)
    entries = [make_episode_entry(ep_dir, image_size, args, drop_episode_ids) for ep_dir in progress(episodes, "Building manifest")]
    dropped_count = sum(entry["status"] == "drop" for entry in entries)

    manifest = {
        "dataset": {
            "raw_dir": str(raw_dir),
            "task_prompt": args.task_prompt,
            "control_mode": "delta_eef",
            "state_semantics": "[eef_xyz, eef_quat_xyzw, gripper_q]",
            "action_semantics": "[delta_xyz, delta_rotvec, target_gripper_q]",
            "target_fps": args.target_fps,
            "action_horizon": args.action_horizon,
            "frame_stride": args.frame_stride,
            "target_offset": args.target_offset,
            "image_size": list(image_size),
            "dropped_episode_count": dropped_count,
            "notes": "Raw data is not modified. Edit segments manually before conversion if needed.",
        },
        "episodes": entries,
    }
    write_yaml(output, manifest)
    print(f"Wrote manifest: {output}")
    print(f"Episodes: {len(manifest['episodes'])}")
    print(f"Keep: {len(entries) - dropped_count}")
    print(f"Drop: {dropped_count}")


if __name__ == "__main__":
    main()
