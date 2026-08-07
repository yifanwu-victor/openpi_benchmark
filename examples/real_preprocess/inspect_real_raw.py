#!/usr/bin/env python3
"""Inspect Insert USB real raw episodes and write QC tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from real_preprocess_utils import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_MATCH_DT_S,
    DEFAULT_RAW_DIR,
    TIMESTAMP_KEYS,
    estimate_fps_from_timestamps,
    inspect_episode,
    progress,
    select_episode_dirs,
)


DEFAULT_OUTPUT_DIR = Path("/shareNFS_40/wyf/workplace/openpi/outputs/insert_usb_real_qc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Insert USB real raw episodes.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-ids", nargs="*", default=None)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--image-size", nargs=2, type=int, default=DEFAULT_IMAGE_SIZE, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--max-match-dt-s", type=float, default=DEFAULT_MAX_MATCH_DT_S)
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


def gripper_change_groups(gripper_q: np.ndarray) -> list[tuple[int, int]]:
    changes = np.flatnonzero(np.abs(np.diff(gripper_q, prepend=gripper_q[0])) > 1e-6)
    if len(changes) == 0:
        return []
    groups = []
    start = prev = int(changes[0])
    for idx in changes[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx
    groups.append((start, prev))
    return groups


def timestamp_stats(states: dict[str, np.ndarray], key: str, sample_timestamps: np.ndarray) -> dict[str, float | int]:
    if key not in states:
        return {"max": float("nan"), "p95": float("nan"), "count_gt_threshold": -1}
    timestamps = np.asarray(states[key], dtype=np.float64)
    abs_dt = np.abs(timestamps - sample_timestamps)
    return {
        "median": float(np.median(abs_dt)),
        "p95": float(np.percentile(abs_dt, 95)),
        "max": float(np.max(abs_dt)),
    }


def inspect_to_row(ep_dir: Path, image_size: tuple[int, int], args: argparse.Namespace) -> dict:
    episode = inspect_episode(ep_dir, image_size)
    states = episode["states"]
    num_frames = int(episode["num_frames"])

    sample_timestamps = np.asarray(states["sample_timestamps"], dtype=np.float64)
    eef_pos = np.asarray(states["eef_pos"], dtype=np.float32)
    gripper_q = np.asarray(states["gripper_q"], dtype=np.float32).reshape(-1)
    eef_delta = np.r_[0.0, np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)]
    gripper_groups = gripper_change_groups(gripper_q)
    leading_idle = leading_idle_frames(eef_delta, args.start_idle_dx_threshold)
    trailing_idle = trailing_idle_frames(eef_delta, args.end_idle_dx_threshold)

    row = {
        "episode_id": ep_dir.name,
        "num_frames": num_frames,
        "state_shape_warnings": ";".join(episode["state_shape_warnings"]),
        "duration_s": float(sample_timestamps[-1] - sample_timestamps[0]),
        "sample_fps_est": estimate_fps_from_timestamps(sample_timestamps),
        "main_fps_est": estimate_fps_from_timestamps(np.asarray(states["main_timestamps"])),
        "wrist_fps_est": estimate_fps_from_timestamps(np.asarray(states["wrist_timestamps"])),
        "gripper_min": float(np.min(gripper_q)),
        "gripper_max": float(np.max(gripper_q)),
        "gripper_start": float(gripper_q[0]),
        "gripper_end": float(gripper_q[-1]),
        "gripper_change_count": len(gripper_groups),
        "gripper_change_groups": ";".join(f"{start}-{end}" for start, end in gripper_groups),
        "eef_delta_median": float(np.median(eef_delta[1:])),
        "eef_delta_max": float(np.max(eef_delta)),
        "leading_idle_frames": int(leading_idle),
        "trailing_idle_frames": int(trailing_idle),
        "suggest_trim_start": int(max(0, leading_idle - 1)) if leading_idle >= args.idle_min_frames else 0,
        "suggest_trim_end": int(num_frames - trailing_idle) if trailing_idle >= args.idle_min_frames else num_frames - 1,
    }

    has_alignment_issue = False
    for key in TIMESTAMP_KEYS:
        stats = timestamp_stats(states, key, sample_timestamps)
        row[f"{key}_abs_dt_median"] = stats["median"]
        row[f"{key}_abs_dt_p95"] = stats["p95"]
        row[f"{key}_abs_dt_max"] = stats["max"]
        if stats["max"] > args.max_match_dt_s:
            has_alignment_issue = True

    row["needs_review"] = bool(
        has_alignment_issue
        or row["gripper_change_count"] < 2
        or row["gripper_change_count"] > 3
        or row["suggest_trim_start"] > 0
        or row["suggest_trim_end"] < num_frames - 1
        or bool(row["state_shape_warnings"])
    )
    return row


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    image_size = tuple(args.image_size)
    episodes = select_episode_dirs(raw_dir, args.episode_ids, args.episode_start, args.episode_end, args.max_episodes)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [inspect_to_row(ep_dir, image_size, args) for ep_dir in progress(episodes, "Inspecting episodes")]

    csv_path = output_dir / "episode_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "raw_dir": str(raw_dir),
        "episode_count": len(rows),
        "frame_count": int(sum(row["num_frames"] for row in rows)),
        "needs_review_count": int(sum(bool(row["needs_review"]) for row in rows)),
        "sample_fps_median": float(np.median([row["sample_fps_est"] for row in rows])),
        "main_fps_median": float(np.median([row["main_fps_est"] for row in rows])),
        "wrist_fps_median": float(np.median([row["wrist_fps_est"] for row in rows])),
        "max_timestamp_abs_dt_s": float(max(max(row[f"{key}_abs_dt_max"] for key in TIMESTAMP_KEYS) for row in rows)),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote metrics: {csv_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
