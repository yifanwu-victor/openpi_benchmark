#!/usr/bin/env python3
"""Check Insert USB real episodes for suspicious gripper open/close patterns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from real_preprocess_utils import DEFAULT_RAW_DIR, load_states, progress, select_episode_dirs


DEFAULT_OUTPUT_DIR = Path("/shareNFS_40/wyf/workplace/openpi/outputs/insert_usb_real_gripper_qc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check gripper open-close-open patterns in raw Insert USB episodes.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-ids", nargs="*", default=None)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--min-range", type=float, default=0.05, help="Minimum gripper_q range required to count as motion.")
    parser.add_argument("--open-ratio", type=float, default=0.7, help="Open threshold: min + ratio * range.")
    parser.add_argument("--closed-ratio", type=float, default=0.3, help="Closed threshold: min + ratio * range.")
    return parser.parse_args()


def transition_groups(values: np.ndarray, atol: float = 1e-6) -> list[tuple[int, int]]:
    changes = np.flatnonzero(np.abs(np.diff(values, prepend=values[0])) > atol)
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


def classify_gripper(values: np.ndarray, open_threshold: float, closed_threshold: float) -> list[str]:
    labels = []
    for value in values:
        if value >= open_threshold:
            labels.append("open")
        elif value <= closed_threshold:
            labels.append("closed")
        else:
            labels.append("transition")
    return labels


def stable_sequence(labels: list[str]) -> list[str]:
    sequence = []
    for label in labels:
        if label == "transition":
            continue
        if not sequence or sequence[-1] != label:
            sequence.append(label)
    return sequence


def first_index(labels: list[str], target: str, start: int = 0) -> int:
    for idx in range(start, len(labels)):
        if labels[idx] == target:
            return idx
    return -1


def inspect_episode_gripper(ep_dir: Path, args: argparse.Namespace) -> dict:
    states = load_states(ep_dir)
    gripper_q = np.asarray(states["gripper_q"], dtype=np.float64).reshape(-1)
    if gripper_q.size < 2:
        raise ValueError(f"{ep_dir}: expected at least 2 gripper frames, got {gripper_q.size}")

    gripper_min = float(np.min(gripper_q))
    gripper_max = float(np.max(gripper_q))
    gripper_range = gripper_max - gripper_min
    open_threshold = gripper_min + args.open_ratio * gripper_range
    closed_threshold = gripper_min + args.closed_ratio * gripper_range
    groups = transition_groups(gripper_q)

    reasons = []
    labels: list[str] = []
    sequence: list[str] = []
    first_closed = -1
    first_reopen = -1

    if gripper_range < args.min_range:
        reasons.append("no_gripper_motion")
        final_label = "unknown"
    else:
        labels = classify_gripper(gripper_q, open_threshold, closed_threshold)
        sequence = stable_sequence(labels)
        final_label = labels[-1]
        first_closed = first_index(labels, "closed")
        first_reopen = first_index(labels, "open", first_closed + 1) if first_closed >= 0 else -1

        if labels[0] != "open":
            reasons.append("does_not_start_open")
        if first_closed < 0:
            reasons.append("no_closed_phase")
        if first_closed >= 0 and first_reopen < 0:
            reasons.append("no_reopen_after_close")
        if final_label != "open":
            reasons.append("does_not_end_open")
        if sequence != ["open", "closed", "open"]:
            reasons.append(f"unexpected_stable_sequence:{'->'.join(sequence)}")
        if len(groups) != 2:
            reasons.append(f"transition_group_count:{len(groups)}")

    return {
        "episode_id": ep_dir.name,
        "num_frames": int(gripper_q.size),
        "status": "review" if reasons else "ok",
        "reasons": ";".join(reasons),
        "gripper_start": float(gripper_q[0]),
        "gripper_end": float(gripper_q[-1]),
        "gripper_min": gripper_min,
        "gripper_max": gripper_max,
        "gripper_range": gripper_range,
        "closed_threshold": float(closed_threshold),
        "open_threshold": float(open_threshold),
        "transition_group_count": len(groups),
        "transition_groups": ";".join(f"{start}-{end}" for start, end in groups),
        "stable_sequence": "->".join(sequence),
        "first_closed_frame": first_closed,
        "first_reopen_frame": first_reopen,
    }


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    episodes = select_episode_dirs(raw_dir, args.episode_ids, args.episode_start, args.episode_end, args.max_episodes)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [inspect_episode_gripper(ep_dir, args) for ep_dir in progress(episodes, "Checking gripper")]
    review_rows = [row for row in rows if row["status"] != "ok"]

    csv_path = output_dir / "gripper_qc.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    review_path = output_dir / "review_episodes.txt"
    review_path.write_text("\n".join(row["episode_id"] for row in review_rows) + ("\n" if review_rows else ""), encoding="utf-8")

    summary = {
        "raw_dir": str(raw_dir),
        "episode_count": len(rows),
        "ok_count": len(rows) - len(review_rows),
        "review_count": len(review_rows),
        "review_episodes": [row["episode_id"] for row in review_rows],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote gripper QC: {csv_path}")
    print(f"Wrote review list: {review_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
