#!/usr/bin/env python3
"""Remove huge tactile clip entries from LeRobot episode stats metadata.

LeRobot computes per-episode stats for non-image array features. A tactile clip
field shaped [4, 224, 224, 3] can make `meta/episodes_stats.jsonl` several GB,
and dataset initialization becomes very slow because LeRobot eagerly loads it.

OpenPI normalization only needs state/actions stats, so removing clip stats is
safe for this AnyTouch2 path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_KEYS = ("left_tactile_clip", "right_tactile_clip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune tactile clip stats from LeRobot episodes_stats.jsonl.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", nargs="*", default=DEFAULT_KEYS)
    parser.add_argument("--backup", action="store_true", help="Keep a .bak copy of the original stats file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats_path = args.dataset_root / "meta" / "episodes_stats.jsonl"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing episodes stats file: {stats_path}")

    tmp_path = stats_path.with_suffix(".jsonl.tmp")
    backup_path = stats_path.with_suffix(".jsonl.bak")
    removed = 0
    lines = 0

    with stats_path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            item = json.loads(line)
            stats = item.get("stats", {})
            for key in args.keys:
                if key in stats:
                    del stats[key]
                    removed += 1
            dst.write(json.dumps(item, separators=(",", ":")) + "\n")
            lines += 1

    if args.backup:
        stats_path.replace(backup_path)
    else:
        stats_path.unlink()
    tmp_path.replace(stats_path)

    print(f"Pruned {removed} stats entries from {lines} episodes: {stats_path}")
    if args.backup:
        print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()
