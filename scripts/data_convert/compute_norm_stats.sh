#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
CONFIG_NAME="${CONFIG_NAME:-pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft}"
MAX_FRAMES="${MAX_FRAMES:-}"

cd "$REPO_ROOT"

cmd=(uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME")
if [[ -n "$MAX_FRAMES" ]]; then
  cmd+=(--max-frames "$MAX_FRAMES")
fi

printf '[norm-stats] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
