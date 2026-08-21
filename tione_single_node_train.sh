#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh}"

cd "$REPO_ROOT"
exec bash "$TRAIN_SCRIPT"
