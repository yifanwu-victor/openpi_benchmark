#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
HDF5_DIR="${HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/turn_gear_pair}"
PI_TRAINING_DATA_ROOT="${PI_TRAINING_DATA_ROOT:-/run/ti/pi-training-data}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI_TRAINING_DATA_ROOT}/lerobot}"
REPO_ID="${REPO_ID:-simulation/turn_gear_pair_gelsight_abs_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
CONTROL_MODE="${CONTROL_MODE:-abs_joint}"
GRIPPER_MODE="${GRIPPER_MODE:-left}"
DRY_RUN="${DRY_RUN:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
OVERWRITE="${OVERWRITE:-1}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$OUTPUT_DIR")"

cmd=(
  uv run examples/UniVTAC/convert_sim_to_lerobot.py
  --hdf5-dir "$HDF5_DIR"
  --output-dir "$OUTPUT_DIR"
  --repo-id "$REPO_ID"
  --control-mode "$CONTROL_MODE"
  --gripper-mode "$GRIPPER_MODE"
  --image-size 224 224
)

if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ -n "$MAX_EPISODES" ]]; then
  cmd+=(--max-episodes "$MAX_EPISODES")
fi

printf '[data-convert] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
