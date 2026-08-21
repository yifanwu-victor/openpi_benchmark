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
EPISODE_IDS="${EPISODE_IDS:-}"
FRAME_STRIDE="${FRAME_STRIDE:-}"
TARGET_OFFSET="${TARGET_OFFSET:-}"
TASK_PROMPT="${TASK_PROMPT:-}"
IMAGE_WRITER_PROCESSES="${IMAGE_WRITER_PROCESSES:-}"
IMAGE_WRITER_THREADS="${IMAGE_WRITER_THREADS:-}"
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
if [[ -n "$EPISODE_IDS" ]]; then
  read -r -a episode_ids <<< "$EPISODE_IDS"
  cmd+=(--episode-ids "${episode_ids[@]}")
fi
if [[ -n "$FRAME_STRIDE" ]]; then
  cmd+=(--frame-stride "$FRAME_STRIDE")
fi
if [[ -n "$TARGET_OFFSET" ]]; then
  cmd+=(--target-offset "$TARGET_OFFSET")
fi
if [[ -n "$TASK_PROMPT" ]]; then
  cmd+=(--task-prompt "$TASK_PROMPT")
fi
if [[ -n "$IMAGE_WRITER_PROCESSES" ]]; then
  cmd+=(--image-writer-processes "$IMAGE_WRITER_PROCESSES")
fi
if [[ -n "$IMAGE_WRITER_THREADS" ]]; then
  cmd+=(--image-writer-threads "$IMAGE_WRITER_THREADS")
fi

printf '[data-convert] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
