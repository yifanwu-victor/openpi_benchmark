#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
PI_TRAINING_DATA_ROOT="${PI_TRAINING_DATA_ROOT:-/run/ti/pi-training-data}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI_TRAINING_DATA_ROOT}/lerobot}"

HDF5_DIR="${HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/turn_gear_pair}"
REPO_ID="${REPO_ID:-simulation/turn_gear_pair_gelsight_abs_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"

CONFIG_NAME="${CONFIG_NAME:-pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft}"
EXP_NAME="${EXP_NAME:-turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
SKIP_CONVERT="${SKIP_CONVERT:-1}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

if [[ "$SKIP_CONVERT" != "1" ]]; then
  HDF5_DIR="$HDF5_DIR" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  REPO_ID="$REPO_ID" \
  CONTROL_MODE=abs_joint \
  GRIPPER_MODE=left \
  OVERWRITE=1 \
  bash scripts/data_convert/convert_turn_gear_pair_gelsight_sim_abs_joint.sh
fi

if [[ "$FORCE_NORM_STATS" == "1" || ! -d "assets/${CONFIG_NAME}/${REPO_ID}" ]]; then
  CONFIG_NAME="$CONFIG_NAME" bash scripts/data_convert/compute_norm_stats.sh
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[train] dry run complete"
  exit 0
fi

train_args=(
  "$CONFIG_NAME"
  --exp-name "$EXP_NAME"
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
  --num-train-steps "$NUM_TRAIN_STEPS"
  --batch-size "$BATCH_SIZE"
  --save-interval "$SAVE_INTERVAL"
  --keep-period "$KEEP_PERIOD"
  --fsdp-devices "$FSDP_DEVICES"
)

if [[ "$RESUME" == "1" ]]; then
  train_args+=(--resume)
else
  train_args+=(--overwrite)
fi

printf '[train] command:'
printf ' %q' uv run scripts/train.py "${train_args[@]}"
printf '\n'
exec uv run scripts/train.py "${train_args[@]}"
