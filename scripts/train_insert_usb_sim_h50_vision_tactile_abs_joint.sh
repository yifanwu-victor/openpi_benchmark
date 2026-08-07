#!/usr/bin/env bash
set -euo pipefail

# Train Insert USB sim abs-joint LoRA with four vision+tactile inputs and horizon 50.
#
# Defaults:
#   - dataset: /shareNFS_40/wyf/data/lerobot/wyf/insert_usb_sim_abs_joint
#   - GPU: CUDA_VISIBLE_DEVICES=1
#   - global batch size: 32
#   - train steps: 20000
#
# This config reuses the norm stats from the two-view h50 config because the
# state/action representation is identical; images are not part of norm stats.
#
# Checkpoints are first written to LOCAL_CHECKPOINT_BASE on the home disk.
# After training finishes, this script syncs the run directory to FINAL_CHECKPOINT_BASE,
# verifies the copied files with rsync checksum dry-run, then removes the home copy.
#
# Optional overrides:
#   GPU_ID=2 EXP_NAME=my_exp bash scripts/train_insert_usb_sim_h50_vision_tactile.sh
#   LOCAL_CHECKPOINT_BASE=/home/wyf/openpi_checkpoints bash scripts/train_insert_usb_sim_h50_vision_tactile.sh

REPO_ROOT="${REPO_ROOT:-/shareNFS_40/wyf/workplace/openpi}"
CONFIG_NAME="${CONFIG_NAME:-pi05_insert_usb_sim_abs_joint_vision_tactile_lora_h50}"
STATS_CONFIG_NAME="${STATS_CONFIG_NAME:-pi05_insert_usb_sim_abs_joint_vision_lora_h50}"
EXP_NAME="${EXP_NAME:-insert_usb_sim_abs_joint_vision_tactile_lora_h50_v1}"

GPU_ID="${GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20000}"
KEEP_PERIOD="${KEEP_PERIOD:-20000}"

LOCAL_CHECKPOINT_BASE="${LOCAL_CHECKPOINT_BASE:-/home/wyf/openpi_checkpoints}"
FINAL_CHECKPOINT_BASE="${FINAL_CHECKPOINT_BASE:-/shareNFS_40/wyf/workplace/openpi/checkpoints}"

OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/shareNFS_40/wyf/data/openpi}"
HF_HOME="${HF_HOME:-/shareNFS_40/wyf/data/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/shareNFS_40/wyf/data/hf_cache/datasets}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/shareNFS_40/wyf/data/lerobot}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

NORM_STATS_DIR="${REPO_ROOT}/assets/${STATS_CONFIG_NAME}/wyf/insert_usb_sim_abs_joint"
LOCAL_RUN_DIR="${LOCAL_CHECKPOINT_BASE}/${CONFIG_NAME}/${EXP_NAME}"
FINAL_RUN_DIR="${FINAL_CHECKPOINT_BASE}/${CONFIG_NAME}/${EXP_NAME}"

cd "$REPO_ROOT"
mkdir -p "$LOCAL_CHECKPOINT_BASE" "$FINAL_CHECKPOINT_BASE" "$HF_DATASETS_CACHE"

export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME HF_HOME HF_DATASETS_CACHE HF_LEROBOT_HOME
export XLA_PYTHON_CLIENT_PREALLOCATE XLA_FLAGS

if [[ "${FORCE_NORM_STATS:-0}" == "1" || ! -d "$NORM_STATS_DIR" ]]; then
  echo "[norm-stats] computing stats for ${STATS_CONFIG_NAME}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" uv run scripts/compute_norm_stats.py --config-name "$STATS_CONFIG_NAME"
else
  echo "[norm-stats] found ${NORM_STATS_DIR}; set FORCE_NORM_STATS=1 to recompute"
fi

echo "[train] config=${CONFIG_NAME}"
echo "[train] exp=${EXP_NAME}"
echo "[train] GPU=${GPU_ID}, global_batch_size=${BATCH_SIZE}"
echo "[train] local checkpoint dir=${LOCAL_RUN_DIR}"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
uv run scripts/train.py \
  "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --checkpoint-base-dir "$LOCAL_CHECKPOINT_BASE" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD"

if [[ ! -d "$LOCAL_RUN_DIR" ]]; then
  echo "[error] local run directory was not created: ${LOCAL_RUN_DIR}" >&2
  exit 1
fi

echo "[sync] ${LOCAL_RUN_DIR} -> ${FINAL_RUN_DIR}"
mkdir -p "$(dirname "$FINAL_RUN_DIR")"
rsync -a --delete "${LOCAL_RUN_DIR}/" "${FINAL_RUN_DIR}/"

echo "[verify] checksum dry-run"
VERIFY_OUTPUT="$(rsync -a --delete --checksum --dry-run --itemize-changes "${LOCAL_RUN_DIR}/" "${FINAL_RUN_DIR}/")"
if [[ -n "$VERIFY_OUTPUT" ]]; then
  echo "[error] verification failed; rsync still sees differences:" >&2
  echo "$VERIFY_OUTPUT" >&2
  exit 1
fi

case "$LOCAL_RUN_DIR" in
  "$HOME"/*)
    echo "[cleanup] removing verified home checkpoint copy: ${LOCAL_RUN_DIR}"
    rm -r -- "$LOCAL_RUN_DIR"
    ;;
  *)
    echo "[error] refusing to delete path outside home: ${LOCAL_RUN_DIR}" >&2
    exit 1
    ;;
esac

echo "[done] final checkpoint directory: ${FINAL_RUN_DIR}"
