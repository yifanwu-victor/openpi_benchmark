#!/usr/bin/env bash
set -euo pipefail

# Train Insert USB sim delta-EEF JAX full fine-tuning with four vision+tactile inputs and horizon 50.
#
# Defaults:
#   - dataset: /shareNFS_40/wyf/data/lerobot/simulation/insert_usb_sim_delta_eef
#   - GPUs: CUDA_VISIBLE_DEVICES=4,5,6,7
#   - JAX FSDP devices: 4
#   - global batch size: 32
#   - train steps: 20000
#
# This config reuses the norm stats from the two-view delta-EEF h50 config because
# state/action representation is identical; images are not part of norm stats.
#
# Checkpoints are written directly to CHECKPOINT_BASE_DIR on the data disk.
#
# Optional overrides:
#   GPU_IDS=4,5,6,7 EXP_NAME=my_exp bash scripts/train_insert_usb_sim_delta_eef_h50_vision_tactile_full_ft_jax.sh
#   CHECKPOINT_BASE_DIR=/shareNFS_40/wyf/workplace/openpi/checkpoints bash scripts/train_insert_usb_sim_delta_eef_h50_vision_tactile_full_ft_jax.sh

REPO_ROOT="${REPO_ROOT:-/shareNFS_40/wyf/workplace/openpi}"
CONFIG_NAME="${CONFIG_NAME:-pi05_insert_usb_sim_delta_eef_vision_tactile_full_ft_h50_jax}"
STATS_CONFIG_NAME="${STATS_CONFIG_NAME:-pi05_insert_usb_sim_delta_eef_vision_full_ft_h50_jax}"
EXP_NAME="${EXP_NAME:-insert_usb_sim_delta_eef_vision_tactile_full_ft_h50_jax_v1}"

GPU_IDS="${GPU_IDS:-2,3,4,5}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
KEEP_PERIOD="${KEEP_PERIOD:-10000}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/shareNFS_40/wyf/workplace/openpi/checkpoints}"

OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/shareNFS_40/wyf/data/openpi}"
HF_HOME="${HF_HOME:-/shareNFS_40/wyf/data/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/shareNFS_40/wyf/data/hf_cache/datasets}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/shareNFS_40/wyf/data/lerobot}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

NORM_STATS_DIR="${REPO_ROOT}/assets/${STATS_CONFIG_NAME}/simulation/insert_usb_sim_delta_eef"
RUN_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"

cd "$REPO_ROOT"
mkdir -p "$CHECKPOINT_BASE_DIR" "$HF_DATASETS_CACHE"

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OPENPI_DATA_HOME HF_HOME HF_DATASETS_CACHE HF_LEROBOT_HOME
export XLA_PYTHON_CLIENT_PREALLOCATE

if [[ "${XLA_FLAGS:-}" == *"--xla_gpu_autotune_level=0"* ]]; then
  echo "[env] removing XLA_FLAGS=${XLA_FLAGS}; this flag is unstable with the current Blackwell-compatible JAX stack"
  unset XLA_FLAGS
fi

if [[ "${FORCE_NORM_STATS:-0}" == "1" || ! -d "$NORM_STATS_DIR" ]]; then
  echo "[norm-stats] computing stats for ${STATS_CONFIG_NAME}"
  uv run scripts/compute_norm_stats.py --config-name "$STATS_CONFIG_NAME"
else
  echo "[norm-stats] found ${NORM_STATS_DIR}; set FORCE_NORM_STATS=1 to recompute"
fi

echo "[train] config=${CONFIG_NAME}"
echo "[train] exp=${EXP_NAME}"
echo "[train] GPUs=${GPU_IDS}, fsdp_devices=${FSDP_DEVICES}, global_batch_size=${BATCH_SIZE}"
echo "[train] checkpoint dir=${RUN_DIR}"

uv run scripts/train.py \
  "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --overwrite \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD" \
  --fsdp-devices "$FSDP_DEVICES"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[error] checkpoint run directory was not created: ${RUN_DIR}" >&2
  exit 1
fi

echo "[done] checkpoint directory: ${RUN_DIR}"
