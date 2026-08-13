#!/usr/bin/env bash
set -euo pipefail

cd /shareNFS_40/wyf/workplace/openpi
export HF_LEROBOT_HOME=/shareNFS_40/wyf/data/lerobot

uv run examples/UniVTAC/convert_sim_to_lerobot.py \
  --hdf5-dir /shareNFS_40/wyf/data/UniVTAC/sim/GelSight_1_2_100 \
  --output-dir /shareNFS_40/wyf/data/lerobot/simulation/insert_block_gelsight_delta_eef \
  --repo-id simulation/insert_block_gelsight_delta_eef \
  --control-mode delta_eef \
  --overwrite

# Train Insert Block GelSight sim delta-EEF full fine-tuning with two vision inputs.
#
# Assumes LeRobot data already exists at:
#   /shareNFS_40/wyf/data/lerobot/simulation/insert_block_gelsight_delta_eef
#
# Defaults:
#   - config: pi05_insert_block_gelsight_sim_delta_eef_vision_full_ft
#   - GPUs: CUDA_VISIBLE_DEVICES=4,5,6,7
#   - JAX FSDP devices: 4
#   - global batch size: 64
#   - action horizon: 50
#   - train steps: 20000
#   - save interval: 5000
#   - EMA: 0.999

REPO_ROOT="${REPO_ROOT:-/shareNFS_40/wyf/workplace/openpi}"
CONFIG_NAME="${CONFIG_NAME:-pi05_insert_block_gelsight_sim_delta_eef_vision_full_ft}"
STATS_CONFIG_NAME="${STATS_CONFIG_NAME:-$CONFIG_NAME}"
EXP_NAME="${EXP_NAME:-insert_block_gelsight_sim_delta_eef_vision_full_ft_v1}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
RESUME="${RESUME:-0}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/shareNFS_40/wyf/workplace/openpi/checkpoints}"

OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/shareNFS_40/wyf/data/openpi}"
HF_HOME="${HF_HOME:-/shareNFS_40/wyf/data/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/shareNFS_40/wyf/data/hf_cache/datasets}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/shareNFS_40/wyf/data/lerobot}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

NORM_STATS_DIR="${REPO_ROOT}/assets/${STATS_CONFIG_NAME}/simulation/insert_block_gelsight_delta_eef"
RUN_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"

cd "$REPO_ROOT"
mkdir -p "$CHECKPOINT_BASE_DIR" "$HF_DATASETS_CACHE"

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OPENPI_DATA_HOME HF_HOME HF_DATASETS_CACHE HF_LEROBOT_HOME
export XLA_PYTHON_CLIENT_PREALLOCATE

if [[ "${XLA_FLAGS:-}" == *"--xla_gpu_autotune_level=0"* ]]; then
  echo "[env] removing XLA_FLAGS=${XLA_FLAGS}; this flag can make JAX GPU kernels fragile"
  unset XLA_FLAGS
fi

if [[ "$FORCE_NORM_STATS" == "1" || ! -d "$NORM_STATS_DIR" ]]; then
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
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD" \
  --fsdp-devices "$FSDP_DEVICES" \
  $(if [[ "$RESUME" == "1" ]]; then echo "--resume"; else echo "--overwrite"; fi)

echo "[done] checkpoint directory: ${RUN_DIR}"
