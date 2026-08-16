#!/usr/bin/env bash
set -euo pipefail

# Convert Insert USB GelSight sim data to LeRobot delta-joint format, compute norm stats,
# then train pi0.5 vision-only full fine-tuning.
#
# Data semantics:
#   state   = [franka_joint7, gripper_q]                         # dim 8
#   actions = [target_joint7 - current_joint7, target_gripper_q]  # dim 8
#
# Defaults:
#   - raw HDF5: /shareNFS_40/wyf/data/UniVTAC/sim/insert_USB
#   - LeRobot: /shareNFS_40/wyf/data/lerobot/simulation/insert_usb_gelsight_delta_joint
#   - config: pi05_insert_usb_gelsight_sim_delta_joint_vision_full_ft
#   - GPUs: CUDA_VISIBLE_DEVICES=4,5,6,7
#   - JAX FSDP devices: 4
#   - global batch size: 64
#   - action horizon: 50
#   - train steps: 20000
#   - save interval: 5000
#   - EMA: 0.999
#
# Useful overrides:
#   HDF5_DIR=/path/to/hdf5 GPU_IDS=0,1 FSDP_DEVICES=2 bash scripts/train_insert_usb_gelsight_sim_delta_joint_vision_full_ft.sh
#   SKIP_CONVERT=1 FORCE_NORM_STATS=1 bash scripts/train_insert_usb_gelsight_sim_delta_joint_vision_full_ft.sh

REPO_ROOT="${REPO_ROOT:-/shareNFS_40/wyf/workplace/openpi}"

HDF5_DIR="${HDF5_DIR:-/shareNFS_40/wyf/data/UniVTAC/sim/insert_USB}"
OUTPUT_DIR="${OUTPUT_DIR:-/shareNFS_40/wyf/data/lerobot/simulation/insert_usb_gelsight_delta_joint}"
REPO_ID="${REPO_ID:-simulation/insert_usb_gelsight_delta_joint}"

CONFIG_NAME="${CONFIG_NAME:-pi05_insert_usb_gelsight_sim_delta_joint_vision_full_ft}"
STATS_CONFIG_NAME="${STATS_CONFIG_NAME:-$CONFIG_NAME}"
EXP_NAME="${EXP_NAME:-insert_usb_gelsight_sim_delta_joint_vision_full_ft_v1}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
RESUME="${RESUME:-0}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/shareNFS_40/wyf/workplace/openpi/checkpoints}"

OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/shareNFS_40/wyf/data/openpi}"
HF_HOME="${HF_HOME:-/shareNFS_40/wyf/data/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/shareNFS_40/wyf/data/hf_cache/datasets}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/shareNFS_40/wyf/data/lerobot}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

NORM_STATS_DIR="${REPO_ROOT}/assets/${STATS_CONFIG_NAME}/${REPO_ID}"
RUN_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"

cd "$REPO_ROOT"
mkdir -p "$CHECKPOINT_BASE_DIR" "$HF_DATASETS_CACHE" "$(dirname "$OUTPUT_DIR")"

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OPENPI_DATA_HOME HF_HOME HF_DATASETS_CACHE HF_LEROBOT_HOME
export XLA_PYTHON_CLIENT_PREALLOCATE

if [[ "${XLA_FLAGS:-}" == *"--xla_gpu_autotune_level=0"* ]]; then
  echo "[env] removing XLA_FLAGS=${XLA_FLAGS}; this flag can make JAX GPU kernels fragile"
  unset XLA_FLAGS
fi

if [[ "$SKIP_CONVERT" != "1" ]]; then
  echo "[convert] ${HDF5_DIR} -> ${OUTPUT_DIR}"
  uv run examples/UniVTAC/convert_sim_to_lerobot.py \
    --hdf5-dir "$HDF5_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --repo-id "$REPO_ID" \
    --control-mode delta_joint \
    --overwrite
else
  echo "[convert] skipped; using existing ${OUTPUT_DIR}"
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
