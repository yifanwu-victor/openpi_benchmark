#!/usr/bin/env bash
set -euo pipefail

# Train Insert USB v3 with PyTorch full fine-tuning and two independent SigLIP towers:
#   base/wrist images      -> original PaliGemma SigLIP
#   left/right tactile     -> copied tactile SigLIP
#
# Stored LeRobot data:
#   state   = [franka_joint7, gripper_q]          # dim 8, absolute
#   actions = [target_joint7, target_gripper_q]   # dim 8, absolute
#
# Training transform:
#   actions = [target_joint7 - current_joint7, target_gripper_q]
#
# Default:
#   config: pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_dual_siglip_torch_full_ft
#   dataset: /shareNFS_40/wyf/data/lerobot/simulation/insert_usb_v3_gelsight_abs_joint
#   GPUs: 0,1,2,3
#   global batch size: 64
#   train steps: 10000

REPO_ROOT="${REPO_ROOT:-/shareNFS_40/wyf/workplace/openpi}"

CONFIG_NAME="${CONFIG_NAME:-pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_dual_siglip_torch_full_ft}"
EXP_NAME="${EXP_NAME:-insert_usb_v3_dual_siglip_torch_full_ft_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/shareNFS_40/wyf/workplace/openpi/checkpoints}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
REPO_ID="${REPO_ID:-simulation/insert_usb_v3_gelsight_abs_joint}"

OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/shareNFS_40/wyf/data/openpi}"
HF_HOME="${HF_HOME:-/shareNFS_40/wyf/data/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/shareNFS_40/wyf/data/hf_cache/datasets}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/shareNFS_40/wyf/data/lerobot}"

cd "$REPO_ROOT"
mkdir -p "$CHECKPOINT_BASE_DIR" "$HF_DATASETS_CACHE"

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OPENPI_DATA_HOME HF_HOME HF_DATASETS_CACHE HF_LEROBOT_HOME

NORM_STATS_DIR="${REPO_ROOT}/assets/${CONFIG_NAME}/${REPO_ID}"
if [[ "$FORCE_NORM_STATS" == "1" || ! -d "$NORM_STATS_DIR" ]]; then
  echo "[norm-stats] computing stats for ${CONFIG_NAME}"
  uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
else
  echo "[norm-stats] found ${NORM_STATS_DIR}; set FORCE_NORM_STATS=1 to recompute"
fi

echo "[train] config=${CONFIG_NAME}"
echo "[train] exp=${EXP_NAME}"
echo "[train] GPUs=${GPU_IDS}, nproc_per_node=${NPROC_PER_NODE}, global_batch_size=${BATCH_SIZE}"
echo "[train] checkpoint_base_dir=${CHECKPOINT_BASE_DIR}"

uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train_pytorch.py \
  "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --overwrite \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD"
