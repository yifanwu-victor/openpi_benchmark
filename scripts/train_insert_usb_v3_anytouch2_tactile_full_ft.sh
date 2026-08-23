#!/usr/bin/env bash
set -euo pipefail

# Convert Insert USB v3 UniVTAC sim data to LeRobot with AnyTouch2 tactile clips,
# compute normalization stats, then train the PyTorch AnyTouch2 tactile encoder
# full fine-tuning config.
#
# Model/data semantics:
#   visual input: base image + wrist image
#   tactile input: left/right tactile clips, each [4, 224, 224, 3]
#   prefix order: left tactile -> right tactile -> base -> wrist -> language
#   LeRobot state: [joint7, gripper]               # dim 8, absolute
#   LeRobot action: [target joint7, target gripper] # dim 8, absolute
#   Training action: [target joint7 - current joint7, abs gripper]
#
# Required checkpoint:
#   /shareNFS_40/wyf/workplace/AnyTouch2/checkpoints/anytouch2.pth
#
# If `checkpoint-4frames.pth` exists but `anytouch2.pth` does not, this script
# creates a symlink automatically.

REPO_ROOT="${REPO_ROOT:-/shareNFS_40/wyf/workplace/openpi}"

CONFIG_NAME="${CONFIG_NAME:-pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft}"
EXP_NAME="${EXP_NAME:-insert_usb_v3_anytouch2_tactile_torch_full_ft_v1}"

HDF5_DIR="${HDF5_DIR:-/shareNFS_40/wyf/data/UniVTAC/sim/insert_USB_v3/GelSight/noise3cm_1_5mm}"
OUTPUT_DIR="${OUTPUT_DIR:-/shareNFS_40/wyf/data/lerobot/simulation/insert_usb_v3_gelsight_abs_joint_anytouch2_clip}"
REPO_ID="${REPO_ID:-simulation/insert_usb_v3_gelsight_abs_joint_anytouch2_clip}"

ANYTOUCH_CHECKPOINT_DIR="${ANYTOUCH_CHECKPOINT_DIR:-/shareNFS_40/wyf/workplace/AnyTouch2/checkpoints}"
ANYTOUCH_CHECKPOINT_PATH="${ANYTOUCH_CHECKPOINT_PATH:-${ANYTOUCH_CHECKPOINT_DIR}/anytouch2.pth}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/shareNFS_40/wyf/workplace/openpi/checkpoints}"

SKIP_CONVERT="${SKIP_CONVERT:-0}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
RESUME="${RESUME:-0}"

OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/shareNFS_40/wyf/data/openpi}"
HF_HOME="${HF_HOME:-/shareNFS_40/wyf/data/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/shareNFS_40/wyf/data/hf_cache/datasets}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/shareNFS_40/wyf/data/lerobot}"

cd "$REPO_ROOT"
mkdir -p "$CHECKPOINT_BASE_DIR" "$HF_DATASETS_CACHE" "$(dirname "$OUTPUT_DIR")"

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OPENPI_DATA_HOME HF_HOME HF_DATASETS_CACHE HF_LEROBOT_HOME

if [[ ! -f "$ANYTOUCH_CHECKPOINT_PATH" && -f "${ANYTOUCH_CHECKPOINT_DIR}/checkpoint-4frames.pth" ]]; then
  echo "[anytouch2] creating symlink: ${ANYTOUCH_CHECKPOINT_PATH} -> checkpoint-4frames.pth"
  ln -s "${ANYTOUCH_CHECKPOINT_DIR}/checkpoint-4frames.pth" "$ANYTOUCH_CHECKPOINT_PATH"
fi

if [[ ! -f "$ANYTOUCH_CHECKPOINT_PATH" ]]; then
  echo "[error] Missing AnyTouch2 checkpoint: ${ANYTOUCH_CHECKPOINT_PATH}" >&2
  echo "[hint] Download xxuan01/AnyTouch2-Model, then place or symlink checkpoint-4frames.pth as anytouch2.pth." >&2
  exit 1
fi

if [[ "$SKIP_CONVERT" != "1" ]]; then
  echo "[convert] ${HDF5_DIR} -> ${OUTPUT_DIR}"
  uv run examples/anytouch2/convert_univtac_sim_delta_eef_to_lerobot.py \
    --hdf5-dir "$HDF5_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --repo-id "$REPO_ID" \
    --control-mode abs_joint \
    --overwrite
else
  echo "[convert] skipped; using existing ${OUTPUT_DIR}"
fi

echo "[metadata] pruning tactile clip stats from LeRobot metadata"
uv run examples/anytouch2/prune_tactile_clip_episode_stats.py \
  --dataset-root "$OUTPUT_DIR"

echo "[metadata] checking dataset feature dimensions"
uv run python - <<PY
import json
from pathlib import Path
root = Path("${OUTPUT_DIR}")
info = json.loads((root / "meta" / "info.json").read_text())
features = info["features"]
state_shape = tuple(features["state"]["shape"])
action_shape = tuple(features["actions"]["shape"])
left_shape = tuple(features["left_tactile_clip"]["shape"])
right_shape = tuple(features["right_tactile_clip"]["shape"])
print("state", state_shape, "actions", action_shape, "left_clip", left_shape, "right_clip", right_shape)
if state_shape != (8,) or action_shape != (8,):
    raise SystemExit(
        f"Expected abs_joint dataset with state/actions shape (8,)/(8,), got {state_shape}/{action_shape}. "
        "Rerun without SKIP_CONVERT=1 so the dataset is regenerated."
    )
if left_shape != (4, 224, 224, 3) or right_shape != (4, 224, 224, 3):
    raise SystemExit(f"Unexpected tactile clip shapes: {left_shape}, {right_shape}")
PY

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
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD" \
  $(if [[ "$RESUME" == "1" ]]; then echo "--resume"; else echo "--overwrite"; fi)

echo "[done] checkpoint directory: ${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
