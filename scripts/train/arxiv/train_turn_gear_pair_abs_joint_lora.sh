#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
PI_TRAINING_DATA_ROOT="${PI_TRAINING_DATA_ROOT:-/run/ti/pi-training-data}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI_TRAINING_DATA_ROOT}/lerobot}"
HDF5_DIR="${HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/turn_gear_pair}"
REPO_ID="${REPO_ID:-simulation/turn_gear_pair_gelsight_abs_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
TACTILE_CONFIG_NAME="${TACTILE_CONFIG_NAME:-pi05_turn_gear_pair_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora}"
TACTILE_EXP_NAME="${TACTILE_EXP_NAME:-turn_gear_pair_abs_joint_vision_tactile_dual_siglip_lora_v1}"
TACTILE_GPU_IDS="${TACTILE_GPU_IDS:-6,7}"
VISION_CONFIG_NAME="${VISION_CONFIG_NAME:-pi05_turn_gear_pair_sim_abs_joint_relative_action_vision_lora}"
VISION_EXP_NAME="${VISION_EXP_NAME:-turn_gear_pair_abs_joint_vision_lora_v1}"
VISION_GPU_IDS="${VISION_GPU_IDS:-6,7}"
REWRITE_DATASET="${REWRITE_DATASET:-0}"

if [[ "$REWRITE_DATASET" == "1" && "${SKIP_CONVERT:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "[turn-gear-pair] removing existing dataset before conversion: $OUTPUT_DIR"
  rm -rf -- "$OUTPUT_DIR"
fi

export REPO_ROOT PI_TRAINING_DATA_ROOT HF_LEROBOT_HOME HDF5_DIR REPO_ID OUTPUT_DIR
export TACTILE_CONFIG_NAME TACTILE_EXP_NAME TACTILE_GPU_IDS
export VISION_CONFIG_NAME VISION_EXP_NAME VISION_GPU_IDS

exec bash "${REPO_ROOT}/scripts/train/train_pull_drawer_abs_joint_lora.sh"
