#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
PI_TRAINING_DATA_ROOT="${PI_TRAINING_DATA_ROOT:-/run/ti/pi-training-data}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI_TRAINING_DATA_ROOT}/lerobot}"
HDF5_DIR="${HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/insert_block_v2}"
REPO_ID="${REPO_ID:-simulation/insert_block_v2_gelsight_abs_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
TASK_LABEL="${TASK_LABEL:-insert-block-v2}"
TACTILE_CONFIG_NAME="${TACTILE_CONFIG_NAME:-pi05_insert_block_v2_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora}"
TACTILE_NORM_CONFIG_NAME="${TACTILE_NORM_CONFIG_NAME:-pi05_insert_block_v2_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft}"
TACTILE_EXP_NAME="${TACTILE_EXP_NAME:-insert_block_v2_abs_joint_vision_tactile_dual_siglip_lora_v1}"
TACTILE_GPU_IDS="${TACTILE_GPU_IDS:-6,7}"
STATS_GPU_IDS="${STATS_GPU_IDS:-${TACTILE_GPU_IDS}}"
VISION_CONFIG_NAME="${VISION_CONFIG_NAME:-pi05_insert_block_v2_sim_abs_joint_relative_action_vision_lora}"
VISION_NORM_CONFIG_NAME="${VISION_NORM_CONFIG_NAME:-pi05_insert_block_v2_sim_abs_joint_relative_action_vision_full_ft}"
VISION_EXP_NAME="${VISION_EXP_NAME:-insert_block_v2_abs_joint_vision_lora_v1}"
VISION_GPU_IDS="${VISION_GPU_IDS:-6,7}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"

export REPO_ROOT PI_TRAINING_DATA_ROOT HF_LEROBOT_HOME HDF5_DIR REPO_ID OUTPUT_DIR TASK_LABEL
export TACTILE_CONFIG_NAME TACTILE_NORM_CONFIG_NAME TACTILE_EXP_NAME TACTILE_GPU_IDS STATS_GPU_IDS
export VISION_CONFIG_NAME VISION_NORM_CONFIG_NAME VISION_EXP_NAME VISION_GPU_IDS
export FSDP_DEVICES BATCH_SIZE NUM_TRAIN_STEPS

exec bash "${REPO_ROOT}/scripts/train/train_insert_block_v1_abs_joint_lora.sh"
