#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
PI_TRAINING_DATA_ROOT="${PI_TRAINING_DATA_ROOT:-/run/ti/pi-training-data}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI_TRAINING_DATA_ROOT}/lerobot}"
HDF5_DIR="${HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/roughness_regrasp}"
REPO_ID="${REPO_ID:-simulation/roughness_regrasp_gelsight_abs_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
TASK_LABEL="${TASK_LABEL:-roughness-regrasp-vision}"
SKIP_TACTILE=1
VISION_CONFIG_NAME="${VISION_CONFIG_NAME:-pi05_roughness_regrasp_sim_abs_joint_relative_action_vision_lora}"
VISION_NORM_CONFIG_NAME="${VISION_NORM_CONFIG_NAME:-${VISION_CONFIG_NAME}}"
VISION_EXP_NAME="${VISION_EXP_NAME:-roughness_regrasp_abs_joint_vision_lora_v1}"
VISION_GPU_IDS="${VISION_GPU_IDS:-2,3}"
STATS_GPU_IDS="${STATS_GPU_IDS:-${VISION_GPU_IDS}}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"

export REPO_ROOT PI_TRAINING_DATA_ROOT HF_LEROBOT_HOME HDF5_DIR REPO_ID OUTPUT_DIR TASK_LABEL
export SKIP_TACTILE VISION_CONFIG_NAME VISION_NORM_CONFIG_NAME VISION_EXP_NAME VISION_GPU_IDS STATS_GPU_IDS
export FSDP_DEVICES BATCH_SIZE NUM_TRAIN_STEPS

exec bash "${REPO_ROOT}/scripts/train/train_sim_abs_joint_lora.sh"
