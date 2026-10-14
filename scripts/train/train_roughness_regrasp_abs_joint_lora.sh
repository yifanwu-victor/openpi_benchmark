#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
PI_TRAINING_DATA_ROOT="${PI_TRAINING_DATA_ROOT:-/run/ti/pi-training-data}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI_TRAINING_DATA_ROOT}/lerobot}"
HDF5_DIR="${HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/roughness_regrasp}"
REPO_ID="${REPO_ID:-simulation/roughness_regrasp_gelsight_abs_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
TACTILE_CONFIG_NAME="${TACTILE_CONFIG_NAME:-pi05_roughness_regrasp_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora}"
TACTILE_EXP_NAME="${TACTILE_EXP_NAME:-roughness_regrasp_abs_joint_vision_tactile_dual_siglip_lora_v1}"
TACTILE_GPU_IDS="${TACTILE_GPU_IDS:-4,5}"
VISION_CONFIG_NAME="${VISION_CONFIG_NAME:-pi05_roughness_regrasp_sim_abs_joint_relative_action_vision_lora}"
VISION_EXP_NAME="${VISION_EXP_NAME:-roughness_regrasp_abs_joint_vision_lora_v1}"
VISION_GPU_IDS="${VISION_GPU_IDS:-4,5}"

export REPO_ROOT PI_TRAINING_DATA_ROOT HF_LEROBOT_HOME HDF5_DIR REPO_ID OUTPUT_DIR
export TACTILE_CONFIG_NAME TACTILE_EXP_NAME TACTILE_GPU_IDS
export VISION_CONFIG_NAME VISION_EXP_NAME VISION_GPU_IDS

exec bash "${REPO_ROOT}/scripts/train/train_pull_drawer_abs_joint_lora.sh"
