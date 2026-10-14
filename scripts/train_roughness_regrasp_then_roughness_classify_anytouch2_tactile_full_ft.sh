#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
cd "$REPO_ROOT"

GPU_IDS="${GPU_IDS:-4,5,6,7}" \
TURN_TASK_LABEL="roughness_regrasp" \
TURN_DISPLAY_NAME="Roughness Regrasp" \
TURN_RAW_DIR="/home/tione/notebook/data/UniVTAC/sim/roughness_regrasp" \
TURN_REPO_ID="simulation/roughness_regrasp_gelsight_abs_joint" \
TURN_OUTPUT_DIR="${HF_LEROBOT_HOME:-/run/ti/pi-training-data/lerobot}/simulation/roughness_regrasp_gelsight_abs_joint" \
TURN_CONFIG_NAME="pi05_roughness_regrasp_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft" \
TURN_NORM_CONFIG_NAME="pi05_roughness_regrasp_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora" \
TURN_NORM_STATS_FILE="${REPO_ROOT}/assets/pi05_roughness_regrasp_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora/simulation/roughness_regrasp_gelsight_abs_joint/norm_stats.json" \
TURN_EXP_NAME="roughness_regrasp_anytouch2_tactile_torch_full_ft_v1" \
WEIGHT_TASK_LABEL="roughness_classify" \
WEIGHT_DISPLAY_NAME="Roughness Classify" \
WEIGHT_RAW_DIR="/home/tione/notebook/data/UniVTAC/sim/roughness_classify" \
WEIGHT_REPO_ID="simulation/roughness_classify_gelsight_abs_joint" \
WEIGHT_OUTPUT_DIR="${HF_LEROBOT_HOME:-/run/ti/pi-training-data/lerobot}/simulation/roughness_classify_gelsight_abs_joint" \
WEIGHT_CONFIG_NAME="pi05_roughness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft" \
WEIGHT_NORM_CONFIG_NAME="pi05_roughness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft" \
WEIGHT_NORM_STATS_FILE="${REPO_ROOT}/assets/pi05_roughness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft/simulation/roughness_classify_gelsight_abs_joint/norm_stats.json" \
WEIGHT_EXP_NAME="roughness_classify_anytouch2_tactile_torch_full_ft_v1" \
bash scripts/train_turn_gear_pair_then_weight_classify_anytouch2_tactile_full_ft.sh
