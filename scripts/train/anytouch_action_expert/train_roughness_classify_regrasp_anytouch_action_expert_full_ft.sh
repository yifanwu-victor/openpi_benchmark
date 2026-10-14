#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
export REPO_ROOT="$ROOT" GPU_IDS="${GPU_IDS:-0,1,2,3}" NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
COMMON="$ROOT/scripts/train/anytouch_action_expert/train_sim_abs_joint_anytouch2_action_expert_full_ft.sh"
run_task() { local config="$1" exp="$2" raw="$3" repo="$4" source="$5"; CONFIG_NAME="$config" EXP_NAME="$exp" HDF5_DIR="$raw" REPO_ID="$repo" NORM_STATS_FILE="$ROOT/assets/$config/$repo/norm_stats.json" NORM_STATS_SOURCE_FILE="$source" NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}" bash "$COMMON" "$@"; }
run_task "${ROUGHNESS_CLASSIFY_CONFIG_NAME:-pi05_roughness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_action_expert_torch_full_ft}" "${ROUGHNESS_CLASSIFY_EXP_NAME:-roughness_classify_anytouch2_action_expert_torch_full_ft_v1}" "${ROUGHNESS_CLASSIFY_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/roughness_classify}" "${ROUGHNESS_CLASSIFY_REPO_ID:-simulation/roughness_classify_gelsight_abs_joint}" "${ROUGHNESS_CLASSIFY_NORM_STATS_SOURCE_FILE:-$ROOT/assets/pi05_roughness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft/simulation/roughness_classify_gelsight_abs_joint/norm_stats.json}"
run_task "${ROUGHNESS_REGRASP_CONFIG_NAME:-pi05_roughness_regrasp_sim_abs_joint_relative_action_vision_tactile_anytouch2_action_expert_torch_full_ft}" "${ROUGHNESS_REGRASP_EXP_NAME:-roughness_regrasp_anytouch2_action_expert_torch_full_ft_v1}" "${ROUGHNESS_REGRASP_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/roughness_regrasp}" "${ROUGHNESS_REGRASP_REPO_ID:-simulation/roughness_regrasp_gelsight_abs_joint}" "${ROUGHNESS_REGRASP_NORM_STATS_SOURCE_FILE:-$ROOT/assets/pi05_roughness_regrasp_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft/simulation/roughness_regrasp_gelsight_abs_joint/norm_stats.json}"
run_task "${HARDNESS_CONFIG_NAME:-pi05_hardness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_action_expert_torch_full_ft}" \
  "${HARDNESS_EXP_NAME:-hardness_classify_anytouch2_action_expert_torch_full_ft_v1}" \
  "${HARDNESS_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/hardness_classify}" \
  "${HARDNESS_REPO_ID:-simulation/hardness_classify_gelsight_abs_joint}" \
  "${HARDNESS_NORM_STATS_SOURCE_FILE:-$ROOT/assets/pi05_hardness_classify_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora/simulation/hardness_classify_gelsight_abs_joint/norm_stats.json}"
