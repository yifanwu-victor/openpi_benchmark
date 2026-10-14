#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
COMMON_LAUNCHER="${REPO_ROOT}/scripts/train/train_sim_abs_joint_random_anytouch2_full_ft.sh"
SCRIPT_ARGS=("$@")

run_task() {
  local config_name="$1"
  local exp_name="$2"
  local hdf5_dir="$3"
  local repo_id="$4"
  local norm_stats_source="$5"
  local num_train_steps="$6"

  CONFIG_NAME="$config_name" EXP_NAME="$exp_name" HDF5_DIR="$hdf5_dir" REPO_ID="$repo_id" \
  NORM_STATS_FILE="${REPO_ROOT}/assets/${config_name}/${repo_id}/norm_stats.json" \
  NORM_STATS_SOURCE_FILE="$norm_stats_source" NUM_TRAIN_STEPS="$num_train_steps" \
  bash "$COMMON_LAUNCHER" "${SCRIPT_ARGS[@]}"
}

echo "[sequence] starting roughness_classify on GPUs ${GPU_IDS}"
run_task \
  "${ROUGHNESS_CLASSIFY_CONFIG_NAME:-pi05_roughness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft}" \
  "${ROUGHNESS_CLASSIFY_EXP_NAME:-roughness_classify_random_anytouch2_full_ft_v1}" \
  "${ROUGHNESS_CLASSIFY_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/roughness_classify}" \
  "${ROUGHNESS_CLASSIFY_REPO_ID:-simulation/roughness_classify_gelsight_abs_joint}" \
  "${ROUGHNESS_CLASSIFY_NORM_STATS_SOURCE_FILE:-${REPO_ROOT}/assets/pi05_roughness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft/simulation/roughness_classify_gelsight_abs_joint/norm_stats.json}" \
  "${ROUGHNESS_CLASSIFY_NUM_TRAIN_STEPS:-10000}"

echo "[sequence] starting roughness_regrasp on GPUs ${GPU_IDS}"
run_task \
  "${ROUGHNESS_REGRASP_CONFIG_NAME:-pi05_roughness_regrasp_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft}" \
  "${ROUGHNESS_REGRASP_EXP_NAME:-roughness_regrasp_random_anytouch2_full_ft_v1}" \
  "${ROUGHNESS_REGRASP_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/roughness_regrasp}" \
  "${ROUGHNESS_REGRASP_REPO_ID:-simulation/roughness_regrasp_gelsight_abs_joint}" \
  "${ROUGHNESS_REGRASP_NORM_STATS_SOURCE_FILE:-${REPO_ROOT}/assets/pi05_roughness_regrasp_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora/simulation/roughness_regrasp_gelsight_abs_joint/norm_stats.json}" \
  "${ROUGHNESS_REGRASP_NUM_TRAIN_STEPS:-10000}"
