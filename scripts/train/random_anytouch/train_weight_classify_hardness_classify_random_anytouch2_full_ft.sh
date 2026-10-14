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

  CONFIG_NAME="$config_name" \
  EXP_NAME="$exp_name" \
  HDF5_DIR="$hdf5_dir" \
  REPO_ID="$repo_id" \
  NORM_STATS_FILE="${REPO_ROOT}/assets/${config_name}/${repo_id}/norm_stats.json" \
  NORM_STATS_SOURCE_FILE="$norm_stats_source" \
  NUM_TRAIN_STEPS="$num_train_steps" \
  bash "$COMMON_LAUNCHER" "${SCRIPT_ARGS[@]}"
}

echo "[sequence] starting weight_classify on GPUs ${GPU_IDS}"
run_task \
  "${WEIGHT_CONFIG_NAME:-pi05_weight_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft}" \
  "${WEIGHT_EXP_NAME:-weight_classify_random_anytouch2_full_ft_v1}" \
  "${WEIGHT_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/weight_classify}" \
  "${WEIGHT_REPO_ID:-simulation/weight_classify_gelsight_abs_joint}" \
  "${WEIGHT_NORM_STATS_SOURCE_FILE:-${REPO_ROOT}/assets/pi05_weight_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft/simulation/weight_classify_gelsight_abs_joint/norm_stats.json}" \
  "${WEIGHT_NUM_TRAIN_STEPS:-10000}"

echo "[sequence] starting hardness_classify on GPUs ${GPU_IDS}"
run_task \
  "${HARDNESS_CONFIG_NAME:-pi05_hardness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft}" \
  "${HARDNESS_EXP_NAME:-hardness_classify_random_anytouch2_full_ft_v1}" \
  "${HARDNESS_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/hardness_classify}" \
  "${HARDNESS_REPO_ID:-simulation/hardness_classify_gelsight_abs_joint}" \
  "${HARDNESS_NORM_STATS_SOURCE_FILE:-${REPO_ROOT}/assets/pi05_hardness_classify_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora/simulation/hardness_classify_gelsight_abs_joint/norm_stats.json}" \
  "${HARDNESS_NUM_TRAIN_STEPS:-10000}"
