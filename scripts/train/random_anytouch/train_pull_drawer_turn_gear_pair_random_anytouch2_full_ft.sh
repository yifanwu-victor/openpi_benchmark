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

echo "[sequence] starting pull_drawer on GPUs ${GPU_IDS}"
run_task \
  "${PULL_DRAWER_CONFIG_NAME:-pi05_pull_drawer_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft}" \
  "${PULL_DRAWER_EXP_NAME:-pull_drawer_random_anytouch2_full_ft_v1}" \
  "${PULL_DRAWER_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/pull_drawer}" \
  "${PULL_DRAWER_REPO_ID:-simulation/pull_drawer_gelsight_abs_joint}" \
  "${PULL_DRAWER_NORM_STATS_SOURCE_FILE:-${REPO_ROOT}/assets/pi05_pull_drawer_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora/simulation/pull_drawer_gelsight_abs_joint/norm_stats.json}" \
  "${PULL_DRAWER_NUM_TRAIN_STEPS:-10000}"

echo "[sequence] starting turn_gear_pair on GPUs ${GPU_IDS}"
run_task \
  "${TURN_GEAR_CONFIG_NAME:-pi05_turn_gear_pair_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft}" \
  "${TURN_GEAR_EXP_NAME:-turn_gear_pair_random_anytouch2_full_ft_v1}" \
  "${TURN_GEAR_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/turn_gear_pair}" \
  "${TURN_GEAR_REPO_ID:-simulation/turn_gear_pair_gelsight_abs_joint}" \
  "${TURN_GEAR_NORM_STATS_SOURCE_FILE:-${REPO_ROOT}/assets/pi05_turn_gear_pair_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora/simulation/turn_gear_pair_gelsight_abs_joint/norm_stats.json}" \
  "${TURN_GEAR_NUM_TRAIN_STEPS:-10000}"
