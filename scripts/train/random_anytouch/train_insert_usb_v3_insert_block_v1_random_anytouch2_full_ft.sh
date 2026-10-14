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

# Insert USB v3 was already completed; keep this section commented so reruns
# continue directly with Insert Block V1.
# echo "[sequence] starting insert_usb_v3 on GPUs ${GPU_IDS}"
# run_task \
#   "${INSERT_USB_CONFIG_NAME:-pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft}" \
#   "${INSERT_USB_EXP_NAME:-insert_usb_v3_random_anytouch2_full_ft_v1}" \
#   "${INSERT_USB_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/insert_USB/GelSight/noise3cm_1_5mm}" \
#   "${INSERT_USB_REPO_ID:-simulation/insert_usb_v3_gelsight_abs_joint}" \
#   "${INSERT_USB_NORM_STATS_SOURCE_FILE:-${REPO_ROOT}/assets/pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft/simulation/insert_usb_v3_gelsight_abs_joint/norm_stats.json}" \
#   "${INSERT_USB_NUM_TRAIN_STEPS:-10000}"

echo "[sequence] starting insert_block_v1 on GPUs ${GPU_IDS}"
run_task \
  "${INSERT_BLOCK_V1_CONFIG_NAME:-pi05_insert_block_v1_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft}" \
  "${INSERT_BLOCK_V1_EXP_NAME:-insert_block_v1_random_anytouch2_full_ft_v1}" \
  "${INSERT_BLOCK_V1_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/insert_block_v1}" \
  "${INSERT_BLOCK_V1_REPO_ID:-simulation/insert_block_v1_gelsight_abs_joint}" \
  "${INSERT_BLOCK_V1_NORM_STATS_SOURCE_FILE:-${REPO_ROOT}/assets/pi05_insert_block_v1_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft/simulation/insert_block_v1_gelsight_abs_joint/norm_stats.json}" \
  "${INSERT_BLOCK_V1_NUM_TRAIN_STEPS:-10000}"
