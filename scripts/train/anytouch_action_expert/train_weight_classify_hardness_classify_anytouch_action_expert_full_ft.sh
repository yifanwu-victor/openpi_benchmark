#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
export REPO_ROOT="$ROOT" GPU_IDS="${GPU_IDS:-0,1,2,3}" NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
COMMON="$ROOT/scripts/train/anytouch_action_expert/train_sim_abs_joint_anytouch2_action_expert_full_ft.sh"
run_task() {
  local task="$1" config="$2" exp="$3" raw="$4" repo="$5" source="$6"
  CONFIG_NAME="$config" EXP_NAME="$exp" HDF5_DIR="$raw" REPO_ID="$repo" \
    NORM_STATS_FILE="$ROOT/assets/$config/$repo/norm_stats.json" NORM_STATS_SOURCE_FILE="$source" \
    NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}" bash "$COMMON" "$@"
}
# run_task weight_classify \
#   "${WEIGHT_CONFIG_NAME:-pi05_weight_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_action_expert_torch_full_ft}" \
#   "${WEIGHT_EXP_NAME:-weight_classify_anytouch2_action_expert_torch_full_ft_v1}" \
#   "${WEIGHT_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/weight_classify}" \
#   "${WEIGHT_REPO_ID:-simulation/weight_classify_gelsight_abs_joint}" \
#   "${WEIGHT_NORM_STATS_SOURCE_FILE:-$ROOT/assets/pi05_weight_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft/simulation/weight_classify_gelsight_abs_joint/norm_stats.json}"
run_task hardness_classify \
  "${HARDNESS_CONFIG_NAME:-pi05_hardness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_action_expert_torch_full_ft}" \
  "${HARDNESS_EXP_NAME:-hardness_classify_anytouch2_action_expert_torch_full_ft_v1}" \
  "${HARDNESS_HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/hardness_classify}" \
  "${HARDNESS_REPO_ID:-simulation/hardness_classify_gelsight_abs_joint}" \
  "${HARDNESS_NORM_STATS_SOURCE_FILE:-$ROOT/assets/pi05_hardness_classify_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora/simulation/hardness_classify_gelsight_abs_joint/norm_stats.json}"
