#!/usr/bin/env bash
set -euo pipefail
ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
export REPO_ROOT="$ROOT" GPU_IDS="${GPU_IDS:-0,1,2,3}" NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
COMMON="$ROOT/scripts/train/anytouch_action_expert/train_sim_abs_joint_anytouch2_action_expert_full_ft.sh"
export CONFIG_NAME="${CONFIG_NAME:-pi05_insert_block_v2_sim_abs_joint_relative_action_vision_tactile_anytouch2_action_expert_torch_full_ft}"
export EXP_NAME="${EXP_NAME:-insert_block_v2_anytouch2_action_expert_torch_full_ft_v1}"
export HDF5_DIR="${HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/insert_block_v2}"
export REPO_ID="${REPO_ID:-simulation/insert_block_v2_gelsight_abs_joint}"
export NORM_STATS_FILE="${NORM_STATS_FILE:-$ROOT/assets/$CONFIG_NAME/$REPO_ID/norm_stats.json}"
export NORM_STATS_SOURCE_FILE="${NORM_STATS_SOURCE_FILE:-$ROOT/assets/pi05_insert_block_v2_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft/simulation/insert_block_v2_gelsight_abs_joint/norm_stats.json}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
exec bash "$COMMON" "$@"
