#!/usr/bin/env bash
set -euo pipefail
export CONFIG_NAME="${CONFIG_NAME:-pi05_roughness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_tactile_expert_torch_full_ft}"
export EXP_NAME="${EXP_NAME:-roughness_classify_anytouch2_tactile_expert_torch_full_ft_v1}"
export REPO_ID="${REPO_ID:-simulation/roughness_classify_gelsight_abs_joint}"
export NORM_STATS_CONFIG_NAME="${NORM_STATS_CONFIG_NAME:-pi05_roughness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft}"
export GPU_IDS="${GPU_IDS:-4,5,6,7}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
exec "${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}/scripts/train/t-rex/train_sim_abs_joint_anytouch2_tactile_expert_full_ft.sh" "$@"
