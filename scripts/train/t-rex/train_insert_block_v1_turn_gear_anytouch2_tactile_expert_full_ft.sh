#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
export GPU_IDS="${GPU_IDS:-4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

echo "[sequence] starting insert_block_v1 on GPUs ${GPU_IDS}"
bash "${REPO_ROOT}/scripts/train/t-rex/train_insert_block_v1_anytouch2_tactile_expert_full_ft.sh" "$@"

echo "[sequence] starting turn_gear_pair on GPUs ${GPU_IDS}"
bash "${REPO_ROOT}/scripts/train/t-rex/train_turn_gear_pair_anytouch2_tactile_expert_full_ft.sh" "$@"
