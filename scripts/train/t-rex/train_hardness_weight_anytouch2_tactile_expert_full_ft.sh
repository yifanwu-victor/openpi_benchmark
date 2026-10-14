#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

echo "[sequence] starting hardness_classify on GPUs ${GPU_IDS}"
bash "${REPO_ROOT}/scripts/train/t-rex/train_hardness_classify_anytouch2_tactile_expert_full_ft.sh" "$@"

echo "[sequence] starting weight_classify on GPUs ${GPU_IDS}"
bash "${REPO_ROOT}/scripts/train/t-rex/train_weight_classify_anytouch2_tactile_expert_full_ft.sh" "$@"
