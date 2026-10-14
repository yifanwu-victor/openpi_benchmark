#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
export GPU_IDS="${GPU_IDS:-4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

echo "[sequence] starting roughness_classify on GPUs ${GPU_IDS}"
bash "${REPO_ROOT}/scripts/train/t-rex/train_roughness_classify_anytouch2_tactile_expert_full_ft.sh" "$@"

echo "[sequence] starting roughness_regrasp on GPUs ${GPU_IDS}"
bash "${REPO_ROOT}/scripts/train/t-rex/train_roughness_regrasp_anytouch2_tactile_expert_full_ft.sh" "$@"
