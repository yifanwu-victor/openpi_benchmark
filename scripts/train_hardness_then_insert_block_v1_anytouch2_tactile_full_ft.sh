#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
cd "$REPO_ROOT"

echo "[1/2] Training hardness classify AnyTouch2 full FT"
GPU_IDS="${HARDNESS_GPU_IDS:-4,5,6,7}" \
RESUME="${HARDNESS_RESUME:-${RESUME:-0}}" \
bash scripts/train_hardness_classify_anytouch2_tactile_full_ft.sh

echo "[2/2] Training insert block v1 AnyTouch2 full FT"
GPU_IDS="${INSERT_BLOCK_V1_GPU_IDS:-4,5,6,7}" \
RESUME="${INSERT_BLOCK_V1_RESUME:-${RESUME:-0}}" \
bash scripts/train_insert_block_v1_anytouch2_tactile_full_ft.sh

echo "[done] Both AnyTouch2 training jobs finished"
