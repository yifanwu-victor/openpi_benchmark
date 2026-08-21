#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
CONFIG_NAME="${CONFIG_NAME:-pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft}"
EXP_NAME="${EXP_NAME:-turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_v1}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-19999}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}/${CHECKPOINT_STEP}}"
PORT="${PORT:-8000}"
RECORD="${RECORD:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/tione/notebook/data/openpi}"
HF_HOME="${HF_HOME:-/home/tione/notebook/data/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/tione/notebook/data/hf_cache/datasets}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/home/tione/notebook/data/lerobot}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
TRITON_PTXAS_DIR="${TRITON_PTXAS_DIR:-${REPO_ROOT}/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin}"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES OPENPI_DATA_HOME HF_HOME HF_DATASETS_CACHE HF_LEROBOT_HOME XLA_PYTHON_CLIENT_PREALLOCATE
export PATH="$HOME/.local/bin:$PATH"
if [[ -x "${TRITON_PTXAS_DIR}/ptxas" ]]; then
  export PATH="${TRITON_PTXAS_DIR}:$PATH"
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "[serve] missing checkpoint dir: $CHECKPOINT_DIR" >&2
  exit 1
fi

cmd=(
  uv run scripts/serve_policy.py
  --port "$PORT"
  policy:checkpoint
  --policy.config="$CONFIG_NAME"
  --policy.dir="$CHECKPOINT_DIR"
)
if [[ "$RECORD" == "1" ]]; then
  cmd+=(--record)
fi

printf '[serve] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
