#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
CONFIG_NAME="${CONFIG_NAME:-pi05_insert_usb_gelsight_sim_abs_joint_relative_action_vision_full_ft_100}"
EXP_NAME="${EXP_NAME:-insert_usb_gelsight_abs_joint_100_v1}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-9999}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}"
DEFAULT_CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}/${CHECKPOINT_STEP}"
DIRECT_CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${CHECKPOINT_STEP}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DEFAULT_CHECKPOINT_DIR}}"
PORT="${PORT:-8006}"
RECORD="${RECORD:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/tione/notebook/data/openpi}"
HF_HOME="${HF_HOME:-/home/tione/notebook/data/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/tione/notebook/data/hf_cache/datasets}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/home/tione/notebook/data/lerobot}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
TRITON_PTXAS_DIR="${TRITON_PTXAS_DIR:-${REPO_ROOT}/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin}"

cd "$REPO_ROOT"
if [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  export PS1="${PS1-}"
  source "${HOME}/.bashrc"
  set -u
fi
unset VIRTUAL_ENV

export CUDA_VISIBLE_DEVICES OPENPI_DATA_HOME HF_HOME HF_DATASETS_CACHE HF_LEROBOT_HOME XLA_PYTHON_CLIENT_PREALLOCATE TRITON_PTXAS_DIR
export PATH="${TRITON_PTXAS_DIR}:$HOME/.local/bin:$PATH"

printf '[serve-insert-usb-100] ptxas path: '
which ptxas
ptxas --version

if [[ ! -d "$CHECKPOINT_DIR" && "$CHECKPOINT_DIR" == "$DEFAULT_CHECKPOINT_DIR" && -d "$DIRECT_CHECKPOINT_DIR" ]]; then
  CHECKPOINT_DIR="$DIRECT_CHECKPOINT_DIR"
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "[serve-insert-usb-100] missing checkpoint dir: $CHECKPOINT_DIR" >&2
  echo "[serve-insert-usb-100] expected training output: $DEFAULT_CHECKPOINT_DIR" >&2
  echo "[serve-insert-usb-100] direct path fallback: $DIRECT_CHECKPOINT_DIR" >&2
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

printf '[serve-insert-usb-100] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
