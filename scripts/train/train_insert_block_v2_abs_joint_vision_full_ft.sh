#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
CONFIG_NAME="${CONFIG_NAME:-pi05_insert_block_v2_sim_abs_joint_relative_action_vision_full_ft}"
EXP_NAME="${EXP_NAME:-insert_block_v2_abs_joint_vision_full_ft_v1}"
REPO_ID="${REPO_ID:-simulation/insert_block_v2_gelsight_abs_joint}"

HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/run/ti/pi-training-data/lerobot}"
DATASET_MARKER="${HF_LEROBOT_HOME}/${REPO_ID}/meta/info.json"
NORM_STATS_FILE="${REPO_ROOT}/assets/${CONFIG_NAME}/${REPO_ID}/norm_stats.json"
BASE_PARAMS="${BASE_PARAMS:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base/params}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-20000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-30}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${OPENPI_RUNNER_SNAPSHOT:-0}" != "1" && "$DRY_RUN" != "1" ]]; then
  SNAPSHOT_DIR="${RUNNER_SNAPSHOT_DIR:-/run/ti/openpi-script-snapshots}"
  mkdir -p "$SNAPSHOT_DIR"
  RUNNER_SNAPSHOT="$(mktemp "${SNAPSHOT_DIR}/train_insert_block_v2_vision_full_ft.XXXXXX.sh")"
  cp "$0" "$RUNNER_SNAPSHOT"
  trap 'rm -f "$RUNNER_SNAPSHOT"' EXIT
  export OPENPI_RUNNER_SNAPSHOT=1
  bash "$RUNNER_SNAPSHOT"
  exit $?
fi

cd "$REPO_ROOT"
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${REPO_ROOT}/.venv}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_LEROBOT_HOME
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
TRITON_PTXAS_DIR="${TRITON_PTXAS_DIR:-${REPO_ROOT}/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin}"
if [[ -x "$TRITON_PTXAS_DIR/ptxas" ]]; then
  export PATH="$TRITON_PTXAS_DIR:$PATH"
fi

TRAIN_ARGS=(
  uv run scripts/train.py "$CONFIG_NAME"
  --exp-name "$EXP_NAME"
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
  --num-train-steps "$NUM_TRAIN_STEPS"
  --batch-size "$BATCH_SIZE"
  --save-interval "$SAVE_INTERVAL"
  --keep-period "$KEEP_PERIOD"
  --fsdp-devices "$FSDP_DEVICES"
)
if [[ "$RESUME" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
else
  TRAIN_ARGS+=(--overwrite)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] dataset=${DATASET_MARKER}"
  echo "[dry-run] norm_stats=${NORM_STATS_FILE}"
  echo "[dry-run] base_params=${BASE_PARAMS}"
  echo "[dry-run] GPUs=${GPU_IDS}, FSDP=${FSDP_DEVICES}, batch=${BATCH_SIZE}, steps=${NUM_TRAIN_STEPS}"
  printf '[dry-run] command:'
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

for required_path in "$DATASET_MARKER" "$NORM_STATS_FILE" "$BASE_PARAMS"; do
  if [[ ! -e "$required_path" ]]; then
    echo "[error] required path not found: ${required_path}" >&2
    exit 1
  fi
done
command -v nvidia-smi >/dev/null 2>&1 || { echo "[error] nvidia-smi was not found" >&2; exit 1; }

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -ne "$FSDP_DEVICES" ]]; then
  echo "[error] GPU_IDS contains ${#GPU_ARRAY[@]} devices but FSDP_DEVICES=${FSDP_DEVICES}" >&2
  exit 1
fi

while true; do
  ALL_READY=1
  for GPU_ID in "${GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ID//[[:space:]]/}"
    USED_MEMORY="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits)"
    USED_MEMORY="${USED_MEMORY//[[:space:]]/}"
    echo "[gpu] ${GPU_ID}: ${USED_MEMORY} MiB used; requires < ${GPU_MEMORY_THRESHOLD_MIB} MiB"
    if (( USED_MEMORY >= GPU_MEMORY_THRESHOLD_MIB )); then
      ALL_READY=0
    fi
  done
  if (( ALL_READY )); then
    break
  fi
  echo "[gpu] waiting ${GPU_WAIT_INTERVAL_SECONDS}s"
  sleep "$GPU_WAIT_INTERVAL_SECONDS"
done

mkdir -p "$CHECKPOINT_BASE_DIR"
printf '[train] command:'
printf ' %q' "${TRAIN_ARGS[@]}"
printf '\n'
"${TRAIN_ARGS[@]}"
