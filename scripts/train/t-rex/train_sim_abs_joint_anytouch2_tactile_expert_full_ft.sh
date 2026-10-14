#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
: "${CONFIG_NAME:?CONFIG_NAME must be set by the task launcher}"
: "${EXP_NAME:?EXP_NAME must be set by the task launcher}"
: "${REPO_ID:?REPO_ID must be set by the task launcher}"
: "${NORM_STATS_CONFIG_NAME:?NORM_STATS_CONFIG_NAME must be set by the task launcher}"

HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/run/ti/pi-training-data/lerobot}"
DATASET_DIR="${DATASET_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
NORM_STATS_FILE="${NORM_STATS_FILE:-${REPO_ROOT}/assets/${NORM_STATS_CONFIG_NAME}/${REPO_ID}/norm_stats.json}"
PYTORCH_BASE_FILE="${PYTORCH_BASE_FILE:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors}"
ANYTOUCH_CHECKPOINT_FILE="${ANYTOUCH_CHECKPOINT_FILE:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/anytouch2/checkpoint-4frames.pth}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints/t-rex}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-20000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-30}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${OPENPI_RUNNER_SNAPSHOT:-0}" != "1" && "$DRY_RUN" != "1" ]]; then
  SNAPSHOT_DIR="${RUNNER_SNAPSHOT_DIR:-/run/ti/openpi-script-snapshots}"
  mkdir -p "$SNAPSHOT_DIR"
  RUNNER_SNAPSHOT="$(mktemp "${SNAPSHOT_DIR}/train_anytouch2_tactile_expert.XXXXXX.sh")"
  cp "$0" "$RUNNER_SNAPSHOT"
  trap 'rm -f "$RUNNER_SNAPSHOT"' EXIT
  export OPENPI_RUNNER_SNAPSHOT=1
  bash "$RUNNER_SNAPSHOT"
  exit $?
fi

cd "$REPO_ROOT"
unset VIRTUAL_ENV
export HF_LEROBOT_HOME CUDA_VISIBLE_DEVICES="$GPU_IDS"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
TRITON_PTXAS_DIR="${TRITON_PTXAS_DIR:-${REPO_ROOT}/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin}"
if [[ -x "$TRITON_PTXAS_DIR/ptxas" ]]; then
  export PATH="$TRITON_PTXAS_DIR:$PATH"
fi

TRAIN_ARGS=(
  uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE"
  scripts/train_pytorch.py "$CONFIG_NAME"
  --exp-name "$EXP_NAME"
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
  --num-train-steps "$NUM_TRAIN_STEPS"
  --batch-size "$BATCH_SIZE"
  --save-interval "$SAVE_INTERVAL"
  --keep-period "$KEEP_PERIOD"
  --fsdp-devices "$NPROC_PER_NODE"
)
if [[ "$RESUME" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
else
  TRAIN_ARGS+=(--overwrite)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] dataset=${DATASET_DIR}"
  echo "[dry-run] norm_stats=${NORM_STATS_FILE}"
  echo "[dry-run] GPUs=${GPU_IDS}, processes=${NPROC_PER_NODE}, batch=${BATCH_SIZE}, steps=${NUM_TRAIN_STEPS}"
  printf '[dry-run] command:'
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

for required_file in \
  "${DATASET_DIR}/meta/info.json" \
  "$NORM_STATS_FILE" \
  "$PYTORCH_BASE_FILE" \
  "$ANYTOUCH_CHECKPOINT_FILE"; do
  if [[ ! -f "$required_file" ]]; then
    echo "[error] required file not found: ${required_file}" >&2
    exit 1
  fi
done

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -ne "$NPROC_PER_NODE" ]]; then
  echo "[error] GPU_IDS contains ${#GPU_ARRAY[@]} devices but NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi
if (( BATCH_SIZE % NPROC_PER_NODE != 0 )); then
  echo "[error] BATCH_SIZE=${BATCH_SIZE} must be divisible by NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi

while true; do
  ALL_READY=1
  for GPU_ID in "${GPU_ARRAY[@]}"; do
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
