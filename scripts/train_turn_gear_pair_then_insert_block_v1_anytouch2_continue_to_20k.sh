#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/run/ti/pi-training-data/lerobot}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-20000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-30}"
START_CHECKPOINT_STEP="${START_CHECKPOINT_STEP:-10000}"
DRY_RUN="${DRY_RUN:-0}"

ANYTOUCH_CHECKPOINT_PATH="${ANYTOUCH_CHECKPOINT_PATH:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/anytouch2/checkpoint-4frames.pth}"
PYTORCH_BASE_PATH="${PYTORCH_BASE_PATH:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors}"

TURN_CONFIG_NAME="pi05_turn_gear_pair_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft"
TURN_EXP_NAME="turn_gear_pair_anytouch2_tactile_torch_full_ft_v1"
TURN_REPO_ID="simulation/turn_gear_pair_gelsight_abs_joint"
TURN_NORM_STATS_FILE="${REPO_ROOT}/assets/pi05_turn_gear_pair_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora/${TURN_REPO_ID}/norm_stats.json"

INSERT_CONFIG_NAME="pi05_insert_block_v1_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft"
INSERT_EXP_NAME="insert_block_v1_anytouch2_tactile_torch_full_ft_v1"
INSERT_REPO_ID="simulation/insert_block_v1_gelsight_abs_joint"
INSERT_NORM_STATS_FILE="${REPO_ROOT}/assets/pi05_insert_block_v1_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft/${INSERT_REPO_ID}/norm_stats.json"

if [[ "${OPENPI_RUNNER_SNAPSHOT:-0}" != "1" && "$DRY_RUN" != "1" ]]; then
  SNAPSHOT_DIR="${RUNNER_SNAPSHOT_DIR:-/run/ti/openpi-script-snapshots}"
  mkdir -p "$SNAPSHOT_DIR"
  RUNNER_SNAPSHOT="$(mktemp "${SNAPSHOT_DIR}/continue_turn_then_insert_v1_anytouch2.XXXXXX.sh")"
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

wait_for_gpus() {
  local label="$1"
  local gpu_id
  local used_memory
  local all_ready
  local -a gpu_array

  IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
  if [[ "${#gpu_array[@]}" -ne "$NPROC_PER_NODE" ]]; then
    echo "[error] GPU_IDS contains ${#gpu_array[@]} devices but NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
    exit 1
  fi

  while true; do
    all_ready=1
    for gpu_id in "${gpu_array[@]}"; do
      used_memory="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits)"
      used_memory="${used_memory//[[:space:]]/}"
      echo "[gpu] ${label}: GPU ${gpu_id} uses ${used_memory} MiB; requires < ${GPU_MEMORY_THRESHOLD_MIB} MiB"
      if (( used_memory >= GPU_MEMORY_THRESHOLD_MIB )); then
        all_ready=0
      fi
    done
    if (( all_ready )); then
      return
    fi
    echo "[gpu] ${label}: waiting ${GPU_WAIT_INTERVAL_SECONDS}s"
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}

require_resume_checkpoint() {
  local label="$1"
  local config_name="$2"
  local exp_name="$3"
  local checkpoint_dir="${CHECKPOINT_BASE_DIR}/${config_name}/${exp_name}/${START_CHECKPOINT_STEP}"
  local file

  for file in model.safetensors optimizer.pt metadata.pt; do
    if [[ ! -f "${checkpoint_dir}/${file}" ]]; then
      echo "[error] ${label}: resume file not found: ${checkpoint_dir}/${file}" >&2
      exit 1
    fi
  done
  echo "[checkpoint] ${label}: found complete step ${START_CHECKPOINT_STEP} checkpoint"
}

train_task() {
  local label="$1"
  local config_name="$2"
  local exp_name="$3"
  local -a train_args=(
    uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE"
    scripts/train_pytorch.py "$config_name"
    --exp-name "$exp_name"
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
    --num-train-steps "$NUM_TRAIN_STEPS"
    --batch-size "$BATCH_SIZE"
    --save-interval "$SAVE_INTERVAL"
    --keep-period "$KEEP_PERIOD"
    --resume
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] %s:' "$label"
    printf ' %q' "${train_args[@]}"
    printf '\n'
    return
  fi

  wait_for_gpus "$label"
  printf '[train] %s:' "$label"
  printf ' %q' "${train_args[@]}"
  printf '\n'
  "${train_args[@]}"
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] order: turn_gear_pair -> insert_block_v1"
  echo "[dry-run] GPUs=${GPU_IDS}, processes=${NPROC_PER_NODE}, batch=${BATCH_SIZE}"
  echo "[dry-run] resume from latest checkpoint at or after ${START_CHECKPOINT_STEP}; target=${NUM_TRAIN_STEPS}"
  train_task "turn_gear_pair" "$TURN_CONFIG_NAME" "$TURN_EXP_NAME"
  train_task "insert_block_v1" "$INSERT_CONFIG_NAME" "$INSERT_EXP_NAME"
  exit 0
fi

for required_file in \
  "$ANYTOUCH_CHECKPOINT_PATH" \
  "$PYTORCH_BASE_PATH" \
  "${HF_LEROBOT_HOME}/${TURN_REPO_ID}/meta/info.json" \
  "$TURN_NORM_STATS_FILE" \
  "${HF_LEROBOT_HOME}/${INSERT_REPO_ID}/meta/info.json" \
  "$INSERT_NORM_STATS_FILE"; do
  if [[ ! -f "$required_file" ]]; then
    echo "[error] required file not found: ${required_file}" >&2
    exit 1
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[error] nvidia-smi not found" >&2
  exit 1
fi

require_resume_checkpoint "turn_gear_pair" "$TURN_CONFIG_NAME" "$TURN_EXP_NAME"
require_resume_checkpoint "insert_block_v1" "$INSERT_CONFIG_NAME" "$INSERT_EXP_NAME"

echo "[1/2] Continuing Turn Gear Pair from 10k to 20k"
train_task "turn_gear_pair" "$TURN_CONFIG_NAME" "$TURN_EXP_NAME"

echo "[2/2] Continuing Insert Block V1 from 10k to 20k"
train_task "insert_block_v1" "$INSERT_CONFIG_NAME" "$INSERT_EXP_NAME"

echo "[done] Turn Gear Pair and Insert Block V1 reached target step ${NUM_TRAIN_STEPS}"
