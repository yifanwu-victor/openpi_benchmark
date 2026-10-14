#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
PI_TRAINING_DATA_ROOT="${PI_TRAINING_DATA_ROOT:-/run/ti/pi-training-data}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI_TRAINING_DATA_ROOT}/lerobot}"
REPO_ID="simulation/insert_block_v1_gelsight_abs_joint"
VISION_CONFIG_NAME="pi05_insert_block_v1_sim_abs_joint_relative_action_vision_full_ft"
VISION_EXP_NAME="${VISION_EXP_NAME:-insert_block_v1_abs_joint_vision_full_ft_v1}"
TACTILE_CONFIG_NAME="pi05_insert_block_v1_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft"
TACTILE_EXP_NAME="${TACTILE_EXP_NAME:-insert_block_v1_abs_joint_vision_tactile_dual_siglip_full_ft_v1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-20000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-30}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
VISION_RESUME="${VISION_RESUME:-${RESUME}}"
TACTILE_RESUME="${TACTILE_RESUME:-${RESUME}}"

if [[ "${OPENPI_RUNNER_SNAPSHOT:-0}" != "1" && "$DRY_RUN" != "1" ]]; then
  SNAPSHOT_DIR="${RUNNER_SNAPSHOT_DIR:-/run/ti/openpi-script-snapshots}"
  mkdir -p "$SNAPSHOT_DIR"
  RUNNER_SNAPSHOT="$(mktemp "${SNAPSHOT_DIR}/train_insert_block_v1_full_ft.XXXXXX.sh")"
  cp "$0" "$RUNNER_SNAPSHOT"
  trap 'rm -f "$RUNNER_SNAPSHOT"' EXIT
  export OPENPI_RUNNER_SNAPSHOT=1
  bash "$RUNNER_SNAPSHOT"
  exit $?
fi

cd "$REPO_ROOT"
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${REPO_ROOT}/.venv}"
TRITON_PTXAS_DIR="${TRITON_PTXAS_DIR:-${REPO_ROOT}/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin}"
export TRITON_PTXAS_DIR
export PATH="${TRITON_PTXAS_DIR}:$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_LEROBOT_HOME
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

DATASET_MARKER="${HF_LEROBOT_HOME}/${REPO_ID}/meta/info.json"
VISION_NORM_STATS_FILE="${REPO_ROOT}/assets/${VISION_CONFIG_NAME}/${REPO_ID}/norm_stats.json"
TACTILE_NORM_STATS_FILE="${REPO_ROOT}/assets/${TACTILE_CONFIG_NAME}/${REPO_ID}/norm_stats.json"
BASE_PARAMS="/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base/params"

wait_for_gpu_memory() {
  local gpu_id
  local used_memory
  local all_ready
  local -a gpu_id_array

  IFS=',' read -r -a gpu_id_array <<< "$GPU_IDS"
  if [[ "${#gpu_id_array[@]}" -ne "$FSDP_DEVICES" ]]; then
    echo "Expected ${FSDP_DEVICES} GPU IDs, got ${GPU_IDS}" >&2
    exit 1
  fi
  if [[ ! "$GPU_MEMORY_THRESHOLD_MIB" =~ ^[0-9]+$ ]]; then
    echo "GPU_MEMORY_THRESHOLD_MIB must be an integer" >&2
    exit 1
  fi

  while true; do
    all_ready=1
    for gpu_id in "${gpu_id_array[@]}"; do
      gpu_id="${gpu_id//[[:space:]]/}"
      if ! used_memory="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits)"; then
        echo "Failed to query GPU ${gpu_id} memory" >&2
        exit 1
      fi
      used_memory="${used_memory//[[:space:]]/}"
      if [[ ! "$used_memory" =~ ^[0-9]+$ ]]; then
        echo "Invalid memory value for GPU ${gpu_id}: ${used_memory}" >&2
        exit 1
      fi
      echo "GPU ${gpu_id} uses ${used_memory} MiB (requires < ${GPU_MEMORY_THRESHOLD_MIB} MiB)"
      if (( used_memory >= GPU_MEMORY_THRESHOLD_MIB )); then
        all_ready=0
      fi
    done
    if (( all_ready )); then
      echo "All GPUs are ready"
      return
    fi
    echo "Waiting ${GPU_WAIT_INTERVAL_SECONDS}s before checking again"
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}

run_training() {
  local label="$1"
  local config_name="$2"
  local exp_name="$3"
  local resume="$4"
  local checkpoint_dir="${CHECKPOINT_BASE_DIR}/${config_name}/${exp_name}"
  local train_args=(
    "$config_name"
    --exp-name "$exp_name"
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
    --num-train-steps "$NUM_TRAIN_STEPS"
    --batch-size "$BATCH_SIZE"
    --save-interval "$SAVE_INTERVAL"
    --keep-period "$KEEP_PERIOD"
    --fsdp-devices "$FSDP_DEVICES"
  )

  wait_for_gpu_memory
  if [[ "$resume" == "1" ]]; then
    train_args+=(--resume)
  else
    train_args+=(--overwrite)
  fi
  echo "Starting ${label} training on GPUs ${GPU_IDS}"
  echo "Checkpoint: ${checkpoint_dir}"
  printf 'uv run scripts/train.py:'
  printf ' %q' "${train_args[@]}"
  printf '\n'
  uv run scripts/train.py "${train_args[@]}"
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "dataset marker: ${DATASET_MARKER}"
  echo "vision norm stats: ${VISION_NORM_STATS_FILE}"
  echo "tactile norm stats: ${TACTILE_NORM_STATS_FILE}"
  echo "base params: ${BASE_PARAMS}"
  echo "CUDA_VISIBLE_DEVICES=${GPU_IDS}"
  echo "training order: vision, then vision+tactile dual-SigLIP"
  echo "vision config: ${VISION_CONFIG_NAME}"
  echo "tactile config: ${TACTILE_CONFIG_NAME}"
  echo "batch size: ${BATCH_SIZE}, train steps: ${NUM_TRAIN_STEPS}, FSDP devices: ${FSDP_DEVICES}"
  exit 0
fi

for required_path in "$DATASET_MARKER" "$VISION_NORM_STATS_FILE" "$TACTILE_NORM_STATS_FILE" "$BASE_PARAMS"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi was not found on PATH" >&2; exit 1; }
command -v ptxas >/dev/null 2>&1 || { echo "ptxas was not found on PATH" >&2; exit 1; }

run_training "vision" "$VISION_CONFIG_NAME" "$VISION_EXP_NAME" "$VISION_RESUME"
run_training "vision+tactile dual-SigLIP" "$TACTILE_CONFIG_NAME" "$TACTILE_EXP_NAME" "$TACTILE_RESUME"

echo "Insert Block V1 full-FT training complete"
