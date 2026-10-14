#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
HDF5_DIR="${HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/insert_block_v1}"
PI_TRAINING_DATA_ROOT="${PI_TRAINING_DATA_ROOT:-/run/ti/pi-training-data}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI_TRAINING_DATA_ROOT}/lerobot}"
REPO_ID="${REPO_ID:-simulation/insert_block_v1_gelsight_abs_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
TASK_LABEL="${TASK_LABEL:-insert-block-v1}"

TACTILE_CONFIG_NAME="${TACTILE_CONFIG_NAME:-pi05_insert_block_v1_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora}"
TACTILE_NORM_CONFIG_NAME="${TACTILE_NORM_CONFIG_NAME:-pi05_insert_block_v1_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft}"
TACTILE_EXP_NAME="${TACTILE_EXP_NAME:-insert_block_v1_abs_joint_vision_tactile_dual_siglip_lora_v1}"
TACTILE_GPU_IDS="${TACTILE_GPU_IDS:-4,5}"
TACTILE_RESUME="${TACTILE_RESUME:-${RESUME:-0}}"
STATS_GPU_IDS="${STATS_GPU_IDS:-${TACTILE_GPU_IDS}}"

VISION_CONFIG_NAME="${VISION_CONFIG_NAME:-pi05_insert_block_v1_sim_abs_joint_relative_action_vision_lora}"
VISION_NORM_CONFIG_NAME="${VISION_NORM_CONFIG_NAME:-pi05_insert_block_v1_sim_abs_joint_relative_action_vision_full_ft}"
VISION_EXP_NAME="${VISION_EXP_NAME:-insert_block_v1_abs_joint_vision_lora_v1}"
VISION_GPU_IDS="${VISION_GPU_IDS:-4,5}"
VISION_RESUME="${VISION_RESUME:-${RESUME:-0}}"

FSDP_DEVICES="${FSDP_DEVICES:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
SKIP_CONVERT="${SKIP_CONVERT:-auto}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
OVERWRITE="${OVERWRITE:-1}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}"
SOURCE_BASHRC="${SOURCE_BASHRC:-0}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-20000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-30}"

cd "$REPO_ROOT"
if [[ "$SOURCE_BASHRC" == "1" && -f "${HOME}/.bashrc" ]]; then
  set +u
  export PS1="${PS1-}"
  source "${HOME}/.bashrc"
  set -u
fi
unset VIRTUAL_ENV

TRITON_PTXAS_DIR="${TRITON_PTXAS_DIR:-${REPO_ROOT}/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin}"
export TRITON_PTXAS_DIR
export PATH="${TRITON_PTXAS_DIR}:$HOME/.local/bin:$PATH"
export HF_LEROBOT_HOME
export CUDA_VISIBLE_DEVICES="$STATS_GPU_IDS"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

DATASET_MARKER="${OUTPUT_DIR}/meta/info.json"

wait_for_gpu_memory() {
  local gpu_ids="$1"
  local phase="$2"
  local gpu_id
  local used_memory
  local all_ready
  local -a gpu_id_array

  IFS=',' read -r -a gpu_id_array <<< "$gpu_ids"
  while true; do
    all_ready=1
    for gpu_id in "${gpu_id_array[@]}"; do
      gpu_id="${gpu_id//[[:space:]]/}"
      if ! used_memory="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits)"; then
        echo "[${TASK_LABEL}] failed to query GPU ${gpu_id} memory" >&2
        exit 1
      fi
      used_memory="${used_memory//[[:space:]]/}"
      echo "[${TASK_LABEL}] ${phase}: GPU ${gpu_id} uses ${used_memory} MiB (requires < ${GPU_MEMORY_THRESHOLD_MIB} MiB)"
      if (( used_memory >= GPU_MEMORY_THRESHOLD_MIB )); then
        all_ready=0
      fi
    done

    if (( all_ready )); then
      echo "[${TASK_LABEL}] ${phase}: all GPUs are ready"
      return
    fi

    echo "[${TASK_LABEL}] ${phase}: waiting ${GPU_WAIT_INTERVAL_SECONDS}s before checking again"
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}

run_norm_stats() {
  local config_name="$1"
  local norm_stats_file="${REPO_ROOT}/assets/${config_name}/${REPO_ID}/norm_stats.json"

  if [[ "$FORCE_NORM_STATS" == "1" || ! -f "$norm_stats_file" ]]; then
    local stats_args=(uv run scripts/compute_norm_stats.py --config-name "$config_name")
    wait_for_gpu_memory "$STATS_GPU_IDS" "norm stats (${config_name})"
    printf '[%s] norm stats (%s):' "$TASK_LABEL" "$config_name"
    printf ' %q' "${stats_args[@]}"
    printf '\n'
    "${stats_args[@]}"
  else
    echo "[${TASK_LABEL}] reusing norm stats: $norm_stats_file"
  fi
}

run_training() {
  local label="$1"
  local config_name="$2"
  local exp_name="$3"
  local gpu_ids="$4"
  local resume="$5"
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

  export CUDA_VISIBLE_DEVICES="$gpu_ids"
  wait_for_gpu_memory "$gpu_ids" "${label} training"
  if [[ "$resume" == "1" ]]; then
    train_args+=(--resume)
  else
    train_args+=(--overwrite)
  fi

  echo "[${TASK_LABEL}] starting ${label} training on GPUs ${gpu_ids}"
  echo "[${TASK_LABEL}] checkpoint: ${checkpoint_dir}"
  printf '[%s] training (%s):' "$TASK_LABEL" "$label"
  printf ' %q' uv run scripts/train.py "${train_args[@]}"
  printf '\n'
  uv run scripts/train.py "${train_args[@]}"
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[${TASK_LABEL}] raw data: $HDF5_DIR"
  echo "[${TASK_LABEL}] shared dataset: $OUTPUT_DIR"
  echo "[${TASK_LABEL}] conversion: abs_joint, left gripper, 224x224, frame stride 1, target offset 1"
  echo "[${TASK_LABEL}] conversion mode: SKIP_CONVERT=${SKIP_CONVERT}, FORCE_CONVERT=${FORCE_CONVERT}"
  echo "[${TASK_LABEL}] norm stats: tactile config ${TACTILE_NORM_CONFIG_NAME}, then vision config ${VISION_NORM_CONFIG_NAME}"
  echo "[${TASK_LABEL}] norm-stats GPUs: ${STATS_GPU_IDS}"
  echo "[${TASK_LABEL}] training order: tactile GPUs ${TACTILE_GPU_IDS}, then vision GPUs ${VISION_GPU_IDS}"
  echo "[${TASK_LABEL}] train steps: ${NUM_TRAIN_STEPS}, batch size: ${BATCH_SIZE}, FSDP devices: ${FSDP_DEVICES}"
  echo "[${TASK_LABEL}] GPU readiness: every selected GPU must use < ${GPU_MEMORY_THRESHOLD_MIB} MiB; retry every ${GPU_WAIT_INTERVAL_SECONDS}s"
  echo "[${TASK_LABEL}] tactile config: ${TACTILE_CONFIG_NAME}"
  echo "[${TASK_LABEL}] vision config: ${VISION_CONFIG_NAME}"
  echo "[${TASK_LABEL}] dry run complete"
  exit 0
fi

if ! command -v ptxas >/dev/null 2>&1; then
  echo "[${TASK_LABEL}] ptxas was not found on PATH" >&2
  exit 1
fi
printf '[%s] ptxas: ' "$TASK_LABEL"
ptxas --version

mkdir -p "$(dirname "$OUTPUT_DIR")"
if [[ "$SKIP_CONVERT" == "1" ]]; then
  echo "[${TASK_LABEL}] skipping conversion (SKIP_CONVERT=1)"
elif [[ "$FORCE_CONVERT" == "1" || ! -f "$DATASET_MARKER" ]]; then
  convert_args=(
    uv run examples/UniVTAC/convert_sim_to_lerobot.py
    --hdf5-dir "$HDF5_DIR"
    --output-dir "$OUTPUT_DIR"
    --repo-id "$REPO_ID"
    --control-mode abs_joint
    --gripper-mode left
    --image-size 224 224
    --frame-stride 1
    --target-offset 1
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    convert_args+=(--overwrite)
  fi
  if [[ -n "$MAX_EPISODES" ]]; then
    convert_args+=(--max-episodes "$MAX_EPISODES")
  fi
  printf '[%s] conversion:' "$TASK_LABEL"
  printf ' %q' "${convert_args[@]}"
  printf '\n'
  "${convert_args[@]}"
else
  echo "[${TASK_LABEL}] reusing existing dataset: $OUTPUT_DIR"
fi

run_norm_stats "$TACTILE_NORM_CONFIG_NAME"
run_norm_stats "$VISION_NORM_CONFIG_NAME"

run_training "vision+tactile dual-SigLIP" "$TACTILE_CONFIG_NAME" "$TACTILE_EXP_NAME" "$TACTILE_GPU_IDS" "$TACTILE_RESUME"
run_training "vision-only" "$VISION_CONFIG_NAME" "$VISION_EXP_NAME" "$VISION_GPU_IDS" "$VISION_RESUME"

echo "[${TASK_LABEL}] complete: tactile training followed by vision training"
