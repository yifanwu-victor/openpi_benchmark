#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/run/ti/pi-training-data/lerobot}"
REPO_ID="${REPO_ID:-simulation/pick_and_place_chips_gelsight_abs_joint}"
DATASET_DIR="${DATASET_DIR:-$HF_LEROBOT_HOME/$REPO_ID}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-20000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-30}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$REPO_ROOT/checkpoints}"
TREX_CHECKPOINT_BASE_DIR="${TREX_CHECKPOINT_BASE_DIR:-$REPO_ROOT/checkpoints/t-rex}"
PYTORCH_BASE_FILE="${PYTORCH_BASE_FILE:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors}"
ANYTOUCH_CHECKPOINT_FILE="${ANYTOUCH_CHECKPOINT_FILE:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/anytouch2/checkpoint-4frames.pth}"
RESUME="${RESUME:-0}"
ANYTOUCH_RESUME="${ANYTOUCH_RESUME:-$RESUME}"
TREX_RESUME="${TREX_RESUME:-$RESUME}"
DRY_RUN="${DRY_RUN:-0}"

TACTILE_STATS_CONFIG_NAME="pi05_pick_and_place_chips_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft"
VISION_STATS_CONFIG_NAME="pi05_pick_and_place_chips_sim_abs_joint_relative_action_vision_full_ft"
VISION_STATS="$REPO_ROOT/assets/$VISION_STATS_CONFIG_NAME/$REPO_ID/norm_stats.json"
TACTILE_STATS="$REPO_ROOT/assets/$TACTILE_STATS_CONFIG_NAME/$REPO_ID/norm_stats.json"
ANYTOUCH_CONFIG_NAME="pi05_pick_and_place_chips_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft"
ANYTOUCH_EXP_NAME="${ANYTOUCH_EXP_NAME:-pick_and_place_chips_anytouch2_tactile_torch_full_ft_v1}"
TREX_CONFIG_NAME="pi05_pick_and_place_chips_sim_abs_joint_relative_action_vision_tactile_anytouch2_tactile_expert_torch_full_ft"
TREX_EXP_NAME="${TREX_EXP_NAME:-pick_and_place_chips_anytouch2_tactile_expert_torch_full_ft_v1}"

cd "$REPO_ROOT"
unset VIRTUAL_ENV
export HF_LEROBOT_HOME CUDA_VISIBLE_DEVICES="$GPU_IDS"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
TRITON_PTXAS_DIR="${TRITON_PTXAS_DIR:-$REPO_ROOT/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin}"
[[ ! -x "$TRITON_PTXAS_DIR/ptxas" ]] || export PATH="$TRITON_PTXAS_DIR:$PATH"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -ne "$NPROC_PER_NODE" ]]; then
  echo "[error] GPU_IDS contains ${#GPU_ARRAY[@]} devices but NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  exit 1
fi
if (( BATCH_SIZE % NPROC_PER_NODE != 0 )); then
  echo "[error] BATCH_SIZE=$BATCH_SIZE must be divisible by NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  exit 1
fi

check_inputs() {
  [[ "$DRY_RUN" == 1 ]] && return
  local required
  for required in "$DATASET_DIR/meta/info.json" "$VISION_STATS" "$TACTILE_STATS" "$PYTORCH_BASE_FILE" "$ANYTOUCH_CHECKPOINT_FILE"; do
    [[ -f "$required" ]] || { echo "[error] required prepared artifact not found: $required" >&2; exit 1; }
  done
}

wait_for_gpus() {
  local phase="$1" gpu_id used all_ready
  while true; do
    all_ready=1
    for gpu_id in "${GPU_ARRAY[@]}"; do
      gpu_id="${gpu_id//[[:space:]]/}"
      used="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits)"
      used="${used//[[:space:]]/}"
      [[ "$used" =~ ^[0-9]+$ ]] || { echo "[error] invalid memory value for GPU $gpu_id: $used" >&2; exit 1; }
      echo "[gpu] $phase: GPU $gpu_id uses $used MiB; requires < $GPU_MEMORY_THRESHOLD_MIB MiB"
      (( used < GPU_MEMORY_THRESHOLD_MIB )) || all_ready=0
    done
    (( all_ready )) && return
    echo "[gpu] $phase: waiting ${GPU_WAIT_INTERVAL_SECONDS}s"
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}

run_training() {
  local config="$1" exp="$2" checkpoint_base="$3" resume_mode="$4"
  local -a command=(uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE"
    scripts/train_pytorch.py "$config" --exp-name "$exp" --checkpoint-base-dir "$checkpoint_base"
    --num-train-steps "$NUM_TRAIN_STEPS" --batch-size "$BATCH_SIZE" --save-interval "$SAVE_INTERVAL"
    --keep-period "$KEEP_PERIOD" --fsdp-devices "$NPROC_PER_NODE")
  [[ "$resume_mode" == 1 ]] && command+=(--resume) || command+=(--overwrite)
  if [[ "$DRY_RUN" == 1 ]]; then printf '[dry-run]'; printf ' %q' "${command[@]}"; printf '\n'; return; fi
  wait_for_gpus "$config training"
  mkdir -p "$checkpoint_base"
  "${command[@]}"
}

check_inputs
run_training "$ANYTOUCH_CONFIG_NAME" "$ANYTOUCH_EXP_NAME" "$CHECKPOINT_BASE_DIR" "$ANYTOUCH_RESUME"
run_training "$TREX_CONFIG_NAME" "$TREX_EXP_NAME" "$TREX_CHECKPOINT_BASE_DIR" "$TREX_RESUME"
