#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
HDF5_DIR="${HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/turn_gear_pair_3cm}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/run/ti/pi-training-data/lerobot}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/run/ti/hf_cache}"
HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
REPO_ID="${REPO_ID:-simulation/turn_gear_pair_3cm_gelsight_abs_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-$HF_LEROBOT_HOME/$REPO_ID}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-20000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-30}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
SKIP_NORM_STATS="${SKIP_NORM_STATS:-1}"
MAX_EPISODES="${MAX_EPISODES:-}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$REPO_ROOT/checkpoints}"
TREX_CHECKPOINT_BASE_DIR="${TREX_CHECKPOINT_BASE_DIR:-$REPO_ROOT/checkpoints/t-rex}"
RANDOM_CHECKPOINT_BASE_DIR="${RANDOM_CHECKPOINT_BASE_DIR:-$REPO_ROOT/checkpoints/random_anytouch}"

VISION_CONFIG_NAME="${VISION_CONFIG_NAME:-pi05_turn_gear_pair_3cm_sim_abs_joint_relative_action_vision_full_ft}"
VISION_EXP_NAME="${VISION_EXP_NAME:-turn_gear_pair_3cm_abs_joint_vision_full_ft_v1}"
TACTILE_CONFIG_NAME="${TACTILE_CONFIG_NAME:-pi05_turn_gear_pair_3cm_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft}"
TACTILE_EXP_NAME="${TACTILE_EXP_NAME:-turn_gear_pair_3cm_abs_joint_vision_tactile_dual_siglip_full_ft_v1}"
ANYTOUCH_CONFIG_NAME="${ANYTOUCH_CONFIG_NAME:-pi05_turn_gear_pair_3cm_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft}"
ANYTOUCH_EXP_NAME="${ANYTOUCH_EXP_NAME:-turn_gear_pair_3cm_anytouch2_tactile_torch_full_ft_v1}"
TREX_CONFIG_NAME="${TREX_CONFIG_NAME:-pi05_turn_gear_pair_3cm_sim_abs_joint_relative_action_vision_tactile_anytouch2_tactile_expert_torch_full_ft}"
TREX_EXP_NAME="${TREX_EXP_NAME:-turn_gear_pair_3cm_anytouch2_tactile_expert_torch_full_ft_v1}"
RANDOM_CONFIG_NAME="${RANDOM_CONFIG_NAME:-pi05_turn_gear_pair_3cm_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft}"
RANDOM_EXP_NAME="${RANDOM_EXP_NAME:-turn_gear_pair_3cm_random_anytouch2_full_ft_v1}"

cd "$REPO_ROOT"
unset VIRTUAL_ENV
export HF_LEROBOT_HOME HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE CUDA_VISIBLE_DEVICES="$GPU_IDS"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if (( ${#GPU_ARRAY[@]} != NPROC_PER_NODE || NPROC_PER_NODE != FSDP_DEVICES )); then
  echo "[error] GPU_IDS, NPROC_PER_NODE and FSDP_DEVICES must have matching counts" >&2; exit 1
fi

wait_for_gpus() {
  local phase="$1" id used ready
  while true; do
    ready=1
    for id in "${GPU_ARRAY[@]}"; do
      id="${id//[[:space:]]/}"
      used="$(nvidia-smi --id="$id" --query-gpu=memory.used --format=csv,noheader,nounits)"
      used="${used//[[:space:]]/}"
      [[ "$used" =~ ^[0-9]+$ ]] || { echo "[error] invalid GPU memory: $used" >&2; exit 1; }
      echo "[gpu] $phase: GPU $id uses ${used} MiB; requires < ${GPU_MEMORY_THRESHOLD_MIB} MiB"
      (( used < GPU_MEMORY_THRESHOLD_MIB )) || ready=0
    done
    (( ready )) && return
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}
stats_file() { printf '%s/assets/%s/%s/norm_stats.json' "$REPO_ROOT" "$1" "$REPO_ID"; }
run_stats() {
  local config="$1" out; out="$(stats_file "$config")"
  if [[ "$SKIP_NORM_STATS" == 1 && "$FORCE_NORM_STATS" != 1 ]]; then
    [[ -f "$out" ]] || { echo "[error] required norm stats missing: $out" >&2; exit 1; }
    echo "[norm-stats] reusing $out"
    return
  fi
  if [[ "$FORCE_NORM_STATS" == 1 || ! -f "$out" ]]; then
    wait_for_gpus "norm stats $config"
    uv run scripts/compute_norm_stats.py --config-name "$config"
  fi
  [[ -f "$out" ]] || { echo "[error] missing norm stats: $out" >&2; exit 1; }
}
run_jax() {
  local config="$1" exp="$2" mode="${3:-$RESUME}"; wait_for_gpus "$config training"
  local -a cmd=(uv run scripts/train.py "$config" --exp-name "$exp" --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" --num-train-steps "$NUM_TRAIN_STEPS" --batch-size "$BATCH_SIZE" --save-interval "$SAVE_INTERVAL" --keep-period "$KEEP_PERIOD" --fsdp-devices "$FSDP_DEVICES")
  local checkpoint_dir="$CHECKPOINT_BASE_DIR/$config/$exp"
  if [[ "$mode" == 1 || ( "$mode" == auto && -d "$checkpoint_dir/5000" ) ]]; then
    cmd+=(--resume)
  else
    cmd+=(--overwrite)
  fi
  "${cmd[@]}"
}
run_torch() {
  local config="$1" exp="$2" base="$3" mode="${4:-$RESUME}"; wait_for_gpus "$config training"
  local -a cmd=(uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" scripts/train_pytorch.py "$config" --exp-name "$exp" --checkpoint-base-dir "$base" --num-train-steps "$NUM_TRAIN_STEPS" --batch-size "$BATCH_SIZE" --save-interval "$SAVE_INTERVAL" --keep-period "$KEEP_PERIOD" --fsdp-devices "$FSDP_DEVICES")
  [[ "$mode" == 1 ]] && cmd+=(--resume) || cmd+=(--overwrite); "${cmd[@]}"
}
if [[ "$DRY_RUN" == 1 ]]; then
  echo "[dry-run] raw data: $HDF5_DIR"
  echo "[dry-run] dataset: $OUTPUT_DIR ($REPO_ID)"
  echo "[dry-run] GPUs: $GPU_IDS; batch: $BATCH_SIZE; steps: $NUM_TRAIN_STEPS"
  echo "[dry-run] order: vision -> dual-SigLIP -> AnyTouch2 -> T-Rex -> random AnyTouch2"
  exit 0
fi
[[ -d "$HDF5_DIR" ]] || { echo "[error] raw dataset not found: $HDF5_DIR" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "[error] nvidia-smi not found" >&2; exit 1; }
if [[ "$FORCE_CONVERT" == 1 || ! -f "$OUTPUT_DIR/meta/info.json" ]]; then
  args=(uv run examples/UniVTAC/convert_sim_to_lerobot.py --hdf5-dir "$HDF5_DIR" --output-dir "$OUTPUT_DIR" --repo-id "$REPO_ID" --control-mode abs_joint --gripper-mode left --image-size 224 224 --frame-stride 1 --target-offset 1 --overwrite)
  [[ -z "$MAX_EPISODES" ]] || args+=(--max-episodes "$MAX_EPISODES")
  mkdir -p "$(dirname "$OUTPUT_DIR")"; "${args[@]}"
fi
[[ -f "$OUTPUT_DIR/meta/info.json" ]] || { echo "[error] missing LeRobot metadata: $OUTPUT_DIR/meta/info.json" >&2; exit 1; }
run_stats "$VISION_CONFIG_NAME"
run_stats "$TACTILE_CONFIG_NAME"
run_jax "$VISION_CONFIG_NAME" "$VISION_EXP_NAME" "${VISION_RESUME:-auto}"
run_jax "$TACTILE_CONFIG_NAME" "$TACTILE_EXP_NAME" "${TACTILE_RESUME:-$RESUME}"
run_torch "$ANYTOUCH_CONFIG_NAME" "$ANYTOUCH_EXP_NAME" "$CHECKPOINT_BASE_DIR" "${ANYTOUCH_RESUME:-$RESUME}"
run_torch "$TREX_CONFIG_NAME" "$TREX_EXP_NAME" "$TREX_CHECKPOINT_BASE_DIR" "${TREX_RESUME:-$RESUME}"
run_torch "$RANDOM_CONFIG_NAME" "$RANDOM_EXP_NAME" "$RANDOM_CHECKPOINT_BASE_DIR" "${RANDOM_RESUME:-$RESUME}"
echo "[done] turn_gear_pair_3cm five-stage training completed"
