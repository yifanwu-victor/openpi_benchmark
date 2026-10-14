#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
TASK_NAME="${TASK_NAME:-can_empty_select_v3}"
HDF5_DIR="${HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/can_empty_select_v3}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/run/ti/pi-training-data/lerobot}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/run/ti/hf_cache}"
HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
REPO_ID="${REPO_ID:-simulation/can_empty_select_v3_gelsight_abs_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-20000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-30}"
SKIP_CONVERT="${SKIP_CONVERT:-auto}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}"
TREX_CHECKPOINT_BASE_DIR="${TREX_CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints/t-rex}"
RANDOM_CHECKPOINT_BASE_DIR="${RANDOM_CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints/random_anytouch}"
ANYTOUCH_CHECKPOINT_FILE="${ANYTOUCH_CHECKPOINT_FILE:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/anytouch2/checkpoint-4frames.pth}"
PYTORCH_BASE_FILE="${PYTORCH_BASE_FILE:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors}"

VISION_CONFIG_NAME="${VISION_CONFIG_NAME:-pi05_can_empty_select_v3_sim_abs_joint_relative_action_vision_full_ft}"
VISION_EXP_NAME="${VISION_EXP_NAME:-can_empty_select_v3_abs_joint_vision_full_ft_v1}"
TACTILE_CONFIG_NAME="${TACTILE_CONFIG_NAME:-pi05_can_empty_select_v3_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft}"
TACTILE_EXP_NAME="${TACTILE_EXP_NAME:-can_empty_select_v3_abs_joint_vision_tactile_dual_siglip_full_ft_v1}"
ANYTOUCH_CONFIG_NAME="${ANYTOUCH_CONFIG_NAME:-pi05_can_empty_select_v3_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft}"
ANYTOUCH_EXP_NAME="${ANYTOUCH_EXP_NAME:-can_empty_select_v3_anytouch2_tactile_torch_full_ft_v1}"
TREX_CONFIG_NAME="${TREX_CONFIG_NAME:-pi05_can_empty_select_v3_sim_abs_joint_relative_action_vision_tactile_anytouch2_tactile_expert_torch_full_ft}"
TREX_EXP_NAME="${TREX_EXP_NAME:-can_empty_select_v3_anytouch2_tactile_expert_torch_full_ft_v1}"
RANDOM_CONFIG_NAME="${RANDOM_CONFIG_NAME:-pi05_can_empty_select_v3_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft}"
RANDOM_EXP_NAME="${RANDOM_EXP_NAME:-can_empty_select_v3_random_anytouch2_full_ft_v1}"
ANYTOUCH_RESUME="${ANYTOUCH_RESUME:-auto}"
TREX_RESUME="${TREX_RESUME:-0}"
RANDOM_RESUME="${RANDOM_RESUME:-0}"

if [[ "${OPENPI_RUNNER_SNAPSHOT:-0}" != "1" && "$DRY_RUN" != "1" ]]; then
  SNAPSHOT_DIR="${RUNNER_SNAPSHOT_DIR:-/run/ti/openpi-script-snapshots}"
  mkdir -p "$SNAPSHOT_DIR"
  RUNNER_SNAPSHOT="$(mktemp "${SNAPSHOT_DIR}/train_${TASK_NAME}.XXXXXX.sh")"
  cp "$0" "$RUNNER_SNAPSHOT"
  trap 'rm -f "$RUNNER_SNAPSHOT"' EXIT
  export OPENPI_RUNNER_SNAPSHOT=1
  bash "$RUNNER_SNAPSHOT"
  exit $?
fi

cd "$REPO_ROOT"
unset VIRTUAL_ENV
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
export HF_LEROBOT_HOME HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE CUDA_VISIBLE_DEVICES="$GPU_IDS"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
TRITON_PTXAS_DIR="${TRITON_PTXAS_DIR:-${REPO_ROOT}/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin}"
if [[ -x "$TRITON_PTXAS_DIR/ptxas" ]]; then
  export PATH="$TRITON_PTXAS_DIR:$PATH"
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -ne "$NPROC_PER_NODE" || "$NPROC_PER_NODE" -ne "$FSDP_DEVICES" ]]; then
  echo "[error] GPU_IDS, NPROC_PER_NODE and FSDP_DEVICES must all describe ${FSDP_DEVICES} devices" >&2
  exit 1
fi
if (( BATCH_SIZE % NPROC_PER_NODE != 0 )); then
  echo "[error] BATCH_SIZE=${BATCH_SIZE} must be divisible by NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi

wait_for_gpus() {
  local phase="$1"
  local gpu_id used_memory all_ready
  while true; do
    all_ready=1
    for gpu_id in "${GPU_ARRAY[@]}"; do
      gpu_id="${gpu_id//[[:space:]]/}"
      used_memory="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits)"
      used_memory="${used_memory//[[:space:]]/}"
      if [[ ! "$used_memory" =~ ^[0-9]+$ ]]; then
        echo "[error] invalid memory value for GPU ${gpu_id}: ${used_memory}" >&2
        exit 1
      fi
      echo "[gpu] ${phase}: GPU ${gpu_id} uses ${used_memory} MiB; requires < ${GPU_MEMORY_THRESHOLD_MIB} MiB"
      if (( used_memory >= GPU_MEMORY_THRESHOLD_MIB )); then
        all_ready=0
      fi
    done
    if (( all_ready )); then
      return
    fi
    echo "[gpu] ${phase}: waiting ${GPU_WAIT_INTERVAL_SECONDS}s"
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}

stats_file() {
  printf '%s/assets/%s/%s/norm_stats.json' "$REPO_ROOT" "$1" "$REPO_ID"
}

run_norm_stats() {
  local config_name="$1"
  local output_file
  output_file="$(stats_file "$config_name")"
  if [[ "$FORCE_NORM_STATS" == "1" || ! -f "$output_file" ]]; then
    wait_for_gpus "norm stats ${config_name}"
    echo "[norm-stats] computing ${config_name}"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" uv run scripts/compute_norm_stats.py --config-name "$config_name"
  else
    echo "[norm-stats] reusing ${output_file}"
  fi
  if [[ ! -f "$output_file" ]]; then
    echo "[error] norm stats not found after processing: ${output_file}" >&2
    exit 1
  fi
}

run_jax_training() {
  local config_name="$1" exp_name="$2" resume_mode="${3:-$RESUME}"
  local -a args=(uv run scripts/train.py "$config_name" --exp-name "$exp_name"
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" --num-train-steps "$NUM_TRAIN_STEPS"
    --batch-size "$BATCH_SIZE" --save-interval "$SAVE_INTERVAL" --keep-period "$KEEP_PERIOD"
    --fsdp-devices "$FSDP_DEVICES")
  wait_for_gpus "${config_name} training"
  if [[ "$resume_mode" == "1" ]]; then args+=(--resume); else args+=(--overwrite); fi
  echo "[train] ${config_name} on GPUs ${GPU_IDS}"
  printf '[train] command:'; printf ' %q' "${args[@]}"; printf '\n'
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "${args[@]}"
}

run_torch_training() {
  local config_name="$1" exp_name="$2" checkpoint_base_dir="$3" resume_mode="${4:-$RESUME}"
  local -a args=(uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE"
    scripts/train_pytorch.py "$config_name" --exp-name "$exp_name"
    --checkpoint-base-dir "$checkpoint_base_dir" --num-train-steps "$NUM_TRAIN_STEPS"
    --batch-size "$BATCH_SIZE" --save-interval "$SAVE_INTERVAL" --keep-period "$KEEP_PERIOD"
    --fsdp-devices "$FSDP_DEVICES")
  wait_for_gpus "${config_name} training"
  local checkpoint_dir="${checkpoint_base_dir}/${config_name}/${exp_name}"
  local has_checkpoint=0
  if [[ -d "$checkpoint_dir" ]] && find "$checkpoint_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | grep -Eq '^[0-9]+$'; then
    has_checkpoint=1
  fi
  if [[ "$resume_mode" == "1" || ("$resume_mode" == "auto" && "$has_checkpoint" == "1") ]]; then
    args+=(--resume)
    echo "[train] resuming ${config_name} from ${checkpoint_dir}"
  else
    args+=(--overwrite)
  fi
  echo "[train] ${config_name} on GPUs ${GPU_IDS}"
  printf '[train] command:'; printf ' %q' "${args[@]}"; printf '\n'
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "${args[@]}"
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] raw data: ${HDF5_DIR}"
  echo "[dry-run] dataset: ${OUTPUT_DIR} (${REPO_ID})"
  echo "[dry-run] conversion: abs_joint, left gripper, 224x224, frame stride 1, target offset 1"
  echo "[dry-run] GPUs: ${GPU_IDS}; processes/FSDP: ${NPROC_PER_NODE}; batch: ${BATCH_SIZE}; steps: ${NUM_TRAIN_STEPS}"
  echo "[dry-run] order: vision -> vision+tactile dual-SigLIP -> AnyTouch2 -> AnyTouch2 tactile expert -> random AnyTouch2"
  echo "[dry-run] checkpoint roots: ${CHECKPOINT_BASE_DIR}, ${TREX_CHECKPOINT_BASE_DIR}, ${RANDOM_CHECKPOINT_BASE_DIR}"
  exit 0
fi

if [[ ! -d "$HDF5_DIR" ]]; then
  echo "[error] raw dataset directory not found: ${HDF5_DIR}" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[error] nvidia-smi not found" >&2
  exit 1
fi
if [[ "$SKIP_CONVERT" == "1" ]]; then
  echo "[convert] skipped (SKIP_CONVERT=1)"
elif [[ "$FORCE_CONVERT" == "1" || ! -f "${OUTPUT_DIR}/meta/info.json" ]]; then
  mkdir -p "$(dirname "$OUTPUT_DIR")"
  CONVERT_ARGS=(uv run examples/UniVTAC/convert_sim_to_lerobot.py
    --hdf5-dir "$HDF5_DIR" --output-dir "$OUTPUT_DIR" --repo-id "$REPO_ID"
    --control-mode abs_joint --gripper-mode left --image-size 224 224
    --frame-stride 1 --target-offset 1 --overwrite)
  if [[ -n "$MAX_EPISODES" ]]; then CONVERT_ARGS+=(--max-episodes "$MAX_EPISODES"); fi
  echo "[convert] ${HDF5_DIR} -> ${OUTPUT_DIR}"
  "${CONVERT_ARGS[@]}"
else
  echo "[convert] reusing ${OUTPUT_DIR}"
fi

if [[ ! -f "${OUTPUT_DIR}/meta/info.json" ]]; then
  echo "[error] LeRobot metadata not found: ${OUTPUT_DIR}/meta/info.json" >&2
  exit 1
fi
uv run python - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
features = json.loads((root / "meta" / "info.json").read_text())["features"]
expected = {
    "state": (8,),
    "actions": (8,),
    "image": (224, 224, 3),
    "wrist_image": (224, 224, 3),
    "left_tactile_image": (224, 224, 3),
    "right_tactile_image": (224, 224, 3),
}
for key, shape in expected.items():
    actual = tuple(features.get(key, {}).get("shape", ()))
    if actual != shape:
        raise SystemExit(f"Unexpected {key} shape: expected {shape}, got {actual}")
print(f"[dataset] compatible dataset: {root}")
PY

run_norm_stats "$VISION_CONFIG_NAME"
run_norm_stats "$TACTILE_CONFIG_NAME"

if [[ ! -f "$PYTORCH_BASE_FILE" ]]; then
  echo "[error] Pi05 PyTorch base not found: ${PYTORCH_BASE_FILE}" >&2
  exit 1
fi
if [[ ! -f "$ANYTOUCH_CHECKPOINT_FILE" ]]; then
  echo "[error] AnyTouch2 checkpoint not found: ${ANYTOUCH_CHECKPOINT_FILE}" >&2
  exit 1
fi

run_jax_training "$VISION_CONFIG_NAME" "$VISION_EXP_NAME" "${VISION_RESUME:-$RESUME}"
run_jax_training "$TACTILE_CONFIG_NAME" "$TACTILE_EXP_NAME" "${TACTILE_RESUME:-$RESUME}"
run_torch_training "$ANYTOUCH_CONFIG_NAME" "$ANYTOUCH_EXP_NAME" "$CHECKPOINT_BASE_DIR" "$ANYTOUCH_RESUME"
run_torch_training "$TREX_CONFIG_NAME" "$TREX_EXP_NAME" "$TREX_CHECKPOINT_BASE_DIR" "$TREX_RESUME"
# run_torch_training "$RANDOM_CONFIG_NAME" "$RANDOM_EXP_NAME" "$RANDOM_CHECKPOINT_BASE_DIR" "$RANDOM_RESUME"

echo "[done] ${TASK_NAME} five-stage training completed"
