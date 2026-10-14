#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/run/ti/pi-training-data/lerobot}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-20000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-30}"

FORCE_CONVERT="${FORCE_CONVERT:-0}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
RESUME="${RESUME:-0}"
TURN_GEAR_PAIR_RESUME="${TURN_GEAR_PAIR_RESUME:-${RESUME}}"
WEIGHT_CLASSIFY_RESUME="${WEIGHT_CLASSIFY_RESUME:-${RESUME}}"
DRY_RUN="${DRY_RUN:-0}"

ANYTOUCH_CHECKPOINT_PATH="${ANYTOUCH_CHECKPOINT_PATH:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/anytouch2/checkpoint-4frames.pth}"
PYTORCH_BASE_PATH="${PYTORCH_BASE_PATH:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors}"

TURN_RAW_DIR="${TURN_RAW_DIR:-/home/tione/notebook/data/UniVTAC/sim/turn_gear_pair}"
TURN_TASK_LABEL="${TURN_TASK_LABEL:-turn_gear_pair}"
TURN_DISPLAY_NAME="${TURN_DISPLAY_NAME:-Turn Gear Pair}"
TURN_REPO_ID="${TURN_REPO_ID:-simulation/turn_gear_pair_gelsight_abs_joint}"
TURN_OUTPUT_DIR="${TURN_OUTPUT_DIR:-${HF_LEROBOT_HOME}/${TURN_REPO_ID}}"
TURN_CONFIG_NAME="${TURN_CONFIG_NAME:-pi05_turn_gear_pair_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft}"
TURN_NORM_CONFIG_NAME="${TURN_NORM_CONFIG_NAME:-pi05_turn_gear_pair_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora}"
TURN_NORM_STATS_FILE="${TURN_NORM_STATS_FILE:-${REPO_ROOT}/assets/${TURN_NORM_CONFIG_NAME}/${TURN_REPO_ID}/norm_stats.json}"
TURN_EXP_NAME="${TURN_EXP_NAME:-turn_gear_pair_anytouch2_tactile_torch_full_ft_v1}"

WEIGHT_RAW_DIR="${WEIGHT_RAW_DIR:-/home/tione/notebook/data/UniVTAC/sim/weight_classify}"
WEIGHT_TASK_LABEL="${WEIGHT_TASK_LABEL:-weight_classify}"
WEIGHT_DISPLAY_NAME="${WEIGHT_DISPLAY_NAME:-Weight Classify}"
WEIGHT_REPO_ID="${WEIGHT_REPO_ID:-simulation/weight_classify_gelsight_abs_joint}"
WEIGHT_OUTPUT_DIR="${WEIGHT_OUTPUT_DIR:-${HF_LEROBOT_HOME}/${WEIGHT_REPO_ID}}"
WEIGHT_CONFIG_NAME="${WEIGHT_CONFIG_NAME:-pi05_weight_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft}"
WEIGHT_NORM_CONFIG_NAME="${WEIGHT_NORM_CONFIG_NAME:-${WEIGHT_CONFIG_NAME}}"
WEIGHT_NORM_STATS_FILE="${WEIGHT_NORM_STATS_FILE:-${REPO_ROOT}/assets/${WEIGHT_NORM_CONFIG_NAME}/${WEIGHT_REPO_ID}/norm_stats.json}"
WEIGHT_EXP_NAME="${WEIGHT_EXP_NAME:-weight_classify_anytouch2_tactile_torch_full_ft_v1}"

if [[ "${OPENPI_RUNNER_SNAPSHOT:-0}" != "1" && "$DRY_RUN" != "1" ]]; then
  SNAPSHOT_DIR="${RUNNER_SNAPSHOT_DIR:-/run/ti/openpi-script-snapshots}"
  mkdir -p "$SNAPSHOT_DIR"
  RUNNER_SNAPSHOT="$(mktemp "${SNAPSHOT_DIR}/train_turn_then_weight_anytouch2.XXXXXX.sh")"
  cp "$0" "$RUNNER_SNAPSHOT"
  trap 'rm -f "$RUNNER_SNAPSHOT"' EXIT
  export OPENPI_RUNNER_SNAPSHOT=1
  bash "$RUNNER_SNAPSHOT"
  exit $?
fi

cd "$REPO_ROOT"
unset VIRTUAL_ENV
export HF_LEROBOT_HOME
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
TRITON_PTXAS_DIR="${TRITON_PTXAS_DIR:-${REPO_ROOT}/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin}"
if [[ -x "$TRITON_PTXAS_DIR/ptxas" ]]; then
  export PATH="$TRITON_PTXAS_DIR:$PATH"
fi

wait_for_gpus() {
  local phase="$1"
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

ensure_dataset() {
  local label="$1"
  local raw_dir="$2"
  local output_dir="$3"
  local repo_id="$4"
  local marker="${output_dir}/meta/info.json"

  if [[ "$FORCE_CONVERT" != "1" && -f "$marker" ]]; then
    echo "[dataset] ${label}: reusing ${output_dir}"
  else
    echo "[dataset] ${label}: converting ${raw_dir} -> ${output_dir}"
    uv run examples/UniVTAC/convert_sim_to_lerobot.py \
      --hdf5-dir "$raw_dir" \
      --output-dir "$output_dir" \
      --repo-id "$repo_id" \
      --control-mode abs_joint \
      --gripper-mode left \
      --image-size 224 224 \
      --frame-stride 1 \
      --target-offset 1 \
      --overwrite
  fi

  uv run python - "$output_dir" <<'PY'
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
print("[dataset] compatible:", root)
PY
}

ensure_norm_stats() {
  local label="$1"
  local config_name="$2"
  local norm_stats_file="$3"

  if [[ "$FORCE_NORM_STATS" != "1" && -f "$norm_stats_file" ]]; then
    echo "[norm-stats] ${label}: reusing ${norm_stats_file}"
    return
  fi
  wait_for_gpus "${label} norm stats"
  echo "[norm-stats] ${label}: computing with ${config_name}"
  uv run scripts/compute_norm_stats.py --config-name "$config_name"
  if [[ ! -f "$norm_stats_file" ]]; then
    echo "[error] norm stats were not written to ${norm_stats_file}" >&2
    exit 1
  fi
}

train_task() {
  local label="$1"
  local config_name="$2"
  local exp_name="$3"
  local resume="$4"
  local -a train_args=(
    uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE"
    scripts/train_pytorch.py "$config_name"
    --exp-name "$exp_name"
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
    --num-train-steps "$NUM_TRAIN_STEPS"
    --batch-size "$BATCH_SIZE"
    --save-interval "$SAVE_INTERVAL"
    --keep-period "$KEEP_PERIOD"
  )
  if [[ "$resume" == "1" ]]; then
    train_args+=(--resume)
  else
    train_args+=(--overwrite)
  fi

  wait_for_gpus "${label} training"
  printf '[train] %s:' "$label"
  printf ' %q' "${train_args[@]}"
  printf '\n'
  "${train_args[@]}"
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] order: ${TURN_TASK_LABEL} -> ${WEIGHT_TASK_LABEL}"
  echo "[dry-run] GPUs=${GPU_IDS}, processes=${NPROC_PER_NODE}, batch=${BATCH_SIZE}, steps=${NUM_TRAIN_STEPS}"
  echo "[dry-run] turn dataset=${TURN_OUTPUT_DIR}"
  echo "[dry-run] turn norm stats=${TURN_NORM_STATS_FILE}"
  echo "[dry-run] turn config=${TURN_CONFIG_NAME}, exp=${TURN_EXP_NAME}"
  echo "[dry-run] weight dataset=${WEIGHT_OUTPUT_DIR}"
  echo "[dry-run] weight norm stats=${WEIGHT_NORM_STATS_FILE}"
  echo "[dry-run] weight config=${WEIGHT_CONFIG_NAME}, exp=${WEIGHT_EXP_NAME}"
  echo "[dry-run] missing datasets and norm stats will be generated before training"
  exit 0
fi

for required_file in "$ANYTOUCH_CHECKPOINT_PATH" "$PYTORCH_BASE_PATH"; do
  if [[ ! -f "$required_file" ]]; then
    echo "[error] required file not found: ${required_file}" >&2
    exit 1
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[error] nvidia-smi not found" >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_BASE_DIR" "$(dirname "$TURN_OUTPUT_DIR")" "$(dirname "$WEIGHT_OUTPUT_DIR")"

echo "[1/2] Preparing and training ${TURN_DISPLAY_NAME}"
ensure_dataset "$TURN_TASK_LABEL" "$TURN_RAW_DIR" "$TURN_OUTPUT_DIR" "$TURN_REPO_ID"
ensure_norm_stats "$TURN_TASK_LABEL" "$TURN_NORM_CONFIG_NAME" "$TURN_NORM_STATS_FILE"
train_task "$TURN_TASK_LABEL" "$TURN_CONFIG_NAME" "$TURN_EXP_NAME" "$TURN_GEAR_PAIR_RESUME"

echo "[2/2] Preparing and training ${WEIGHT_DISPLAY_NAME}"
ensure_dataset "$WEIGHT_TASK_LABEL" "$WEIGHT_RAW_DIR" "$WEIGHT_OUTPUT_DIR" "$WEIGHT_REPO_ID"
ensure_norm_stats "$WEIGHT_TASK_LABEL" "$WEIGHT_NORM_CONFIG_NAME" "$WEIGHT_NORM_STATS_FILE"
train_task "$WEIGHT_TASK_LABEL" "$WEIGHT_CONFIG_NAME" "$WEIGHT_EXP_NAME" "$WEIGHT_CLASSIFY_RESUME"

echo "[done] ${TURN_DISPLAY_NAME} and ${WEIGHT_DISPLAY_NAME} AnyTouch2 training finished"
