#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
CONFIG_NAME="${CONFIG_NAME:-pi05_insert_block_v1_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft}"
EXP_NAME="${EXP_NAME:-insert_block_v1_anytouch2_tactile_torch_full_ft_v1}"
REPO_ID="${REPO_ID:-simulation/insert_block_v1_gelsight_abs_joint}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/run/ti/pi-training-data/lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
NORM_STATS_FILE="${NORM_STATS_FILE:-${REPO_ROOT}/assets/pi05_insert_block_v1_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft/${REPO_ID}/norm_stats.json}"

ANYTOUCH_CHECKPOINT_PATH="${ANYTOUCH_CHECKPOINT_PATH:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/anytouch2/checkpoint-4frames.pth}"
PYTORCH_BASE_PATH="${PYTORCH_BASE_PATH:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}"

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
  RUNNER_SNAPSHOT="$(mktemp "${SNAPSHOT_DIR}/train_insert_block_v1_anytouch2.XXXXXX.sh")"
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
)
if [[ "$RESUME" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
else
  TRAIN_ARGS+=(--overwrite)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] dataset=${OUTPUT_DIR}"
  echo "[dry-run] norm_stats=${NORM_STATS_FILE}"
  echo "[dry-run] GPUs=${GPU_IDS}, processes=${NPROC_PER_NODE}, batch=${BATCH_SIZE}, steps=${NUM_TRAIN_STEPS}"
  printf '[dry-run] command:'
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

for required_file in \
  "${OUTPUT_DIR}/meta/info.json" \
  "$NORM_STATS_FILE" \
  "$ANYTOUCH_CHECKPOINT_PATH" \
  "$PYTORCH_BASE_PATH"; do
  if [[ ! -f "$required_file" ]]; then
    echo "[error] required file not found: ${required_file}" >&2
    exit 1
  fi
done

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
print("[dataset] compatible single-frame tactile LeRobot dataset:", root)
PY

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -ne "$NPROC_PER_NODE" ]]; then
  echo "[error] GPU_IDS contains ${#GPU_ARRAY[@]} devices but NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
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
