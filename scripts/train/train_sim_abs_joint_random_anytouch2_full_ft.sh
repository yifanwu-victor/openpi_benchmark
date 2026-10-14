#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
: "${CONFIG_NAME:?CONFIG_NAME must be set by the task launcher}"
: "${EXP_NAME:?EXP_NAME must be set by the task launcher}"
: "${REPO_ID:?REPO_ID must be set by the task launcher}"
: "${HDF5_DIR:?HDF5_DIR must be set by the task launcher}"

HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/run/ti/pi-training-data/lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
NORM_STATS_FILE="${NORM_STATS_FILE:-${REPO_ROOT}/assets/${CONFIG_NAME}/${REPO_ID}/norm_stats.json}"
NORM_STATS_SOURCE_FILE="${NORM_STATS_SOURCE_FILE:-}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints/random_anytouch}"
PYTORCH_BASE_FILE="${PYTORCH_BASE_FILE:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
GPU_MEMORY_THRESHOLD_MIB="${GPU_MEMORY_THRESHOLD_MIB:-20000}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-30}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

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
  scripts/train_pytorch.py "$CONFIG_NAME" --exp-name "$EXP_NAME"
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" --num-train-steps "$NUM_TRAIN_STEPS"
  --batch-size "$BATCH_SIZE" --save-interval "$SAVE_INTERVAL" --keep-period "$KEEP_PERIOD"
  --fsdp-devices "$NPROC_PER_NODE"
)
if [[ "$RESUME" == "1" ]]; then TRAIN_ARGS+=(--resume); else TRAIN_ARGS+=(--overwrite); fi

if [[ "$DRY_RUN" != "1" && "${OPENPI_RUNNER_SNAPSHOT:-0}" != "1" ]]; then
  SNAPSHOT_DIR="${RUNNER_SNAPSHOT_DIR:-/run/ti/openpi-script-snapshots}"
  mkdir -p "$SNAPSHOT_DIR"
  RUNNER_SNAPSHOT="$(mktemp "${SNAPSHOT_DIR}/train_random_anytouch2.XXXXXX.sh")"
  cp "$0" "$RUNNER_SNAPSHOT"
  trap 'rm -f "$RUNNER_SNAPSHOT"' EXIT
  export OPENPI_RUNNER_SNAPSHOT=1
  bash "$RUNNER_SNAPSHOT"
  exit $?
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] dataset=${OUTPUT_DIR}"
  echo "[dry-run] raw_data=${HDF5_DIR}"
  echo "[dry-run] norm_stats=${NORM_STATS_FILE}"
  echo "[dry-run] GPUs=${GPU_IDS}, processes=${NPROC_PER_NODE}, batch=${BATCH_SIZE}, steps=${NUM_TRAIN_STEPS}"
  echo "[dry-run] checkpoint_base_dir=${CHECKPOINT_BASE_DIR}"
  echo "[dry-run] conversion=$([[ "$SKIP_CONVERT" == 1 ]] && echo skipped || echo auto)"
  printf '[dry-run] command:'
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_BASE_DIR" "$(dirname "$NORM_STATS_FILE")"
if [[ ! -f "$PYTORCH_BASE_FILE" ]]; then
  echo "[error] Pi05 PyTorch base not found: ${PYTORCH_BASE_FILE}" >&2
  exit 1
fi
if [[ "$SKIP_CONVERT" == "1" ]]; then
  echo "[convert] skipped; using ${OUTPUT_DIR}"
elif [[ "$FORCE_CONVERT" == "1" || ! -f "${OUTPUT_DIR}/meta/info.json" ]]; then
  CONVERT_ARGS=(
    uv run examples/anytouch2/convert_univtac_sim_delta_eef_to_lerobot.py
    --hdf5-dir "$HDF5_DIR" --output-dir "$OUTPUT_DIR" --repo-id "$REPO_ID"
    --control-mode abs_joint --frame-stride 1 --target-offset 1 --gripper-mode left
    --image-size 224 224 --overwrite
  )
  if [[ -n "$MAX_EPISODES" ]]; then CONVERT_ARGS+=(--max-episodes "$MAX_EPISODES"); fi
  echo "[convert] ${HDF5_DIR} -> ${OUTPUT_DIR}"
  "${CONVERT_ARGS[@]}"
else
  echo "[convert] found ${OUTPUT_DIR}/meta/info.json; reusing existing dataset"
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
print(f"[dataset] compatible abs-joint AnyTouch dataset: {root}")
PY

if [[ "$FORCE_NORM_STATS" == "1" || ! -f "$NORM_STATS_FILE" ]]; then
  if [[ "$FORCE_NORM_STATS" != "1" && -n "$NORM_STATS_SOURCE_FILE" && -f "$NORM_STATS_SOURCE_FILE" ]]; then
    echo "[norm-stats] copying compatible stats ${NORM_STATS_SOURCE_FILE} -> ${NORM_STATS_FILE}"
    cp "$NORM_STATS_SOURCE_FILE" "$NORM_STATS_FILE"
  else
    echo "[norm-stats] computing stats for ${CONFIG_NAME}"
    uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
  fi
else
  echo "[norm-stats] found ${NORM_STATS_FILE}; reusing existing stats"
fi
if [[ ! -f "$NORM_STATS_FILE" ]]; then
  echo "[error] norm stats not found after processing: ${NORM_STATS_FILE}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -ne "$NPROC_PER_NODE" ]]; then
  echo "[error] GPU_IDS contains ${#GPU_ARRAY[@]} devices but NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi
if (( BATCH_SIZE % NPROC_PER_NODE != 0 )); then
  echo "[error] BATCH_SIZE=${BATCH_SIZE} must be divisible by NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[error] nvidia-smi not found" >&2
  exit 1
fi
while true; do
  ALL_READY=1
  for GPU_ID in "${GPU_ARRAY[@]}"; do
    USED_MEMORY="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits)"
    USED_MEMORY="${USED_MEMORY//[[:space:]]/}"
    echo "[gpu] ${GPU_ID}: ${USED_MEMORY} MiB used; requires < ${GPU_MEMORY_THRESHOLD_MIB} MiB"
    if (( USED_MEMORY >= GPU_MEMORY_THRESHOLD_MIB )); then ALL_READY=0; fi
  done
  if (( ALL_READY )); then break; fi
  echo "[gpu] waiting ${GPU_WAIT_INTERVAL_SECONDS}s"
  sleep "$GPU_WAIT_INTERVAL_SECONDS"
done

echo "[train] config=${CONFIG_NAME}"
echo "[train] exp=${EXP_NAME}"
echo "[train] GPUs=${GPU_IDS}, processes=${NPROC_PER_NODE}, batch=${BATCH_SIZE}, steps=${NUM_TRAIN_STEPS}"
echo "[train] checkpoint_base_dir=${CHECKPOINT_BASE_DIR}"
printf '[train] command:'
printf ' %q' "${TRAIN_ARGS[@]}"
printf '\n'
"${TRAIN_ARGS[@]}"
