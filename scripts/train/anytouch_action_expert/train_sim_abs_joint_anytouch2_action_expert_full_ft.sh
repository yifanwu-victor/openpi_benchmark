#!/usr/bin/env bash
set -euo pipefail

# Shared launcher for the AnyTouch2 Action-Expert ablation.  It intentionally
# reuses existing LeRobot datasets and state/action norm statistics whenever
# possible; conversion/stat computation is only a fallback for a new task.
REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
: "${CONFIG_NAME:?CONFIG_NAME must be set by the task wrapper}"
: "${EXP_NAME:?EXP_NAME must be set by the task wrapper}"
: "${REPO_ID:?REPO_ID must be set by the task wrapper}"
: "${HDF5_DIR:?HDF5_DIR must be set by the task wrapper}"

PI_TRAINING_DATA_ROOT="${PI_TRAINING_DATA_ROOT:-/run/ti/pi-training-data}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI_TRAINING_DATA_ROOT}/lerobot}"
HF_HOME="${HF_HOME:-/run/ti/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
NORM_STATS_FILE="${NORM_STATS_FILE:-${REPO_ROOT}/assets/${CONFIG_NAME}/${REPO_ID}/norm_stats.json}"
NORM_STATS_SOURCE_FILE="${NORM_STATS_SOURCE_FILE:-}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints/anytouch_action_expert}"
PYTORCH_BASE_FILE="${PYTORCH_BASE_FILE:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors}"
ANYTOUCH_CHECKPOINT_FILE="${ANYTOUCH_CHECKPOINT_FILE:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/anytouch2/checkpoint-4frames.pth}"
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
export PI_TRAINING_DATA_ROOT HF_LEROBOT_HOME HF_HOME HF_DATASETS_CACHE
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
TRITON_PTXAS_DIR="${TRITON_PTXAS_DIR:-${REPO_ROOT}/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin}"
if [[ -x "$TRITON_PTXAS_DIR/ptxas" ]]; then export PATH="$TRITON_PTXAS_DIR:$PATH"; fi

if [[ "${OPENPI_RUNNER_SNAPSHOT:-0}" != "1" && "$DRY_RUN" != "1" ]]; then
  SNAPSHOT_DIR="${RUNNER_SNAPSHOT_DIR:-/run/ti/openpi-script-snapshots}"
  mkdir -p "$SNAPSHOT_DIR"
  RUNNER_SNAPSHOT="$(mktemp "${SNAPSHOT_DIR}/train_action_expert.XXXXXX.sh")"
  cp "$0" "$RUNNER_SNAPSHOT"
  trap 'rm -f "$RUNNER_SNAPSHOT"' EXIT
  export OPENPI_RUNNER_SNAPSHOT=1
  bash "$RUNNER_SNAPSHOT"
  exit $?
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -ne "$NPROC_PER_NODE" ]]; then
  echo "[error] GPU_IDS contains ${#GPU_ARRAY[@]} devices but NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi
if (( BATCH_SIZE % NPROC_PER_NODE != 0 )); then
  echo "[error] BATCH_SIZE=${BATCH_SIZE} is not divisible by NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi

TRAIN_ARGS=(
  uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE"
  scripts/train_pytorch.py "$CONFIG_NAME" --exp-name "$EXP_NAME"
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" --num-train-steps "$NUM_TRAIN_STEPS"
  --batch-size "$BATCH_SIZE" --save-interval "$SAVE_INTERVAL" --keep-period "$KEEP_PERIOD"
  --fsdp-devices "$NPROC_PER_NODE"
)
if [[ "$RESUME" == "1" ]]; then TRAIN_ARGS+=(--resume); else TRAIN_ARGS+=(--overwrite); fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] raw_data=$HDF5_DIR"
  echo "[dry-run] dataset=$OUTPUT_DIR"
  echo "[dry-run] norm_stats=$NORM_STATS_FILE"
  echo "[dry-run] GPUs=$GPU_IDS processes=$NPROC_PER_NODE batch=$BATCH_SIZE steps=$NUM_TRAIN_STEPS"
  echo "[dry-run] checkpoint_base_dir=$CHECKPOINT_BASE_DIR"
  printf '[dry-run] command:'; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'
  exit 0
fi

for required_file in "$PYTORCH_BASE_FILE" "$ANYTOUCH_CHECKPOINT_FILE"; do
  [[ -f "$required_file" ]] || { echo "[error] required file not found: $required_file" >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_BASE_DIR" "$(dirname "$NORM_STATS_FILE")"

if [[ "$SKIP_CONVERT" == "1" ]]; then
  echo "[convert] skipped (SKIP_CONVERT=1)"
elif [[ "$FORCE_CONVERT" == "1" || ! -f "$OUTPUT_DIR/meta/info.json" ]]; then
  [[ -d "$HDF5_DIR" ]] || { echo "[error] raw dataset directory not found: $HDF5_DIR" >&2; exit 1; }
  CONVERT_ARGS=(
    uv run examples/anytouch2/convert_univtac_sim_delta_eef_to_lerobot.py
    --hdf5-dir "$HDF5_DIR" --output-dir "$OUTPUT_DIR" --repo-id "$REPO_ID"
    --control-mode abs_joint --gripper-mode left --image-size 224 224
    --frame-stride 1 --target-offset 1 --overwrite
  )
  if [[ -n "$MAX_EPISODES" ]]; then CONVERT_ARGS+=(--max-episodes "$MAX_EPISODES"); fi
  printf '[convert]'; printf ' %q' "${CONVERT_ARGS[@]}"; printf '\n'
  "${CONVERT_ARGS[@]}"
else
  echo "[convert] reusing $OUTPUT_DIR/meta/info.json"
fi
[[ -f "$OUTPUT_DIR/meta/info.json" ]] || { echo "[error] missing LeRobot metadata: $OUTPUT_DIR/meta/info.json" >&2; exit 1; }

uv run python - "$OUTPUT_DIR/meta/info.json" <<'PY'
import json, pathlib, sys
features = json.loads(pathlib.Path(sys.argv[1]).read_text())['features']
expected = {
    'state': (8,), 'actions': (8,),
    'image': (224, 224, 3), 'wrist_image': (224, 224, 3),
    'left_tactile_image': (224, 224, 3), 'right_tactile_image': (224, 224, 3),
}
for key, shape in expected.items():
    actual = tuple(features.get(key, {}).get('shape', ()))
    if actual != shape:
        raise SystemExit(f'{key}: expected {shape}, got {actual}')
print('[dataset] schema compatible with AnyTouch2 Action-Expert')
PY

if [[ "$FORCE_NORM_STATS" == "1" || ! -f "$NORM_STATS_FILE" ]]; then
  if [[ "$FORCE_NORM_STATS" != "1" && -n "$NORM_STATS_SOURCE_FILE" && -f "$NORM_STATS_SOURCE_FILE" ]]; then
    echo "[norm-stats] copying $NORM_STATS_SOURCE_FILE -> $NORM_STATS_FILE"
    cp "$NORM_STATS_SOURCE_FILE" "$NORM_STATS_FILE"
  else
    echo "[norm-stats] computing $CONFIG_NAME"
    uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
  fi
else
  echo "[norm-stats] reusing $NORM_STATS_FILE"
fi
[[ -f "$NORM_STATS_FILE" ]] || { echo "[error] norm stats unavailable: $NORM_STATS_FILE" >&2; exit 1; }

command -v nvidia-smi >/dev/null || { echo '[error] nvidia-smi not found' >&2; exit 1; }
while true; do
  all_ready=1
  for gpu_id in "${GPU_ARRAY[@]}"; do
    used="$(nvidia-smi --id="${gpu_id//[[:space:]]/}" --query-gpu=memory.used --format=csv,noheader,nounits)"
    used="${used//[[:space:]]/}"
    [[ "$used" =~ ^[0-9]+$ ]] || { echo "[error] invalid GPU memory: $used" >&2; exit 1; }
    echo "[gpu] $gpu_id: $used MiB (requires < $GPU_MEMORY_THRESHOLD_MIB MiB)"
    if (( used >= GPU_MEMORY_THRESHOLD_MIB )); then all_ready=0; fi
  done
  (( all_ready )) && break
  echo "[gpu] waiting ${GPU_WAIT_INTERVAL_SECONDS}s"; sleep "$GPU_WAIT_INTERVAL_SECONDS"
done

echo "[train] $CONFIG_NAME / $EXP_NAME on GPUs $GPU_IDS"
printf '[train] command:'; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'
"${TRAIN_ARGS[@]}"
