#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
HDF5_DIR="${HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/pull_drawer}"
PI_TRAINING_DATA_ROOT="${PI_TRAINING_DATA_ROOT:-/run/ti/pi-training-data}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI_TRAINING_DATA_ROOT}/lerobot}"
REPO_ID="${REPO_ID:-simulation/pull_drawer_gelsight_abs_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
CONFIG_NAME="${CONFIG_NAME:-pi05_pull_drawer_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora}"
EXP_NAME="${EXP_NAME:-pull_drawer_abs_joint_vision_tactile_dual_siglip_lora_v1}"
GPU_IDS="${GPU_IDS:-0,1}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
OVERWRITE="${OVERWRITE:-1}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}"

cd "$REPO_ROOT"
if [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  export PS1="${PS1-}"
  source "${HOME}/.bashrc"
  set -u
fi
unset VIRTUAL_ENV

TRITON_PTXAS_DIR="${TRITON_PTXAS_DIR:-${REPO_ROOT}/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin}"
export TRITON_PTXAS_DIR
export PATH="${TRITON_PTXAS_DIR}:$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_LEROBOT_HOME

NORM_STATS_FILE="${REPO_ROOT}/assets/${CONFIG_NAME}/${REPO_ID}/norm_stats.json"
CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
DATASET_MARKER="${OUTPUT_DIR}/meta/info.json"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[pull-drawer-dual-siglip] raw data: $HDF5_DIR"
  echo "[pull-drawer-dual-siglip] dataset: $OUTPUT_DIR"
  echo "[pull-drawer-dual-siglip] config: $CONFIG_NAME"
  echo "[pull-drawer-dual-siglip] GPUs: $GPU_IDS (FSDP devices: $FSDP_DEVICES)"
  echo "[pull-drawer-dual-siglip] checkpoint: $CHECKPOINT_DIR"
  echo "[pull-drawer-dual-siglip] norm stats: $NORM_STATS_FILE"
  echo "[pull-drawer-dual-siglip] dry run complete"
  exit 0
fi

if ! command -v ptxas >/dev/null 2>&1; then
  echo "[pull-drawer-dual-siglip] ptxas was not found on PATH" >&2
  exit 1
fi
printf '[pull-drawer-dual-siglip] ptxas: '
ptxas --version

mkdir -p "$(dirname "$OUTPUT_DIR")"
if [[ "$SKIP_CONVERT" == "1" ]]; then
  echo "[pull-drawer-dual-siglip] skipping conversion (SKIP_CONVERT=1)"
elif [[ "$FORCE_CONVERT" == "1" || ! -f "$DATASET_MARKER" ]]; then
  convert_args=(
    uv run examples/UniVTAC/convert_sim_to_lerobot.py
    --hdf5-dir "$HDF5_DIR"
    --output-dir "$OUTPUT_DIR"
    --repo-id "$REPO_ID"
    --control-mode abs_joint
    --gripper-mode left
    --image-size 224 224
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    convert_args+=(--overwrite)
  fi
  if [[ -n "$MAX_EPISODES" ]]; then
    convert_args+=(--max-episodes "$MAX_EPISODES")
  fi
  printf '[pull-drawer-dual-siglip] conversion:'
  printf ' %q' "${convert_args[@]}"
  printf '\n'
  "${convert_args[@]}"
else
  echo "[pull-drawer-dual-siglip] reusing existing dataset: $OUTPUT_DIR"
fi

if [[ "$FORCE_NORM_STATS" == "1" || ! -f "$NORM_STATS_FILE" ]]; then
  stats_args=(uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME")
  printf '[pull-drawer-dual-siglip] norm stats:'
  printf ' %q' "${stats_args[@]}"
  printf '\n'
  "${stats_args[@]}"
else
  echo "[pull-drawer-dual-siglip] reusing norm stats: $NORM_STATS_FILE"
fi

train_args=(
  "$CONFIG_NAME"
  --exp-name "$EXP_NAME"
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
  --num-train-steps "$NUM_TRAIN_STEPS"
  --batch-size "$BATCH_SIZE"
  --save-interval "$SAVE_INTERVAL"
  --keep-period "$KEEP_PERIOD"
  --fsdp-devices "$FSDP_DEVICES"
)
if [[ "$RESUME" == "1" ]]; then
  train_args+=(--resume)
else
  train_args+=(--overwrite)
fi

printf '[pull-drawer-dual-siglip] training:'
printf ' %q' uv run scripts/train.py "${train_args[@]}"
printf '\n'
exec uv run scripts/train.py "${train_args[@]}"
