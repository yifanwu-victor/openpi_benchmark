#!/usr/bin/env bash
set -euo pipefail

# Train Insert Block GelSight sim delta-EEF full fine-tuning with vision+tactile inputs on this server.
#
# Assumes LeRobot data already exists at:
#   /home/tione/notebook/data/lerobot/simulation/insert_block_gelsight_delta_eef
#
# Assumes pi05_base already exists at:
#   /home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base
#
# Defaults:
#   - config: pi05_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft
#   - norm stats: pi05_insert_block_gelsight_sim_delta_eef_vision_full_ft
#   - GPUs: CUDA_VISIBLE_DEVICES=0,1,2,3
#   - JAX FSDP devices: 4
#   - global batch size: 64
#   - action horizon: 50
#   - train steps: 20000
#   - save interval: 5000
#   - EMA: 0.999

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
CONFIG_NAME="${CONFIG_NAME:-pi05_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft}"
STATS_CONFIG_NAME="${STATS_CONFIG_NAME:-pi05_insert_block_gelsight_sim_delta_eef_vision_full_ft}"
EXP_NAME="${EXP_NAME:-insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_local_v1}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
RESUME="${RESUME:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/home/tione/notebook/checkpoints}"
PI05_BASE_DIR="${PI05_BASE_DIR:-/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base}"

# Keep this as an ancestor of PI05_BASE_DIR. openpi.shared.download resolves
# symlinks before cache invalidation, so a symlinked checkpoint outside
# OPENPI_DATA_HOME will fail pathlib.relative_to(...).
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/tione/notebook/data/openpi}"
HF_HOME="${HF_HOME:-/home/tione/notebook/data/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/tione/notebook/data/hf_cache/datasets}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/home/tione/notebook/data/lerobot}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"

NORM_STATS_DIR="${REPO_ROOT}/assets/${STATS_CONFIG_NAME}/simulation/insert_block_gelsight_delta_eef"
RUN_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
DATASET_DIR="${HF_LEROBOT_HOME}/simulation/insert_block_gelsight_delta_eef"
PI05_CACHE_DIR="${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_base"

cd "$REPO_ROOT"
mkdir -p "$CHECKPOINT_BASE_DIR" "$HF_DATASETS_CACHE" "$(dirname "$PI05_CACHE_DIR")"
TRITON_PTXAS_DIR="${REPO_ROOT}/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin"
if [[ -x "${TRITON_PTXAS_DIR}/ptxas" ]]; then
  export PATH="${TRITON_PTXAS_DIR}:$PATH"
fi

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OPENPI_DATA_HOME HF_HOME HF_DATASETS_CACHE HF_LEROBOT_HOME
export XLA_PYTHON_CLIENT_PREALLOCATE UV_LINK_MODE GIT_LFS_SKIP_SMUDGE

require_positive_int() {
  local name="$1"
  local value="${!name:-}"
  if [[ ! "$value" =~ ^[0-9]+$ || "$value" -le 0 ]]; then
    echo "[error] ${name} must be a positive integer, got '${value}'" >&2
    exit 1
  fi
}

require_positive_int BATCH_SIZE
require_positive_int NUM_TRAIN_STEPS
require_positive_int SAVE_INTERVAL
require_positive_int LOG_INTERVAL

if [[ "${JAX_NUM_PROCESSES:-1}" =~ ^[0-9]+$ && "${JAX_NUM_PROCESSES:-1}" -gt 1 ]]; then
  if (( BATCH_SIZE % JAX_NUM_PROCESSES != 0 )); then
    echo "[error] BATCH_SIZE=${BATCH_SIZE} must be divisible by JAX_NUM_PROCESSES=${JAX_NUM_PROCESSES}" >&2
    exit 1
  fi
fi

if [[ "$EXP_NAME" =~ bs([0-9]+) ]]; then
  name_bs="${BASH_REMATCH[1]}"
  if [[ "$name_bs" != "$BATCH_SIZE" ]]; then
    echo "[warn] EXP_NAME contains bs${name_bs}, but BATCH_SIZE=${BATCH_SIZE}" >&2
  fi
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[error] missing LeRobot dataset: ${DATASET_DIR}" >&2
  exit 1
fi

if [[ ! -d "${PI05_BASE_DIR}/params" ]]; then
  echo "[error] missing pi05_base params directory: ${PI05_BASE_DIR}/params" >&2
  exit 1
fi

openpi_data_home_real="$(readlink -f "$OPENPI_DATA_HOME")"
pi05_base_real="$(readlink -f "$PI05_BASE_DIR")"
case "${pi05_base_real}/" in
  "${openpi_data_home_real}/"*) ;;
  *)
    echo "[error] PI05_BASE_DIR must be inside OPENPI_DATA_HOME when using a symlinked OpenPI cache." >&2
    echo "        OPENPI_DATA_HOME=${OPENPI_DATA_HOME}" >&2
    echo "        PI05_BASE_DIR=${PI05_BASE_DIR}" >&2
    echo "        Try: unset OPENPI_DATA_HOME" >&2
    echo "        Or:  OPENPI_DATA_HOME=/home/tione/notebook/data/openpi bash $0" >&2
    exit 1
    ;;
esac

# The config loads gs://openpi-assets/checkpoints/pi05_base/params. OpenPI maps that
# to ${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_base/params, so link the
# downloaded local checkpoint into the expected cache location.
if [[ -L "$PI05_CACHE_DIR" ]]; then
  current_target="$(readlink -f "$PI05_CACHE_DIR")"
  expected_target="$(readlink -f "$PI05_BASE_DIR")"
  if [[ "$current_target" != "$expected_target" ]]; then
    rm "$PI05_CACHE_DIR"
    ln -s "$PI05_BASE_DIR" "$PI05_CACHE_DIR"
  fi
elif [[ -e "$PI05_CACHE_DIR" ]]; then
  if [[ ! -d "${PI05_CACHE_DIR}/params" ]]; then
    if [[ -d "$PI05_CACHE_DIR" && -z "$(find "$PI05_CACHE_DIR" -mindepth 1 -print -quit)" ]]; then
      rmdir "$PI05_CACHE_DIR"
      ln -s "$PI05_BASE_DIR" "$PI05_CACHE_DIR"
    else
      echo "[error] ${PI05_CACHE_DIR} exists but does not contain params/." >&2
      echo "        Move it aside or set OPENPI_DATA_HOME/PI05_BASE_DIR explicitly." >&2
      exit 1
    fi
  fi
else
  ln -s "$PI05_BASE_DIR" "$PI05_CACHE_DIR"
fi

if [[ "${XLA_FLAGS:-}" == *"--xla_gpu_autotune_level=0"* ]]; then
  echo "[env] removing XLA_FLAGS=${XLA_FLAGS}; this flag can make JAX GPU kernels fragile"
  unset XLA_FLAGS
fi

if [[ "$FORCE_NORM_STATS" == "1" || ! -d "$NORM_STATS_DIR" ]]; then
  if [[ "${JAX_NUM_PROCESSES:-1}" != "1" && "${JAX_PROCESS_INDEX:-0}" != "0" ]]; then
    echo "[norm-stats] waiting for primary process to create ${NORM_STATS_DIR}"
    for _ in $(seq 1 180); do
      [[ -d "$NORM_STATS_DIR" ]] && break
      sleep 10
    done
    if [[ ! -d "$NORM_STATS_DIR" ]]; then
      echo "[error] norm stats were not created by the primary process: ${NORM_STATS_DIR}" >&2
      exit 1
    fi
  else
    echo "[norm-stats] computing stats for ${STATS_CONFIG_NAME}"
    uv run scripts/compute_norm_stats.py --config-name "$STATS_CONFIG_NAME"
  fi
else
  echo "[norm-stats] found ${NORM_STATS_DIR}; set FORCE_NORM_STATS=1 to recompute"
fi

echo "[train] config=${CONFIG_NAME}"
echo "[train] exp=${EXP_NAME}"
echo "[train] GPUs=${GPU_IDS}, fsdp_devices=${FSDP_DEVICES}, global_batch_size=${BATCH_SIZE}"
echo "[train] steps=${NUM_TRAIN_STEPS}, save_interval=${SAVE_INTERVAL}, keep_period=${KEEP_PERIOD}, log_interval=${LOG_INTERVAL}, resume=${RESUME}"
echo "[train] jax_process=${JAX_PROCESS_INDEX:-0}/${JAX_NUM_PROCESSES:-1}, coordinator=${JAX_COORDINATOR_ADDRESS:-single-node}"
echo "[train] dataset=${DATASET_DIR}"
echo "[train] pi05_base=${PI05_BASE_DIR}"
echo "[train] checkpoint dir=${RUN_DIR}"

uv run scripts/train.py \
  "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD" \
  --log-interval "$LOG_INTERVAL" \
  --fsdp-devices "$FSDP_DEVICES" \
  $(if [[ "$RESUME" == "1" ]]; then echo "--resume"; else echo "--overwrite"; fi)

echo "[done] checkpoint directory: ${RUN_DIR}"
