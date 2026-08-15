#!/usr/bin/env bash
set -euo pipefail

# TI-ONE MPI launcher for OpenPI/JAX multi-node training.
#
# Intended TI-ONE UI command:
#   cd /home/tione/notebook/openpi_benchmark
#   BATCH_SIZE=256 NUM_TRAIN_STEPS=5000 SAVE_INTERVAL=1000 \
#   EXP_NAME=openpi05_2x8_bs256_5k_v1 bash scripts/tione_mpi_train.sh
#
# This script runs on the launcher node and uses mpirun to start one JAX
# process per CVM node. The per-node process then calls TRAIN_SCRIPT.

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_tione_2x8.sh}"
cd "$REPO_ROOT"

fail() {
  echo "[tione-mpi][error] $*" >&2
  exit 1
}

warn() {
  echo "[tione-mpi][warn] $*" >&2
}

require_path() {
  local path="$1"
  local desc="$2"
  [[ -e "$path" ]] || fail "missing ${desc}: ${path}"
}

require_positive_int() {
  local name="$1"
  local value="${!name:-}"
  [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]] || fail "${name} must be a positive integer, got '${value}'"
}

export PATH="$HOME/.local/bin:$PATH"

# Training defaults. EXP_NAME intentionally defaults to a descriptive, timestamped
# value so repeated UI submissions do not accidentally overwrite old runs.
export BATCH_SIZE="${BATCH_SIZE:-256}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_PERIOD="${KEEP_PERIOD:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"
export RESUME="${RESUME:-0}"
export FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"

export MASTER_PORT="${MASTER_PORT:-23456}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/home/tione/notebook/data/jax_cache}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

export JAX_NUM_PROCESSES="${JAX_NUM_PROCESSES:-${NNODES:-${NODE_NUM:-2}}}"
export JAX_LOCAL_DEVICE_COUNT="${JAX_LOCAL_DEVICE_COUNT:-${GPU_NUM_PER_NODE:-8}}"

require_positive_int BATCH_SIZE
require_positive_int NUM_TRAIN_STEPS
require_positive_int SAVE_INTERVAL
require_positive_int LOG_INTERVAL
require_positive_int JAX_NUM_PROCESSES
require_positive_int JAX_LOCAL_DEVICE_COUNT

if [[ "$RESUME" == "1" && -z "${EXP_NAME:-}" ]]; then
  fail "RESUME=1 requires an explicit EXP_NAME"
fi

if [[ -z "${EXP_NAME:-}" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  export EXP_NAME="openpi05_${JAX_NUM_PROCESSES}x${JAX_LOCAL_DEVICE_COUNT}_bs${BATCH_SIZE}_steps${NUM_TRAIN_STEPS}_${stamp}"
  warn "EXP_NAME was not set; using ${EXP_NAME}"
else
  export EXP_NAME
fi

if [[ "$EXP_NAME" =~ bs([0-9]+) ]]; then
  name_bs="${BASH_REMATCH[1]}"
  if [[ "$name_bs" != "$BATCH_SIZE" ]]; then
    warn "EXP_NAME contains bs${name_bs}, but BATCH_SIZE=${BATCH_SIZE}"
  fi
fi

if [[ "$EXP_NAME" =~ steps([0-9]+) ]]; then
  name_steps="${BASH_REMATCH[1]}"
  if [[ "$name_steps" != "$NUM_TRAIN_STEPS" ]]; then
    warn "EXP_NAME contains steps${name_steps}, but NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS}"
  fi
fi

if (( BATCH_SIZE % JAX_NUM_PROCESSES != 0 )); then
  fail "BATCH_SIZE=${BATCH_SIZE} must be divisible by JAX_NUM_PROCESSES=${JAX_NUM_PROCESSES}"
fi

total_devices=$((JAX_NUM_PROCESSES * JAX_LOCAL_DEVICE_COUNT))
per_device_batch=""
if (( BATCH_SIZE % total_devices != 0 )); then
  warn "BATCH_SIZE=${BATCH_SIZE} is not divisible by total devices=${total_devices}; per-device batch will be uneven or framework-dependent"
else
  per_device_batch=$((BATCH_SIZE / total_devices))
fi

require_path "$TRAIN_SCRIPT" "training script"
require_path ".venv" "uv virtualenv"
require_path "/home/tione/notebook/data/lerobot/simulation/insert_block_gelsight_delta_eef" "LeRobot dataset"
require_path "/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base/params" "pi05 base params"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

if [[ -n "${NODE_LIST:-}" ]]; then
  first_node="${NODE_LIST%%,*}"
  coordinator_host="${first_node%%:*}"
else
  coordinator_host="${MASTER_ADDR:-}"
fi
[[ -n "$coordinator_host" ]] || fail "NODE_LIST or MASTER_ADDR is required to derive JAX coordinator host"

export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_ADDRESS:-${coordinator_host}:${MASTER_PORT}}"

hostfile="${OMPI_MCA_orte_default_hostfile:-/etc/mpi/hostfile}"
[[ -f "$hostfile" ]] || fail "MPI hostfile not found: ${hostfile}"

cat <<EOF
[tione-mpi] repo=${REPO_ROOT}
[tione-mpi] train_script=${TRAIN_SCRIPT}
[tione-mpi] node_list=${NODE_LIST:-unknown}
[tione-mpi] hostfile=${hostfile}
[tione-mpi] JAX_COORDINATOR_ADDRESS=${JAX_COORDINATOR_ADDRESS}
[tione-mpi] JAX_NUM_PROCESSES=${JAX_NUM_PROCESSES}
[tione-mpi] JAX_LOCAL_DEVICE_COUNT=${JAX_LOCAL_DEVICE_COUNT}
[tione-mpi] total_devices=${total_devices}
[tione-mpi] BATCH_SIZE=${BATCH_SIZE}${per_device_batch:+, per_device_batch=${per_device_batch}}
[tione-mpi] NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS}
[tione-mpi] SAVE_INTERVAL=${SAVE_INTERVAL}, KEEP_PERIOD=${KEEP_PERIOD}, LOG_INTERVAL=${LOG_INTERVAL}
[tione-mpi] EXP_NAME=${EXP_NAME}
[tione-mpi] checkpoint_dir=/home/tione/notebook/checkpoints/pi05_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft/${EXP_NAME}
[tione-mpi] WANDB_MODE=${WANDB_MODE}, WANDB_DISABLED=${WANDB_DISABLED}
EOF

mpirun --allow-run-as-root \
  -np "${JAX_NUM_PROCESSES}" \
  --hostfile "$hostfile" \
  --map-by ppr:1:node \
  --bind-to none \
  -x PATH \
  -x HOME \
  -x REPO_ROOT \
  -x TRAIN_SCRIPT \
  -x GPU_NUM_PER_NODE \
  -x MASTER_PORT \
  -x JAX_COORDINATOR_ADDRESS \
  -x JAX_NUM_PROCESSES \
  -x JAX_LOCAL_DEVICE_COUNT \
  -x JAX_COMPILATION_CACHE_DIR \
  -x WANDB_MODE \
  -x WANDB_DISABLED \
  -x BATCH_SIZE \
  -x NUM_TRAIN_STEPS \
  -x SAVE_INTERVAL \
  -x KEEP_PERIOD \
  -x LOG_INTERVAL \
  -x EXP_NAME \
  -x RESUME \
  -x FORCE_NORM_STATS \
  bash -lc '
    cd "$REPO_ROOT"
    export PATH="$HOME/.local/bin:$PATH"
    export JAX_PROCESS_INDEX="${OMPI_COMM_WORLD_RANK}"
    if [[ "${OMPI_COMM_WORLD_RANK}" == "0" ]]; then
      export JAX_COORDINATOR_BIND_ADDRESS="0.0.0.0:${MASTER_PORT}"
    fi
    bash "$TRAIN_SCRIPT"
  '
