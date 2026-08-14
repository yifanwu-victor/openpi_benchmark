#!/usr/bin/env bash
set -euo pipefail

# TI-ONE task wrapper for JAX multi-node training:
#   2 CVM nodes x 8 GPUs = 16 global devices.
#
# Submit as a TI-ONE distributed/JAX task with:
#   node count: 2
#   GPUs per node: 8
#   startup command:
#     bash scripts/train_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_tione_2x8.sh

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export FSDP_DEVICES="${FSDP_DEVICES:-8}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export EXP_NAME="${EXP_NAME:-insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_tione_2x8_v1}"
export JAX_LOCAL_DEVICE_COUNT="${JAX_LOCAL_DEVICE_COUNT:-8}"
export JAX_NUM_PROCESSES="${JAX_NUM_PROCESSES:-${NNODES:-${NODE_NUM:-2}}}"
export JAX_PROCESS_INDEX="${JAX_PROCESS_INDEX:-${NODE_RANK:-${RANK:-0}}}"
export MASTER_PORT="${MASTER_PORT:-23456}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/home/tione/notebook/data/jax_cache}"
export OPENPI_SKIP_WANDB_IMAGES="${OPENPI_SKIP_WANDB_IMAGES:-1}"

if [[ "${JAX_NUM_PROCESSES}" != "1" ]]; then
  if [[ -z "${JAX_COORDINATOR_ADDRESS:-}" ]]; then
    if [[ -z "${MASTER_ADDR:-}" ]]; then
      echo "[error] MASTER_ADDR or JAX_COORDINATOR_ADDRESS is required for multi-node JAX training." >&2
      echo "        TI-ONE distributed tasks normally inject MASTER_ADDR, MASTER_PORT, and RANK/NODE_RANK." >&2
      exit 1
    fi
    export JAX_COORDINATOR_ADDRESS="${MASTER_ADDR}:${MASTER_PORT}"
  fi
fi

mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "[tione-2x8] host=$(hostname)"
echo "[tione-2x8] JAX_NUM_PROCESSES=${JAX_NUM_PROCESSES}"
echo "[tione-2x8] JAX_PROCESS_INDEX=${JAX_PROCESS_INDEX}"
echo "[tione-2x8] JAX_COORDINATOR_ADDRESS=${JAX_COORDINATOR_ADDRESS:-single-node}"
echo "[tione-2x8] GPU_IDS=${GPU_IDS}, FSDP_DEVICES=${FSDP_DEVICES}, BATCH_SIZE=${BATCH_SIZE}"

bash scripts/train_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_local.sh
