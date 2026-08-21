#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/tione/notebook/openpi_benchmark}"
HDF5_DIR="${HDF5_DIR:-/home/tione/notebook/data/UniVTAC/sim/insert_USB}"
PI_TRAINING_DATA_ROOT="${PI_TRAINING_DATA_ROOT:-/run/ti/pi-training-data}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI_TRAINING_DATA_ROOT}/lerobot}"
REPO_ID="${REPO_ID:-simulation/insert_usb_gelsight_abs_joint_100}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_LEROBOT_HOME}/${REPO_ID}}"
OVERWRITE="${OVERWRITE:-1}"
DRY_RUN="${DRY_RUN:-0}"

cd "$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$(dirname "$OUTPUT_DIR")"

cmd=(
  uv run examples/UniVTAC/convert_sim_to_lerobot.py
  --hdf5-dir "$HDF5_DIR"
  --output-dir "$OUTPUT_DIR"
  --repo-id "$REPO_ID"
  --control-mode abs_joint
  --gripper-mode left
  --image-size 224 224
  --max-episodes 100
)

if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

printf '[data-convert-insert-usb-100] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then
  echo '[data-convert-insert-usb-100] dry run complete'
  exit 0
fi
exec "${cmd[@]}"
