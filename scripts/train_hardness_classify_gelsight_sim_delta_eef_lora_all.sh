#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the split Hardness Classify GelSight LoRA scripts.
# LeRobot conversion is intentionally not run here; use convert_sim_to_lerobot.py
# separately if the dataset needs to be regenerated.

REPO_ROOT="${REPO_ROOT:-/shareNFS_40/wyf/workplace/openpi}"
cd "$REPO_ROOT"

echo "[info] Hardness training is now split into two scripts:"
echo "  vision:         scripts/train_hardness_classify_gelsight_sim_delta_eef_vision_lora.sh"
echo "  vision+tactile: scripts/train_hardness_classify_gelsight_sim_delta_eef_vision_tactile_lora.sh"
echo
echo "[info] Running both sequentially. Set RUN_BOTH=0 to only print this message."

if [[ "${RUN_BOTH:-1}" != "1" ]]; then
  exit 0
fi

bash scripts/train_hardness_classify_gelsight_sim_delta_eef_vision_lora.sh
bash scripts/train_hardness_classify_gelsight_sim_delta_eef_vision_tactile_lora.sh
