# AnyTouch2 Tactile Encoder Notes

This folder keeps AnyTouch2 experiments separate from the existing UniVTAC
converters and dual-SigLIP training path.

## Data Conversion

`convert_univtac_sim_delta_eef_to_lerobot.py` converts UniVTAC sim HDF5 episodes
to a LeRobot dataset with tactile clips:

```bash
cd /shareNFS_40/wyf/workplace/openpi

uv run examples/anytouch2/convert_univtac_sim_delta_eef_to_lerobot.py \
  --hdf5-dir /shareNFS_40/wyf/data/UniVTAC/sim/insert_USB_v3/GelSight/noise3cm_1_5mm \
  --output-dir /shareNFS_40/wyf/data/lerobot/simulation/insert_usb_v3_gelsight_abs_joint_anytouch2_clip \
  --repo-id simulation/insert_usb_v3_gelsight_abs_joint_anytouch2_clip \
  --control-mode abs_joint \
  --overwrite
```

The standard fields are preserved:

```text
image
wrist_image
state   # [joint7, gripper]
actions # [target joint7, target gripper]
task
```

Additional tactile clip fields:

```text
left_tactile_clip   [4, 224, 224, 3]
right_tactile_clip  [4, 224, 224, 3]
```

For a current frame `t`, clips use raw frames `[t-6, t-4, t-2, t]`; negative
indices are clamped to `0`.

## Model Path

The Torch config is:

```text
pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft
```

It uses:

```text
base/wrist images -> original PaliGemma SigLIP
left/right tactile clips -> AnyTouch2
prefix order -> left tactile, right tactile, base image, wrist image, language
```

AnyTouch2 outputs are compressed per side to:

```text
1 CLS + 5 sensor + 30 patch tokens = 36 tokens
```

Then both sides are concatenated and projected from AnyTouch dim `512` to
OpenPI prefix dim `2048`.

## Checkpoint

The config currently expects:

```text
/shareNFS_40/wyf/workplace/AnyTouch2/checkpoints/anytouch2.pth
```

That checkpoint is required. If it is missing, model construction fails early
instead of silently using random AnyTouch weights.
