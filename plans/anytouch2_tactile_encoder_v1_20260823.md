# AnyTouch2 Tactile Encoder v1

Date: 2026-08-23

This note records the first OpenPI experiment that replaces the tactile-as-extra-SigLIP-view path with an AnyTouch2 tactile encoder.

## Goal

Keep the original OpenPI vision and language path intact, but route GelSight tactile streams through an AnyTouch2 encoder:

```text
left tactile clip -> AnyTouch2
right tactile clip -> AnyTouch2
base image -> PaliGemma SigLIP
wrist image -> PaliGemma SigLIP
language -> PaliGemma text tokens
```

Prefix token order is fixed as:

```text
left tactile -> right tactile -> base image -> wrist image -> language
```

## Difference From Original OpenPI

Original OpenPI pi0.5 uses PaliGemma SigLIP image tokens for image observations. Earlier UniVTAC tactile experiments treated tactile images as additional image views, so tactile and RGB camera frames both went through SigLIP.

This v1 path differs in three ways:

1. Tactile input is a short clip, not a single frame.
2. Tactile clips go through AnyTouch2, not SigLIP.
3. AnyTouch2 tokens are projected into the same prefix width as OpenPI image tokens before concatenation.

The base/wrist image path, language tokenizer, action expert, normalization flow, and websocket observation key names are otherwise intended to stay compatible with the existing OpenPI Torch path.

## Code Paths

AnyTouch2 model wrapper:

```text
src/openpi/models_pytorch/anytouch2/
```

Main wrapper:

```text
src/openpi/models_pytorch/anytouch2/encoder.py
```

Minimal migrated AnyTouch2 model:

```text
src/openpi/models_pytorch/anytouch2/tactile_mae.py
src/openpi/models_pytorch/anytouch2/configs/CLIP-B-16/config.json
```

OpenPI Torch routing:

```text
src/openpi/models/pi0_config.py
src/openpi/models_pytorch/pi0_pytorch.py
src/openpi/models_pytorch/preprocessing_pytorch.py
```

Data transform:

```text
src/openpi/policies/sim_policy.py
src/openpi/training/config.py
```

Dataset converter:

```text
examples/anytouch2/convert_univtac_sim_delta_eef_to_lerobot.py
```

Training script:

```text
scripts/train_insert_usb_v3_anytouch2_tactile_full_ft.sh
```

## Dataset Format

The AnyTouch2 converter writes a LeRobot dataset with standard fields:

```text
image
wrist_image
state
actions
task
```

And new tactile clip fields:

```text
left_tactile_clip   [4, 224, 224, 3]
right_tactile_clip  [4, 224, 224, 3]
```

For current raw frame `t`, clip frames are:

```text
[t-6, t-4, t-2, t]
```

Out-of-range early indices are clamped to frame `0`.

## Current Config

```text
pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft
```

Important settings:

```text
tactile_encoder_type="anytouch2"
image_keys=("left_tactile_0_rgb", "right_tactile_0_rgb", "base_0_rgb", "wrist_0_rgb")
tactile_image_keys=("left_tactile_0_rgb", "right_tactile_0_rgb")
anytouch_num_frames=4
anytouch_stride=2
anytouch_sensor_id=0
anytouch_keep_cls_sensor_tokens=True
anytouch_patch_tokens_per_side=30
```

Dataset:

```text
simulation/insert_usb_v3_gelsight_abs_joint_anytouch2_clip
```

State/action semantics:

```text
LeRobot state   = [joint7, gripper]                # 8 dims, absolute
LeRobot actions = [target joint7, target gripper]  # 8 dims, absolute
Training action = [target joint7 - current joint7, abs gripper]
```

## Token Shape

AnyTouch2 produces one side of tactile tokens as:

```text
1 CLS + 5 sensor + 30 compressed patch tokens = 36 tokens
```

Left and right tactile together:

```text
72 tactile tokens
```

The projector maps:

```text
AnyTouch dim 512 -> OpenPI prefix dim 2048
```

## Checkpoint Requirement

The config expects:

```text
/shareNFS_40/wyf/workplace/AnyTouch2/checkpoints/anytouch2.pth
```

If the official checkpoint is downloaded as:

```text
checkpoint-4frames.pth
```

create:

```bash
cd /shareNFS_40/wyf/workplace/AnyTouch2/checkpoints
ln -s checkpoint-4frames.pth anytouch2.pth
```

The training script creates this symlink automatically if possible.

## Run

```bash
cd /shareNFS_40/wyf/workplace/openpi

bash scripts/train_insert_usb_v3_anytouch2_tactile_full_ft.sh
```

Useful overrides:

```bash
SKIP_CONVERT=1 bash scripts/train_insert_usb_v3_anytouch2_tactile_full_ft.sh
FORCE_NORM_STATS=1 bash scripts/train_insert_usb_v3_anytouch2_tactile_full_ft.sh
GPU_IDS=4,5,6,7 NPROC_PER_NODE=4 bash scripts/train_insert_usb_v3_anytouch2_tactile_full_ft.sh
```

## Notes

This is Torch-only. JAX configs and previous SigLIP tactile configs should remain unaffected.

AnyTouch2 normalization happens inside the model wrapper, not in the LeRobot converter. This keeps online inference and offline training closer: both can pass raw `uint8` clip tensors and let the encoder apply its own preprocessing.
