# OpenPI Delta Joint Serve Interface

This note records the websocket serve interface for the Insert USB GelSight sim delta-joint model trained by:

```bash
bash scripts/train_insert_usb_gelsight_sim_delta_joint_vision_full_ft.sh
```

The server only runs:

```text
observation -> OpenPI policy -> action_chunk
```

It does not collect images, read simulator state, or execute robot commands. The client is responsible for collecting the current observation, sending it to the server, converting returned delta-joint actions to executable commands, and applying safety/control logic.

## Checkpoint

Config:

```text
pi05_insert_usb_gelsight_sim_delta_joint_vision_full_ft
```

Dataset:

```text
/shareNFS_40/wyf/data/lerobot/simulation/insert_usb_gelsight_delta_joint
```

Training action horizon:

```text
50
```

Prompt source:

```text
LeRobot task field, originally from HDF5 instruction attr
```

For the current Insert USB sim dataset, the first observed prompt is:

```text
Pick up the USB plug from the blue slot and insert it into the red USB slot.
```

## Serve Command

Replace `CHECKPOINT_STEP` with the trained step directory, for example `9999` or `19999`.

```bash
cd /shareNFS_40/wyf/workplace/openpi
source ~/.bashrc

CUDA_VISIBLE_DEVICES=0 \
OPENPI_DATA_HOME=/shareNFS_40/wyf/data/openpi \
HF_HOME=/shareNFS_40/wyf/data/hf_cache \
HF_DATASETS_CACHE=/shareNFS_40/wyf/data/hf_cache/datasets \
HF_LEROBOT_HOME=/shareNFS_40/wyf/data/lerobot \
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_insert_usb_gelsight_sim_delta_joint_vision_full_ft \
  --policy.dir=/shareNFS_40/wyf/workplace/openpi/checkpoints/pi05_insert_usb_gelsight_sim_delta_joint_vision_full_ft/insert_usb_gelsight_sim_delta_joint_vision_full_ft_v1/CHECKPOINT_STEP
```

`--port` is a top-level `serve_policy.py` option and must appear before the `policy:checkpoint` subcommand.

## Client Observation

This config is vision-only. It uses two image streams and ignores tactile inputs.

Send a dict with these keys:

```python
{
    "observation/state": np.ndarray,        # shape [8], float32
    "observation/image": np.ndarray,        # head/base RGB image, HWC uint8
    "observation/wrist_image": np.ndarray,  # wrist RGB image, HWC uint8
    "prompt": str,
}
```

State convention:

```text
observation/state = [
  panda_joint1,
  panda_joint2,
  panda_joint3,
  panda_joint4,
  panda_joint5,
  panda_joint6,
  panda_joint7,
  gripper_q,
]
```

Shape and dtype:

```text
observation/state:       [8] float32
observation/image:       [H, W, 3] uint8 RGB
observation/wrist_image: [H, W, 3] uint8 RGB
prompt:                  non-empty str
```

The training converter resized images to `224x224`. The server/model transform can handle normal image preprocessing, but the safest client convention is to send RGB `uint8` images with the same visual channel order as training.

## Server Output

The server returns:

```python
{
    "actions": np.ndarray  # shape [50, 8], float32
}
```

Action convention:

```text
actions[t] = [
  delta_panda_joint1,
  delta_panda_joint2,
  delta_panda_joint3,
  delta_panda_joint4,
  delta_panda_joint5,
  delta_panda_joint6,
  delta_panda_joint7,
  target_gripper_q,
]
```

`delta_panda_joint*` is relative to the observation state used for this inference request, not relative to the episode first frame.

## Client-Side Conversion To Absolute Joint Target

Before executing a returned action, convert it to absolute qpos:

```python
current_qpos8 = obs["observation/state"].astype(np.float32)
action = action_chunk[i].astype(np.float32)

target_qpos8 = np.empty(8, dtype=np.float32)
target_qpos8[:7] = current_qpos8[:7] + action[:7]
target_qpos8[7] = action[7]
```

For open-loop chunk execution, keep the same base `current_qpos8` for all actions from the same inference request:

```python
base_qpos8 = obs["observation/state"].astype(np.float32)
for i in range(open_loop_horizon):
    action = action_chunk[i].astype(np.float32)
    target_qpos8 = np.empty(8, dtype=np.float32)
    target_qpos8[:7] = base_qpos8[:7] + action[:7]
    target_qpos8[7] = action[7]
    execute_abs_joint_target(target_qpos8)
```

After consuming the chosen `open_loop_horizon`, query the server again with a fresh observation.

## Tactile Note

This config uses `SimVisionInputs`, not `SimVisionTactileInputs`.

It consumes:

```text
observation/image
observation/wrist_image
observation/state
prompt
```

It does not consume:

```text
observation/left_tactile_image
observation/right_tactile_image
```

If the client always sends tactile images, they are extra fields for this config and are not used by the model.

