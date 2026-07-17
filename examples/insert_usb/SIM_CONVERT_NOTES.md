# Insert USB Sim To LeRobot Notes

This note records the intended conversion strategy for the UniVTAC simulated
Insert USB tactile dataset.

## Source Data

Default source directory:

```text
/shareNFS_40/wyf/data/UniVTAC/sim/Insert_USB_Tactile
```

The directory currently contains one HDF5 file per successful simulated
episode, for example:

```text
2.hdf5
3.hdf5
4.hdf5
...
```

Observed dataset summary:

```text
episode count: 101
total frames: 22043
frames per episode: roughly 197-251
```

The collection entry point is:

```text
/shareNFS_40/wyf/UniVTAC/scripts/collect_data.py
```

Frames were saved every 2 simulator steps in the inspected data. With the base
sim dt of `1/120`, the natural saved-frame FPS is approximately `60`. If the
converter downsamples frames, update the LeRobot FPS metadata accordingly.

## HDF5 Layout

Core fields observed in `2.hdf5`:

```text
observation/head/rgb
observation/wrist/rgb

tactile/left_tactile/rgb
tactile/left_tactile/rgb_marker
tactile/left_tactile/depth
tactile/left_tactile/marker
tactile/left_tactile/pose

tactile/right_tactile/rgb
tactile/right_tactile/rgb_marker
tactile/right_tactile/depth
tactile/right_tactile/marker
tactile/right_tactile/pose

embodiment/joint
embodiment/ee

actor/prism
actor/slot
actor/start_slot

atom/id
atom/tag
step
```

Image fields whose key contains `rgb` are stored as JPEG byte streams, not as
raw `[T, H, W, 3]` arrays.

Decoded image sizes:

```text
observation/head/rgb              -> [270, 480, 3]
observation/wrist/rgb             -> [270, 480, 3]
tactile/*/rgb                     -> [240, 320, 3]
tactile/*/rgb_marker              -> [240, 320, 3]
```

## Image Channel Warning

The source arrays are RGB before JPEG encoding, but `cv2.imencode(".jpg", img)`
was used directly during HDF5 writing. When reading back with
`cv2.imdecode(..., cv2.IMREAD_COLOR)`, the numeric channel order that looks
correct for this dataset is the direct decoded array.

Do not apply `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` for this simulated dataset.
Doing so turns the red USB blue and the cyan slot yellow.

Recommended HDF5 image decode:

```python
buf = np.frombuffer(h5_file["observation/head/rgb"][t], dtype=np.uint8)
image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
```

The resulting `image` should be treated as the RGB image for LeRobot writing.

## Observation Mapping

Recommended LeRobot image fields:

```text
image                <- observation/head/rgb
wrist_image          <- observation/wrist/rgb
left_tactile_image   <- tactile/left_tactile/rgb_marker
right_tactile_image  <- tactile/right_tactile/rgb_marker
```

Use `rgb_marker` for tactile images first, since it matches the visual marker
images used by the current real-data OpenPI configs.

## Embodiment Fields

`embodiment/joint` has shape `[T, 9]`:

```text
[franka_joint_1, ..., franka_joint_7, left_finger, right_finger]
```

The two finger values are effectively the same in the inspected episodes. Store
one scalar gripper value only. Use the right finger:

```python
gripper_q = joint[:, 8:9]
```

`embodiment/ee` has shape `[T, 7]`:

```text
[x, y, z, qw, qx, qy, qz]
```

Important: the simulated EEF quaternion is `wxyz`. This differs from the real
raw `states.npz`, where `eef_quat` was `xyzw`.

Before using SciPy `Rotation.from_quat`, reorder to `xyzw`:

```python
ee = h5_file["embodiment/ee"][()]
eef_pos = ee[:, :3]
quat_wxyz = ee[:, 3:7]
quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
```

## State And Action

Use the same absolute EEF 10D format as the real Insert USB LeRobot dataset:

```text
state   = [eef_pos(3), eef_rot6d(6), gripper_q(1)]
actions = [target_eef_pos(3), target_eef_rot6d(6), target_gripper_q(1)]
```

`eef_rot6d` should be generated from the EEF quaternion by converting to a
rotation matrix and concatenating the first two matrix columns:

```python
rot_mats = Rotation.from_quat(quat_xyzw).as_matrix()
eef_rot6d = np.concatenate([rot_mats[:, :, 0], rot_mats[:, :, 1]], axis=1)
```

The first converter version should use absolute EEF targets:

```text
actions[t] = state[t + target_offset]
```

with `target_offset=1` by default. The last frames should repeat the final valid
target, same as the real-data converter.

The resulting state/action dimension is:

```text
3 + 6 + 1 = 10
```

OpenPI will later pad this to the model action dimension internally.

## Suggested Output

Suggested repo id for the first simulated LeRobot dataset:

```text
wyf/insert_usb_sim_abs_eef
```

Suggested local output:

```text
/shareNFS_40/wyf/data/lerobot/wyf/insert_usb_sim_abs_eef
```

The prompt can stay aligned with the current real-data LoRA configs:

```text
pick and insert the USB
```

## OpenPI Configs

The first active OpenPI training config for the simulated dataset is the
two-camera LoRA version:

```text
pi05_insert_usb_sim_abs_eef_vision_lora
```

It uses:

```text
repo_id="wyf/insert_usb_sim_abs_eef"
state/action dim = 10
action_horizon = 10
batch_size = 16
prompt = "pick and insert the USB"
```

After converting the LeRobot dataset, compute normalization stats:

```bash
cd /shareNFS_40/wyf/workplace/openpi
export HF_LEROBOT_HOME=/shareNFS_40/wyf/data/lerobot

uv run scripts/compute_norm_stats.py \
  --config-name pi05_insert_usb_sim_abs_eef_vision_lora
```

Then train:

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS=--xla_gpu_autotune_level=0 \
uv run scripts/train.py \
  pi05_insert_usb_sim_abs_eef_vision_lora \
  --exp-name insert_usb_sim_vision_lora_v1 \
  --overwrite \
  --num-train-steps 30000 \
  --save-interval 5000
```

A four-image simulated LoRA config should use the same repo id and add tactile
image keys after this two-camera version is checked.
