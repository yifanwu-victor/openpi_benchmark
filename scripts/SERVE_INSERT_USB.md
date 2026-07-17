# Insert USB Policy Server

This note records how to start the OpenPI policy server for the current Insert USB checkpoints.

The server only handles:

```text
observation -> policy inference -> action_chunk
```

It does not collect camera images, tactile images, Franka state, or run `deoxys_control`. The real robot runtime should run as a websocket client, send observations to this server, receive an action chunk, and execute the actions with its own safety/control logic.

## Common Command

Use the existing OpenPI entrypoint:

```bash
CUDA_VISIBLE_DEVICES=2 \
XLA_FLAGS=--xla_gpu_autotune_level=0 \
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=<CONFIG_NAME> \
  --policy.dir=<CHECKPOINT_DIR>
```

Insert USB does not need a separate serve script for now. Switch versions by changing `--policy.config` and `--policy.dir`.
With Tyro, top-level options such as `--port`, `--record`, and `--default-prompt` must appear before the `policy:checkpoint` subcommand. `--policy.config` and `--policy.dir` belong after `policy:checkpoint`.

## LoRA Vision

Config:

```text
pi05_insert_usb_real_abs_eef_vision_lora_single_gpu
```

Recommended checkpoint:

```text
checkpoints/pi05_insert_usb_real_abs_eef_vision_lora_single_gpu/insert_usb_lora_single_gpu_v1/15000
```

Serve command:

```bash
CUDA_VISIBLE_DEVICES=2 \
XLA_FLAGS=--xla_gpu_autotune_level=0 \
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_insert_usb_real_abs_eef_vision_lora_single_gpu \
  --policy.dir=checkpoints/pi05_insert_usb_real_abs_eef_vision_lora_single_gpu/insert_usb_lora_single_gpu_v1/15000
```

## Vision Action Expert FT

Config:

```text
pi05_insert_usb_real_abs_eef_vision_action_expert_ft
```

Recommended checkpoint:

```text
checkpoints/pi05_insert_usb_real_abs_eef_vision_action_expert_ft/insert_usb_action_expert_ft_v1/11000
```

Serve command:

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_FLAGS=--xla_gpu_autotune_level=0 \
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_insert_usb_real_abs_eef_vision_action_expert_ft \
  --policy.dir=checkpoints/pi05_insert_usb_real_abs_eef_vision_action_expert_ft/insert_usb_action_expert_ft_v1/11000
```

## Vision + Tactile Action Expert FT

Config:

```text
pi05_insert_usb_real_abs_eef_vision_tactile_action_expert_ft
```

Recommended checkpoint:

```text
checkpoints/pi05_insert_usb_real_abs_eef_vision_tactile_action_expert_ft/insert_usb_vision_tactile_action_expert_ft_v1/8000
```

Serve command:

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_FLAGS=--xla_gpu_autotune_level=0 \
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_insert_usb_real_abs_eef_vision_tactile_action_expert_ft \
  --policy.dir=checkpoints/pi05_insert_usb_real_abs_eef_vision_action_expert_ft/insert_usb_action_expert_ft_v1/10000
```

## Notes

- Checkpoint steps can be replaced if another step is preferred. For example, change `/8000` to `/5000`.
- `HF_LEROBOT_HOME` is usually not important for serving because the server reads checkpoint assets and normalization stats from the checkpoint directory, not from the LeRobot training dataset.
- If another process already uses port `8000`, change `--port`, for example `--port 8001`.
- Add `--record` if policy input/output records are needed for debugging.

## Client Connection

The robot runtime should connect as a websocket client:

```python
from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy(host="<server_ip>", port=8000)
action_chunk = policy.infer(observation)["actions"]
```

The observation keys must match the selected config. The server returns an action chunk; the robot runtime is responsible for action scheduling, safety checks, and command execution.
