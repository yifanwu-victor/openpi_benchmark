# AnyTouch2 Tactile Encoder

This integration uses the PyTorch trainer and fine-tunes the AnyTouch2 backbone
and the newly added `512 -> 2048` projector.

## Checkpoint

Download the AnyTouch2 `checkpoint-4frames.pth` file (the `hf-mirror.com`
endpoint is useful on machines that cannot reach Hugging Face directly) and
place it at:

```text
/home/tione/notebook/data/openpi/openpi-assets/checkpoints/anytouch2/checkpoint-4frames.pth
```

The insert USB v3 config uses this path:

```text
pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft
```

It initializes the rest of the model from the local PyTorch Pi05 base
checkpoint produced by `examples/convert_jax_model_to_pytorch.py`:

```text
/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors
```

## Data conversion

The converter stores one tactile frame per LeRobot sample. Training requests
the history `[t-6, t-4, t-2, t]` lazily through LeRobot's
`delta_timestamps`; indices are clamped inside the current episode, so history
never crosses an episode boundary.

```bash
cd /home/tione/notebook/openpi_benchmark
export HF_LEROBOT_HOME=/run/ti/pi-training-data/lerobot

uv run examples/anytouch2/convert_univtac_sim_delta_eef_to_lerobot.py \
  --hdf5-dir /home/tione/notebook/data/UniVTAC/sim/insert_USB/GelSight/noise3cm_1_5mm \
  --output-dir /run/ti/pi-training-data/lerobot/simulation/insert_usb_v3_gelsight_abs_joint \
  --repo-id simulation/insert_usb_v3_gelsight_abs_joint \
  --control-mode abs_joint \
  --overwrite
```

Run `uv run scripts/compute_norm_stats.py --config-name <config>` after
conversion. Only state and action statistics are computed; RGB and tactile
images use their model-specific fixed preprocessing. Start training with:

```bash
bash scripts/train_insert_usb_v3_anytouch2_tactile_full_ft.sh
```

The script invokes `scripts/train_pytorch.py`, uses project `openpi-anytouch`,
and expects the local checkpoint paths above to exist.

## Prefix and serving

The prefix order is fixed to:

```text
left AnyTouch2 (36) -> right AnyTouch2 (36) -> base SigLIP -> wrist SigLIP -> language
```

AnyTouch2 serving uses a strict direct-clip protocol. The client owns tactile
history and sends this payload for every policy request:

```python
{
    "observation/state": state,                       # float32 [8]
    "observation/image": base_image,                 # uint8 [224, 224, 3]
    "observation/wrist_image": wrist_image,          # uint8 [224, 224, 3]
    "observation/left_tactile_clip": left_clip,      # uint8 [4, 224, 224, 3]
    "observation/right_tactile_clip": right_clip,    # uint8 [4, 224, 224, 3]
    "prompt": prompt,
    "reset": True,                                   # first request of an episode only
}
```

Both clips must be C-contiguous RGB `uint8` arrays in oldest-to-newest order,
using frames `[t-6, t-4, t-2, t]`. The client updates its tactile history on
every environment step, even when it is still consuming a previous action
chunk and does not query the server. Before seven frames are available, it
left-pads history by repeating the first frame of the current episode. Episode
reset clears the client history, and retrying the same websocket request must
not insert the current frame a second time.

The AnyTouch2 server rejects single-frame tactile keys, missing left or right
clips, channel-first clips, floating-point clips, and shapes other than
`[4, 224, 224, 3]`. It does not keep tactile history. RGB images remain in
OpenPI's `[-1, 1]` convention after standard OpenPI preprocessing; AnyTouch2
applies its fixed CLIP mean and standard deviation internally. The response
protocol is unchanged and contains finite `actions` with shape `[T, 8]`.
