# Sparsh DINOv2 Tactile Encoder Plan

Date: 2026-08-23

This note sketches how to add a Sparsh DINOv2 tactile encoder to OpenPI after the AnyTouch2 v1 path.

## Source Project

Local Sparsh repo:

```text
/shareNFS_40/wyf/workplace/sparsh
```

Relevant files:

```text
tactile_ssl/model/vision_transformer.py
tactile_ssl/downstream_task/sl_module.py
config/model/dinov2_vit.yaml
config/experiment/dinov2_vit.yaml
config/data/gelsight.yaml
```

The README points to the pretrained DINOv2 base backbone:

```text
https://huggingface.co/facebook/sparsh-dinov2-base
```

## What Sparsh DINOv2 Looks Like

From the local config/code:

```text
model: vit_base
embed_dim: 768
patch_size: 16
input image: 224x224
patch tokens: 14 * 14 = 196
register tokens: 1 in the DINOv2 config
```

`VisionTransformer.forward_features(x)` returns:

```python
{
    "x_norm_regtokens": ...,
    "x_norm_patchtokens": ...,
    "x_prenorm": ...,
    "masks": ...,
}
```

For OpenPI, the useful outputs are:

```text
register token(s) + patch tokens
```

A first conservative token layout per tactile side:

```text
1 register + 30 compressed patch tokens = 31 tokens
```

Left + right tactile:

```text
62 tactile tokens
```

Then project:

```text
Sparsh dim 768 -> OpenPI prefix dim 2048
```

## Proposed OpenPI Architecture

Keep the current OpenPI vision path:

```text
base image -> PaliGemma SigLIP
wrist image -> PaliGemma SigLIP
language -> PaliGemma tokenizer/language tokens
```

Route tactile images through Sparsh:

```text
left tactile image -> Sparsh DINOv2
right tactile image -> Sparsh DINOv2
```

Prefix order should match the current tactile-first experiments:

```text
left tactile -> right tactile -> base image -> wrist image -> language
```

## Data Conversion

Sparsh GelSight should use the same temporal input style as the local Sparsh config:

```text
num_frames: 2
frame_stride: 5
out_format: concat_ch_img
```

So this path should use a new LeRobot conversion format instead of reusing the ordinary single-frame tactile image fields.

Recommended first dataset format:

```text
image
wrist_image
left_tactile_sparsh
right_tactile_sparsh
state
actions
task
```

For each selected frame `t`, build tactile inputs as:

```text
prev = max(0, t - 5)
left_tactile_sparsh  = concat(left_tactile[prev],  left_tactile[t],  channel_axis)  # [224, 224, 6]
right_tactile_sparsh = concat(right_tactile[prev], right_tactile[t], channel_axis)  # [224, 224, 6]
```

This is different from AnyTouch2:

```text
AnyTouch2: [t-6, t-4, t-2, t] as a 4-frame clip
Sparsh:    [t-5, t] concatenated into one 6-channel image
```

For Insert USB v3 relative joint training:

```text
state   = [joint7, gripper]                # 8 dims, absolute
actions = [target joint7, target gripper]  # 8 dims, absolute
```

The OpenPI config should keep:

```text
delta_joint_actions=True
```

so training sees:

```text
[target joint7 - current joint7, abs gripper]
```

This means Sparsh DINOv2 should use a separate LeRobot dataset, for example:

```text
simulation/insert_usb_v3_gelsight_abs_joint_sparsh_2f
```

Do not overload the existing `left_tactile_image` / `right_tactile_image` fields with 6-channel arrays; those names are already used elsewhere as normal RGB images. Use explicit Sparsh field names so the model transform can route them safely.

## Model Code Organization

Add a separate module instead of modifying AnyTouch2:

```text
src/openpi/models_pytorch/sparsh/
```

Suggested files:

```text
encoder.py              # OpenPI-facing SparshDINOv2TactileEncoder
vision_transformer.py   # minimal migrated Sparsh ViT code, if not using package import
layers/                 # only if local ViT dependencies are needed
README.md               # checkpoint and token-shape notes
```

Two possible implementation strategies:

### Option A: Vendor Minimal Sparsh Code

Copy only the required ViT/layer code from:

```text
/shareNFS_40/wyf/workplace/sparsh/tactile_ssl/model/
```

Pros:

```text
OpenPI does not depend on the sparsh repo being on PYTHONPATH.
Training/serve are reproducible from openpi alone.
```

Cons:

```text
Need to copy several layer files and keep them in sync manually.
```

### Option B: Import Local Sparsh Package

Install sparsh editable into the OpenPI environment or append path.

Pros:

```text
Less code migration.
Easier to try quickly.
```

Cons:

```text
Serve/train depends on external repo layout.
Environment becomes more fragile.
```

Recommendation: start with Option B for a one-step smoke test, then vendor minimal code once the output shape and checkpoint loading are verified.

## Config Fields

Extend `Pi0Config` in the same style as AnyTouch2:

```python
tactile_encoder_type="sparsh_dinov2"
sparsh_checkpoint_path=<required>
sparsh_model_size="base"
sparsh_image_size=224
sparsh_in_chans=6
sparsh_frame_offsets=(-5, 0)
sparsh_patch_tokens_per_side=30
sparsh_keep_register_tokens=True
sparsh_trainable=True
sparsh_remove_background=False
```

If sharing names with future Sparsh MAE/DINO/IJEPA is likely, use:

```python
sparsh_backbone="dinov2"
```

## PI0Pytorch Routing

In `src/openpi/models_pytorch/pi0_pytorch.py`:

```text
if tactile_encoder_type == "sparsh_dinov2":
    self.sparsh_encoder = SparshDINOv2TactileEncoder(...)
```

In `embed_prefix()`:

```text
for key in tactile_sparsh_keys:
    tactile_emb = self.sparsh_encoder(observation.tactile_sparsh[key])
```

The wrapper should accept either:

```text
[B, H, W, 6]
[B, 6, H, W]
```

and return:

```text
[B, 31, 2048] per tactile side
```

## Preprocessing

Sparsh configs use resize to 224. The converter should write `224x224x6` tactile tensors so train/serve see the same shape. The Sparsh wrapper should normalize inside the model path.

Open questions to verify before training:

```text
Does facebook/sparsh-dinov2-base expect ImageNet normalization or Sparsh-specific normalization?
Does the pretrained backbone expect GelSight Mini style background-removed images?
Does our UniVTAC GelSight color distribution need BGR/RGB conversion?
```

The local `config/data/gelsight.yaml` has:

```text
remove_bg: true
out_format: concat_ch_img
num_frames: 2
frame_stride: 5
resize: [224, 224]
```

For the OpenPI v1 Sparsh path, use two-frame channel concatenation by default. Background subtraction can be a separate flag:

```text
--sparsh-remove-background none|first_frame
```

The most conservative first implementation is `none`, because online deployment is easier: the client only needs a rolling previous tactile frame and does not need a calibrated background image. A closer-to-Sparsh variant can subtract the first tactile frame of each episode before concatenation.

## Checkpoint Loading

Sparsh downstream code loads DINO checkpoints using key prefix:

```text
teacher_encoder.backbone
```

So a loader should support at least:

```text
checkpoint["model"]["teacher_encoder.backbone.*"]
checkpoint["model"]["backbone.*"]
plain state_dict
```

For HF checkpoints, first inspect downloaded files before finalizing exact key conversion.

Recommended local path:

```text
/shareNFS_40/wyf/data/openpi/sparsh/checkpoints/sparsh-dinov2-base
```

## Training Config Proposal

First config:

```text
pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_sparsh_dinov2_torch_full_ft
```

Use:

```text
repo_id="simulation/insert_usb_v3_gelsight_abs_joint_sparsh_2f"
state_dim=8
output_action_dim=8
delta_joint_actions=True
image_keys=("base_0_rgb", "wrist_0_rgb")
tactile_sparsh_keys=("left_tactile_sparsh", "right_tactile_sparsh")
tactile_encoder_type="sparsh_dinov2"
batch_size=64
num_train_steps=10000
save_interval=5000
keep_period=5000
```

Torch-only, matching AnyTouch2.

## Serve / Client Contract

Because Sparsh uses two tactile frames, the online client should maintain a small rolling buffer and send explicit 6-channel tactile fields:

```text
observation/image
observation/wrist_image
observation/left_tactile_sparsh   # [224, 224, 6], concat(t-5, t)
observation/right_tactile_sparsh  # [224, 224, 6], concat(t-5, t)
observation/state
prompt
```

This keeps the policy server stateless. The server remains a pure `observation -> action_chunk` service and does not need to remember previous frames.

Server output remains relative-joint-trained but served as absolute joint through `AbsoluteActions`:

```text
actions: [T, 8] = [abs joint7, abs gripper]
```

## Risks

1. Sparsh dependency may require `xformers`; current OpenPI torch environment may or may not have a compatible build.
2. HF checkpoint key names need inspection.
3. Background subtraction policy may matter; `none` is easiest for deployment, while `first_frame` is closer to Sparsh training assumptions.
4. Full fine-tuning Sparsh + OpenPI may increase memory less than dual SigLIP, but DDP behavior still needs smoke testing.

## First Smoke Tests

1. Download checkpoint and inspect keys.
2. Instantiate Sparsh encoder only:

```text
input:  [B, 224, 224, 6]
output: [B, 31, 2048]
```

3. OpenPI forward smoke:

```text
1 GPU, batch=2, 1 step
```

4. DDP smoke:

```text
4 GPU, batch=64, 1 step
```

5. Then full run.
