# Torch Environment Notes

This repo uses two local uv environments for the current Blackwell machine.
Keep this note updated when changing `pyproject.toml` or `uv.lock`.

## Current Training Environment

Use the default `.venv` for Torch DDP training experiments.

```text
torch:       2.12.0.dev20260408+cu128
torchvision: 0.27.0.dev20260407+cu128
triton:      3.7.0+git282c8251
cuDNN:       9.20.0.48
NCCL:        2.29.7
index:       https://download.pytorch.org/whl/nightly/cu128
venv:        .venv
```

Observed result:

```text
4-GPU Torch DDP smoke passed.
config: pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_dual_siglip_tactile_first_torch_full_ft
global batch size: 64
per-GPU batch size: 16
num_train_steps: 1
loss: 0.0541
peak allocated memory on rank 0: 24.83 GB
```

The smoke command used:

```bash
cd /shareNFS_40/wyf/workplace/openpi

CUDA_VISIBLE_DEVICES=0,1,4,5 \
OPENPI_DATA_HOME=/shareNFS_40/wyf/data/openpi \
HF_HOME=/shareNFS_40/wyf/data/hf_cache \
HF_DATASETS_CACHE=/shareNFS_40/wyf/data/hf_cache/datasets \
HF_LEROBOT_HOME=/shareNFS_40/wyf/data/lerobot \
WANDB_MODE=disabled \
uv run python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_pytorch.py \
  pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_dual_siglip_tactile_first_torch_full_ft \
  --exp-name smoke_nightly_cu128_b64_4gpu \
  --checkpoint-base-dir /shareNFS_40/wyf/workplace/openpi/checkpoints_smoke \
  --num-train-steps 1 \
  --batch-size 64 \
  --save-interval 999999 \
  --keep-period 999999
```

## Stable Inference Candidate

Use `.venv-infer` as the candidate environment for serve / `torch.compile` /
cudagraph inference experiments. Always pass `--no-sync` so uv does not rewrite
`.venv-infer` according to the current nightly training `pyproject.toml`.

```text
torch:       2.11.0+cu128
torchvision: 0.26.0+cu128
triton:      3.6.0
cuDNN:       9.19.0.56
NCCL:        2.28.9
index:       https://download.pytorch.org/whl/cu128
venv:        .venv-infer
```

Observed result:

```text
CUDA tensor sanity test passed.
JAX GPU devices were visible.
4-GPU Torch DDP smoke failed with SIGSEGV after model creation.
```

Restore files for this state:
The stable dependency files are also kept as:

```text
pyproject.infer.toml
uv.infer.lock
```

Original restore files for this state:

```text
pyproject.toml.bak_before_nightly_cu128_20260822_112551
uv.lock.bak_before_nightly_cu128_20260822_112551
```

Restore command:

```bash
cd /shareNFS_40/wyf/workplace/openpi

cp pyproject.toml.bak_before_nightly_cu128_20260822_112551 pyproject.toml
cp uv.lock.bak_before_nightly_cu128_20260822_112551 uv.lock
UV_HTTP_TIMEOUT=300 uv sync
```

Inference command template:

```bash
cd /shareNFS_40/wyf/workplace/openpi

UV_PROJECT_ENVIRONMENT=.venv-infer \
uv run --no-sync scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=<CONFIG_NAME> \
  --policy.dir=<CHECKPOINT_DIR>
```

Infer environment sanity check:

```bash
cd /shareNFS_40/wyf/workplace/openpi

UV_PROJECT_ENVIRONMENT=.venv-infer uv run --no-sync python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_arch_list())
x = torch.randn(2, device="cuda")
print(x + 1)
PY
```

## Other Tested States

### torch 2.11.0+cu130

```text
torch:       2.11.0+cu130
torchvision: 0.26.0+cu130
cuDNN:       9.19.0.56
NCCL:        2.28.9
```

Observed result:

```text
CUDA tensor sanity test passed.
4-GPU Torch DDP smoke still failed with SIGSEGV after model creation.
JAX CUDA12 plugin reported cuPTI errors after mixing cu13 Torch dependencies.
```

Restore files for the state before this experiment:

```text
pyproject.toml.bak_before_cu130_20260821_160317
uv.lock.bak_before_cu130_20260821_160317
```

### torch 2.7.1+cu128

```text
torch:       2.7.1+cu128
torchvision: 0.22.1+cu128
cuDNN:       9.7.1.26
NCCL:        2.26.2
```

Observed result:

```text
CUDA tensor sanity test passed.
JAX reported a cuDNN minor-version mismatch because jaxlib expected a newer cuDNN.
This was not kept as the main environment.
```

Restore files for the state before this experiment:

```text
pyproject.toml.bak_before_revert_to_torch27_cu128_20260822_023920
uv.lock.bak_before_revert_to_torch27_cu128_20260822_023920
```

## Recommended Layout

Current single-repo layout:

```text
/shareNFS_40/wyf/workplace/openpi
  .venv        nightly cu128, Torch DDP training
  .venv-infer  stable 2.11 cu128, serve / torch.compile / cudagraph inference
```

`uv` caches downloaded wheels, but different exact Torch/CUDA/cuDNN/NCCL
versions are different wheel files. Switching among these states can still
require downloading or reinstalling large NVIDIA packages.

Default training uses the normal project environment:

```bash
cd /shareNFS_40/wyf/workplace/openpi
uv run scripts/train_pytorch.py ...
```

Inference uses the named environment and disables sync:

```bash
cd /shareNFS_40/wyf/workplace/openpi
UV_PROJECT_ENVIRONMENT=.venv-infer uv run --no-sync scripts/serve_policy.py ...
```
