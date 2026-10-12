# Environment baseline

The dependency files were updated against the environment on `tencent_v1` on 2026-10-12. This branch uses a newer Torch/CUDA stack than the original OpenPI setup; Transformers remains at the version required by the bundled model patch.

| Component | Version |
| --- | --- |
| Python | 3.11 |
| Torch | 2.12.0.dev20260408+cu128 |
| Torchvision (Linux x86_64) | 0.27.0.dev20260407+cu128 |
| Triton | 3.7.0+git282c8251 |
| Transformers | 4.53.2 + bundled transformers_replace |
| JAX / jaxlib | 0.6.2 |
| Flax | 0.10.2 |
| NumPy | 1.26.4 |
| W&B | 0.28.2 |
| h5py | 3.13.0 |
| safetensors | 0.5.3 |
| tyro | 0.9.22 |
| cuDNN (lock) | 9.20.0.48 |
| NCCL (lock) | 2.29.7 |
| LeRobot | git commit 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 |

The selected Linux versions above match the installed development environment. `uv lock --check` and `uv sync --locked --dry-run` pass. Runtime checks reuse that environment with the cleaned source on PYTHONPATH; a complete fresh environment installation and full GPU training have not been performed as part of this cleanup.

Install with `GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen`, then copy `src/openpi/models_pytorch/transformers_replace/*` into `.venv/lib/python3.11/site-packages/transformers/`. Repeat the patch after reinstalling Transformers. The standard package index is the Tsinghua mirror and the Torch stack uses the explicit PyTorch CUDA nightly index. Old nightly wheels may eventually disappear; preserve a wheel cache for long-term reproducibility.

Do not use `uv sync --upgrade` without checking Transformers patch compatibility, LeRobot schema/API compatibility, and checkpoint tensor shapes. The original experiment environment and data were not modified by this cleanup.
