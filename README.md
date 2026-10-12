# AnyTouch2 + π0.5：视觉、触觉、语言到相对关节动作

这个分支整理了 `openpi_benchmark` 中的 **AnyTouch2 token concat 全量微调流程**：HDF5 → LeRobot → relative-action 统计量 → PyTorch 训练 → WebSocket 推理。任务名称不是模型结构的一部分；换任务只需要换数据集 ID、语言指令和实验名。

实现对应原来的 `train_insert_block_v1_anytouch2_tactile_full_ft.sh`，也对应 `pi05_insert_block_v2_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft` 这类 checkpoint。这里不启用 T-Rex tactile expert、action-expert tactile token 或随机初始化实验。旧任务专用 shell 脚本和历史配置已从这个分支移除，原实验目录和 main 不受影响。

## 1. 模型如何接入

```text
左触觉 [B,4,224,224,3] ─┐
                       ├─ 共享 AnyTouch2 → 每侧 36×512 → Linear(512,2048)
右触觉 [B,4,224,224,3] ─┘

VLM prefix = concat(
    左触觉 36 tokens,
    右触觉 36 tokens,
    主相机 SigLIP tokens,
    腕相机 SigLIP tokens,
    语言 tokens,
)
    → PaliGemma → π0.5 action expert / flow matching → 50 步 action chunk
```

拼接沿 **token 序列维 `dim=1`** 进行，不是通道拼接。左右两侧使用同一个 AnyTouch2 encoder 和 projector。4 帧输入经 temporal kernel/stride=2 的 patch embedding 编码，保留 1 个 CLS + 5 个 sensor tokens，剩余 patch tokens 用线性插值压缩到 30 个，因此每侧共 36 个 token。512 维触觉特征投影到 VLM 的 2048 维。

触觉 clip 从当前 episode 的 `[t-6,t-4,t-2,t]` 取帧；这里的 2 是数据采样间隔，和 encoder 内部 temporal stride 是两件事。episode 开始处重复首帧补齐。AnyTouch2 使用固定 CLIP mean/std；普通相机使用 OpenPI 的图像预处理。

本配置为 PyTorch full fine-tuning：VLM、action expert、AnyTouch2 和 projector 都参与训练。保持原实验的 `pi05=True`、`discrete_state_input=False`、`action_horizon=50`、`pytorch_compile_mode=None`。因此本配置不把 state 编码为离散语言 token；state 仍用于 relative action 转换和推理输出还原。环境状态/动作是 8 维，模型内部 padding 到 32 维。

主要代码：

| 文件 | 用途 |
| --- | --- |
| `src/openpi/models_pytorch/anytouch2/tactile_mae.py` | AnyTouch2 时空 patch embedding、sensor tokens、checkpoint 加载 |
| `src/openpi/models_pytorch/anytouch2/encoder.py` | 图像归一化、patch 压缩、512→2048 投影 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | `embed_prefix()` 中 concat，flow-matching 训练与采样 |
| `src/openpi/training/config.py` | `make_anytouch2_config()`，一个通用配置替代任务列表 |
| `src/openpi/training/data_loader.py` | LeRobot 按时间偏移读取触觉历史与未来动作 |
| `src/openpi/policies/sim_policy.py` | 输入映射和固定 token 顺序 |
| `src/openpi/transforms.py` | `DeltaActions` / `AbsoluteActions`、归一化 |
| `scripts/anytouch2.py` | 统一 stats / train / serve 入口 |

## 2. 环境与预训练权重

使用 Linux、Python 3.11 和 CUDA。仓库按当前腾讯机器的已用版本更新了 `pyproject.toml` / `uv.lock`，其中 Torch 为 CUDA 12.8 nightly；这不是通用 CPU 环境。现有环境可复用；全新机器安装是否成功取决于锁定 nightly wheel 是否仍可下载。

```bash
git clone --branch anytouch2-clean git@github.com:yifanwu-victor/openpi_benchmark.git
cd openpi_benchmark
unset VIRTUAL_ENV
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen

# OpenPI 的 PyTorch backbone 需要仓库自带的 Transformers 补丁。
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_LEROBOT_HOME=/path/to/lerobot
export ANYTOUCH_CHECKPOINT=/path/to/checkpoint-4frames.pth
export PI05_BASE=/path/to/pi05_base_pytorch
export REPO_ID=simulation/my_task
```

`ANYTOUCH_CHECKPOINT` 是 AnyTouch2 4-frame `.pth` 文件；`PI05_BASE` 是包含 `model.safetensors` 的 **目录**。权重、数据集、归一化 assets、日志和输出 checkpoint 不提交到 Git。项目仍使用 OpenPI tokenizer；首次运行需要对应 tokenizer 缓存或下载能力。

如果只有 π0.5 JAX base checkpoint，可以使用保留的转换器：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /path/to/pi05_base \
  --output-path "$PI05_BASE" \
  --config-name pi05_base
```

完整版本表和验证范围见 [环境说明](docs/environment.md)。

## 3. HDF5 → LeRobot

转换器适用于 **UniVTAC 风格的仿真 HDF5**，不是任意机器人 HDF5 的自动识别器。每个文件是一个 episode，需要：

| HDF5 字段 | 含义 / LeRobot 输出 |
| --- | --- |
| `step` | 仿真步号，用于推断 FPS |
| `embodiment/joint` | `[N,9]`：7 个机械臂关节 + 左右夹爪关节 |
| `observation/head/rgb` | JPEG byte stream → `image` |
| `observation/wrist/rgb` | JPEG byte stream → `wrist_image` |
| `tactile/left_tactile/rgb_marker` | JPEG byte stream → `left_tactile_image` |
| `tactile/right_tactile/rgb_marker` | JPEG byte stream → `right_tactile_image` |
| attr `instruction` | 每条 episode 的语言指令 → LeRobot task |

也支持 `episode_context_json` 中的 instruction，或使用 `--task-prompt` 覆盖。新任务不需要修改 Python 配置名。

```bash
uv run examples/anytouch2/convert_univtac_sim_delta_eef_to_lerobot.py \
  --hdf5-dir /path/to/hdf5_episodes \
  --output-dir "$HF_LEROBOT_HOME/$REPO_ID" \
  --repo-id "$REPO_ID" \
  --control-mode abs_joint \
  --frame-stride 1 \
  --target-offset 1 \
  --gripper-mode left
```

文件名中的 `delta_eef` 是历史命名；**这条 relative-joint 流程必须使用 `abs_joint`**。默认图像为 224×224，每条样本只存一帧触觉，训练时再按需取 4 帧，不重复存整段 clip。输出已有时默认拒绝覆盖。`--dry-run` 检查首条 episode 而不写数据；`--max-episodes` 可用于小规模试跑。

state 是当前 `[q1,...,q7,gripper]`，action 是下一 **原始帧** 的绝对目标；`target-offset` 以原始帧计，末尾目标截断到最后一帧。若改 `frame-stride`，触觉历史跨度会按转换后的采样频率变化。默认 FPS 按仿真 120 Hz 和 step 间隔推算，也可显式传 `--fps`。

源数据的 JPEG 使用项目现有的 OpenCV 解码通道约定；换数据来源时应目视检查颜色，必要时修改 `examples/UniVTAC/convert_sim_to_lerobot.py` 的流映射和解码函数。

## 4. Relative VTLA 的动作语义与统计量

这里的 relative 指 **关节目标相对于当前观测的偏移**。对于时刻 t 的整个 50 步 chunk：

```text
relative_action[k, :7] = absolute_target[k, :7] - state_t[:7]
relative_action[k,  7] = absolute_target[k,  7]
```

每个未来目标都减去同一个 `state_t`，不是逐帧差分；夹爪保持绝对值。LeRobot 中保存绝对目标，`DeltaActions` 在训练和统计计算时统一转换，避免重复做 delta。π0.5 使用 quantile normalization。推理输出先反归一化，再加回当前 state，最终返回 `[50,8]` 的绝对关节目标。

```bash
uv run scripts/anytouch2.py \
  --mode stats --repo-id "$REPO_ID" \
  --batch-size 64 --num-workers 2
```

输出为 `assets/pi05_anytouch2_concat/$REPO_ID/norm_stats.json`。只统计 state 和经过 relative 转换的 actions，图像不参与统计。数据少于 batch size 时降低 `--batch-size`。`--max-frames` 适合调试，正式训练建议用完整数据。更换数据集、动作语义或 action horizon 后重新计算。

## 5. 训练任意任务

四卡 DDP 示例，global batch size 为 64，每卡 16：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun \
  --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/anytouch2.py \
  --mode train --repo-id "$REPO_ID" \
  --anytouch-checkpoint "$ANYTOUCH_CHECKPOINT" \
  --base-checkpoint "$PI05_BASE" \
  --exp-name my_task_v1 \
  --batch-size 64 --num-train-steps 10000
```

单卡使用 `CUDA_VISIBLE_DEVICES=0 uv run scripts/anytouch2.py` 加相同参数，按显存降低 batch size。DDP 的 global batch size 必须能被进程数整除。默认关闭 W&B，可加 `--wandb-enabled`。默认不覆盖已有实验目录；恢复训练加 `--resume`，或换一个 `--exp-name`。

保存目录是 `checkpoints/pi05_anytouch2_concat/my_task_v1/<step>/`，包括模型、optimizer、metadata 和 `assets/$REPO_ID/norm_stats.json`。训练器沿用零基步数：训练 10000 步的最后目录通常为 `9999`，20000 步通常为 `19999`；以实际保存目录为准。

`--assets-base-dir` 和 `--checkpoint-base-dir` 可自定义位置。先用 `--dry-run` 查看配置；它不加载模型或数据，也不验证训练是否能完整运行。

## 6. 推理与已有 checkpoint

```bash
CUDA_VISIBLE_DEVICES=7 uv run scripts/anytouch2.py \
  --mode serve --repo-id "$REPO_ID" \
  --anytouch-checkpoint "$ANYTOUCH_CHECKPOINT" \
  --checkpoint checkpoints/pi05_anytouch2_concat/my_task_v1/9999 \
  --port 9003
```

Serving 从训练 checkpoint 内加载归一化统计量，不依赖训练数据集目录；`repo-id` 必须与 checkpoint 下 `assets/<repo-id>` 匹配。当前模型构造仍需要 AnyTouch2 初始化文件，随后完整微调权重会覆盖它；不需要 π0.5 base 权重。

原 insert_block_v2 concat checkpoint 的例子：

```bash
CUDA_VISIBLE_DEVICES=7 uv run scripts/anytouch2.py \
  --mode serve \
  --repo-id simulation/insert_block_v2_gelsight_abs_joint \
  --anytouch-checkpoint "$ANYTOUCH_CHECKPOINT" \
  --checkpoint /home/tione/notebook/openpi_benchmark/checkpoints/pi05_insert_block_v2_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft/insert_block_v2_anytouch2_tactile_torch_full_ft_v1/20000 \
  --port 9003
```

客户端每次请求必须发送：

```python
observation = {
    "observation/state": state,                 # float32 [8]
    "observation/image": base_image,            # uint8 [224,224,3]
    "observation/wrist_image": wrist_image,     # uint8 [224,224,3]
    "observation/left_tactile_clip": left_clip,  # uint8 [4,224,224,3]
    "observation/right_tactile_clip": right_clip,
    "prompt": "insert the block into the slot",
}
```

clip 必须为连续内存 RGB uint8 数组，按 `[t-6,t-4,t-2,t]` 从旧到新排列。客户端应在 **每个环境步** 更新触觉历史，即使还在执行上一段 action chunk；episode 切换时清空并用首帧补齐。服务器不维护历史，拒绝单帧触觉、缺失 clip、错误 dtype 或 shape。协议包装在 `src/openpi/policies/policy.py` 的 `TactileClipPolicy`。

## 7. 验证与后续修改

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" uv run pytest -q \
  scripts/anytouch2_test.py \
  src/openpi/policies/policy_test.py \
  src/openpi/transforms_test.py \
  packages/openpi-client/src/openpi_client/websocket_client_policy_test.py \
  src/openpi/training/data_loader_test.py
```

测试覆盖 concat 顺序、history 时间偏移、HDF5 action target、relative chunk 及还原、推理 clip 协议。它们不替代完整训练或机器人评估。优化 token 压缩看 `encoder.py`；优化融合看 `embed_prefix()`；更换数据结构看 converter 和 history data config。改变模型结构时需要重新训练并检查旧权重兼容性。

基础代码来自 OpenPI；保留 `LICENSE` 和 `LICENSE_GEMMA.txt`。使用预训练权重时同时遵循相应权重的许可。
