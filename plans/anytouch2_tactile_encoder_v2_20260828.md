# AnyTouch2 Tactile Encoder v2

Date: 2026-08-28

本文件记录当前已经落地的 OpenPI + AnyTouch2 PyTorch 版本，并替代
`anytouch2_tactile_encoder_v1_20260823.md` 作为后续开发基线。旧文档保留用于追踪最初方案。

## 当前目标与状态

AnyTouch2 已作为独立触觉 encoder 接入 OpenPI PyTorch 路径：

```text
left tactile clip  -> AnyTouch2 -> 36 tokens
right tactile clip -> AnyTouch2 -> 36 tokens
base image         -> PaliGemma SigLIP
wrist image        -> PaliGemma SigLIP
language           -> PaliGemma tokenizer / embedding
```

Prefix 顺序固定为：

```text
left AnyTouch 36 -> right AnyTouch 36 -> base SigLIP -> wrist SigLIP -> language
```

当前版本为 PyTorch-only，使用 `scripts/train_pytorch.py`。JAX vision-only 和 dual-SigLIP
路径不受影响。

## 模型与训练设置

当前实际设置是 full fine-tuning：

```text
PaliGemma / language model       trainable
main SigLIP                      trainable
Action Expert                    trainable
action/time projections          trainable
AnyTouch2 backbone               trainable
AnyTouch 512 -> 2048 projector   trainable
```

也就是说，当前 `anytouch_trainable=True`，不是 frozen。若后续需要冻结 AnyTouch backbone，
只需在目标 config 中设置 `anytouch_trainable=False`；projector 可继续保持
`anytouch_projector_trainable=True`。

AnyTouch 每侧保留：

```text
1 CLS + 5 sensor + 30 compressed patch tokens = 36 tokens
```

两侧共 72 个 tactile prefix tokens，projector 将 `512 -> 2048`。

公共训练参数：

```text
batch size       64
steps            10,000
save/keep        5,000
optimizer        AdamW, gradient clip 1.0
warmup           5,000
learning rate    5e-5
EMA              disabled
W&B project      openpi-anytouch
```

当前 AnyTouch configs 设置 `pytorch_compile_mode=None`，因此不会启用 `torch.compile`
或 CUDA Graph。后续若要测试 autotune，优先尝试
`max-autotune-no-cudagraphs`，不要直接使用会启用 CUDA Graph 的模式。

## 权重

AnyTouch2 checkpoint 已下载：

```text
/home/tione/notebook/data/openpi/openpi-assets/checkpoints/anytouch2/checkpoint-4frames.pth
```

本机 JAX Pi05 base 已转换为 PyTorch：

```text
/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors
```

Insert USB v3 已完成一轮 10k full FT：

```text
/home/tione/notebook/openpi_benchmark/checkpoints/
pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft/
insert_usb_v3_anytouch2_tactile_torch_full_ft_v1/10000
```

## 数据与归一化

LeRobot 数据继续保存单帧触觉，而不是把四帧 clip 重复写进数据集：

```text
image                  uint8 [224, 224, 3]
wrist_image            uint8 [224, 224, 3]
left_tactile_image     uint8 [224, 224, 3]
right_tactile_image    uint8 [224, 224, 3]
state                  float32 [8]
actions                float32 [8]
```

训练 data loader 使用 LeRobot `delta_timestamps` lazy 读取：

```text
[t-6, t-4, t-2, t]
```

历史索引限制在当前 episode 内，episode 开头自动 clamp，不跨 episode。

State/action 语义：

```text
LeRobot state   = [absolute joint7, absolute gripper]
LeRobot actions = [absolute target joint7, absolute target gripper]
Training action = [relative target joint7, absolute target gripper]
```

普通 RGB 使用 OpenPI 图像预处理。AnyTouch2 wrapper 内部使用固定 CLIP mean/std。
Norm stats 只包含 state/actions；当 repo ID、state/action 表达和 delta mask 相同时，可复用
已有 dual-SigLIP norm stats，无需为 AnyTouch 重算。

当前复用的数据集：

```text
/run/ti/pi-training-data/lerobot/simulation/insert_usb_v3_gelsight_abs_joint
/run/ti/pi-training-data/lerobot/simulation/insert_block_v1_gelsight_abs_joint
/run/ti/pi-training-data/lerobot/simulation/hardness_classify_gelsight_abs_joint
```

## 当前 Configs 与脚本

## Random AnyTouch2 消融版本（2026-09-02）

为对比 AnyTouch2 预训练初始化，新增旧版单 Action Expert 架构的随机初始化消融配置。
随机版本只跳过 AnyTouch2 checkpoint 加载；Pi05 PyTorch base、RGB SigLIP、PaliGemma 和
Action Expert 仍从 base 权重初始化，AnyTouch2 backbone 与 projector 全量训练。

随机实验的配置名统一使用：

```text
pi05_<task>_sim_abs_joint_relative_action_vision_tactile_anytouch2_random_torch_full_ft
```

五个组合/独立入口位于 `scripts/train/random_anytouch/`；公共 launcher 位于
`scripts/train/train_sim_abs_joint_random_anytouch2_full_ft.sh`。launcher 会在缺少数据 marker
时转换 LeRobot、缺少 stats 时计算或复用兼容 stats，并将 checkpoint 写入：

```text
/home/tione/notebook/openpi_benchmark/checkpoints/random_anytouch
```

该消融保持 AnyTouch2 v2 的 strict-clip 输入和单 Action Expert 十步采样，不使用 T-Rex 的
6+4 级联 Tactile Expert。五个入口按 weight/hardness、roughness classify/regrasp、
insert USB/block v1、pull drawer/turn gear 顺序执行，另有 Insert Block V2 独立 20k 脚本。

Configs：

```text
pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft
pi05_insert_block_v1_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft
pi05_hardness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft
```

训练脚本：

```text
scripts/train_insert_usb_v3_anytouch2_tactile_full_ft.sh
scripts/train_insert_block_v1_anytouch2_tactile_full_ft.sh
scripts/train_hardness_classify_anytouch2_tactile_full_ft.sh
scripts/train_hardness_then_insert_block_v1_anytouch2_tactile_full_ft.sh
```

顺序脚本先训练 Hardness Classify，再训练 Insert Block V1：

```bash
cd /home/tione/notebook/openpi_benchmark
bash scripts/train_hardness_then_insert_block_v1_anytouch2_tactile_full_ft.sh
```

默认 GPU：

```text
Hardness Classify  4,5,6,7
Insert Block V1    0,1,2,3
```

两个新任务脚本不转换 LeRobot、不计算 norm stats，只检查现有 dataset marker、schema、
norm stats、AnyTouch checkpoint 和 PyTorch base，然后等待 GPU 显存低于 20,000 MiB 后训练。

## Strict-Clip Serving 协议

AnyTouch serve 不再根据 websocket infer 请求拼历史。原因是 `open_loop_horizon > 1`
时 server 看不到每个环境步，无法还原训练使用的 `[t-6,t-4,t-2,t]`。

历史由 client 在每个环境步维护。每次真正请求模型时发送：

```python
{
    "observation/state": state,                       # float32 [8]
    "observation/image": base_image,                 # uint8 [224, 224, 3]
    "observation/wrist_image": wrist_image,          # uint8 [224, 224, 3]
    "observation/left_tactile_clip": left_clip,      # uint8 [4, 224, 224, 3]
    "observation/right_tactile_clip": right_clip,    # uint8 [4, 224, 224, 3]
    "prompt": prompt,
    "reset": True,                                   # only first request of episode
}
```

规则：

```text
clip order        oldest -> newest, [t-6,t-4,t-2,t]
dtype             uint8
layout            C-contiguous [4,224,224,3]
color             RGB
episode start     left-pad by repeating first frame
episode reset     client clears history
retry             must not append the same current frame twice
```

AnyTouch serve 使用 `TactileClipPolicy` 严格检查左右 clip，并拒绝单帧 tactile keys、缺少任意
一侧、float、channel-first、错误尺寸或非连续数组。`reset`/`episode_reset` 会在进入模型前移除；
server 本身不保存触觉历史。

普通 vision 和 dual-SigLIP checkpoint 仍使用原来的单帧接口：

```text
observation/left_tactile_image
observation/right_tactile_image
```

websocket/msgpack 层本身不定义 observation schema；具体接口由 checkpoint config、policy
wrapper 和 data transform 决定。目前没有 metadata 自动协商，client 必须知道连接的是普通
checkpoint 还是 AnyTouch checkpoint。

## 代码入口

```text
src/openpi/models_pytorch/anytouch2/encoder.py     AnyTouch preprocessing/projector
src/openpi/models_pytorch/pi0_pytorch.py           token dispatch/order
src/openpi/training/config.py                      model/data/train configs
src/openpi/training/data_loader.py                 lazy history timestamps
src/openpi/policies/sim_policy.py                  tactile clip transform
src/openpi/policies/policy.py                      strict TactileClipPolicy
src/openpi/policies/policy_config.py               AnyTouch wrapper selection
examples/anytouch2/README.md                        client protocol
```

## 已验证项目

```text
AnyTouch checkpoint load
JAX -> PyTorch Pi05 base conversion
AnyTouch output 36 tokens per side / 72 total
AnyTouch/projector gradient path
single-frame LeRobot lazy history
history does not cross episode
strict-clip shape/dtype/layout/reset handling
ordinary sim policy regression
config loading and reused norm stats
training launcher bash syntax and dry-run
Python compile, Ruff focused checks, git diff --check
```

## 后续可选优化

1. 单独实验 `anytouch_trainable=False`，只训练 projector 与 OpenPI 主干，比较显存、速度和效果。
2. 在单独 config 中尝试 `pytorch_compile_mode="max-autotune-no-cudagraphs"`，保留当前
   `None` config 作为稳定基线。
3. 若未来需要统一 client，可在 server metadata 中声明 `tactile_input_mode="clip"`、帧数和
   offsets；当前版本不做自动协商。
4. 正式评估 Insert USB v3、Insert Block V1、Hardness Classify 三个任务，并记录成功率、
   inference latency、峰值显存和不同 `open_loop_horizon` 的表现。
