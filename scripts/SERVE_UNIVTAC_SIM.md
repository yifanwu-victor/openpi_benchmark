# UniVTAC Sim Policy Server

本文档记录 UniVTAC 仿真测评时，OpenPI policy server 的启动方式和当前可用的 sim checkpoint。

server 只负责：

```text
observation -> OpenPI policy inference -> action_chunk
```

server 不负责启动 IsaacLab/UniVTAC 环境，不负责采集相机或触觉图像，也不负责执行 Franka action。仿真侧 client 需要通过 websocket 连接 server，发送 observation，并消费 server 返回的 action chunk。

## 通用启动方式

从 OpenPI repo 根目录启动：

```bash
cd /shareNFS_40/wyf/workplace/openpi
source ~/.bashrc

CUDA_VISIBLE_DEVICES=<GPU_ID> \
OPENPI_DATA_HOME=/shareNFS_40/wyf/data/openpi \
XLA_FLAGS=--xla_gpu_autotune_level=0 \
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=<CONFIG_NAME> \
  --policy.dir=<CHECKPOINT_DIR>
```

注意 `--port` 是 `serve_policy.py` 的顶层参数，必须放在 `policy:checkpoint` 前面。`--policy.config` 和 `--policy.dir` 属于 `policy:checkpoint`，放在后面。

server 默认监听：

```text
0.0.0.0:8000
```

如果同一台机器上已经有另一个 server 占用 `8000`，改成 `--port 8001`，同时把 UniVTAC deploy yaml 里的 `openpi.port` 改成同一个端口。

## Observation / Action 约定

当前四个 sim checkpoint 都是 `abs_joint` 接口：

```text
state/action dim = 8
state/action = [7 Franka arm joints, 1 gripper qpos]
```

server 返回：

```python
{
    "actions": np.ndarray  # shape [20, 8]
}
```

其中 `20` 来自当前 config 的 `action_horizon=20`。仿真侧可以设置自己的 `open_loop_horizon`，但不能超过 server 返回的 chunk 长度。

图像默认应按 RGB 的 `uint8 HWC` 数组发送：

```text
[H, W, 3], np.uint8, RGB
```

OpenPI server 会做 resize/pad 到 `224x224`，并把 `uint8 [0,255]` 转成模型需要的 `float32 [-1,1]`。server 不会自动做 BGR/RGB 通道交换；如果颜色不对，需要在 UniVTAC client 侧通过 `image_color_order` 调整。

两路 vision config 只使用：

```text
observation/image
observation/wrist_image
observation/state
prompt
```

如果 client 同时发送 tactile 图像，两路 config 会忽略多出来的 tactile 字段。

四路 vision+tactile config 使用：

```text
observation/image
observation/wrist_image
observation/left_tactile_image
observation/right_tactile_image
observation/state
prompt
```

## Insert USB: 两路 Vision

Config:

```text
pi05_insert_usb_sim_abs_joint_vision_lora
```

Dataset:

```text
/shareNFS_40/wyf/data/lerobot/wyf/insert_usb_sim_abs_joint
```

Default prompt:

```text
pick and insert the USB
```

Recommended checkpoint:

```text
/shareNFS_40/wyf/workplace/openpi/checkpoints/pi05_insert_usb_sim_abs_joint_vision_lora/insert_usb_sim_abs_joint_vision_lora_v1/19999
```

Serve command:

```bash
cd /shareNFS_40/wyf/workplace/openpi
source ~/.bashrc

CUDA_VISIBLE_DEVICES=3 \
OPENPI_DATA_HOME=/shareNFS_40/wyf/data/openpi \
XLA_FLAGS=--xla_gpu_autotune_level=0 \
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_insert_usb_sim_abs_joint_vision_lora \
  --policy.dir=/shareNFS_40/wyf/workplace/openpi/checkpoints/pi05_insert_usb_sim_abs_joint_vision_lora/insert_usb_sim_abs_joint_vision_lora_v1/19999
```

## Insert USB: 四路 Vision + Tactile

Config:

```text
pi05_insert_usb_sim_abs_joint_vision_tactile_lora
```

Dataset:

```text
/shareNFS_40/wyf/data/lerobot/wyf/insert_usb_sim_abs_joint
```

Default prompt:

```text
pick and insert the USB
```

Recommended checkpoint:

```text
/shareNFS_40/wyf/workplace/openpi/checkpoints/pi05_insert_usb_sim_abs_joint_vision_tactile_lora/insert_usb_sim_abs_joint_vision_tactile_lora_v1/20000
```

Serve command:

```bash
cd /shareNFS_40/wyf/workplace/openpi
source ~/.bashrc

CUDA_VISIBLE_DEVICES=3 \
OPENPI_DATA_HOME=/shareNFS_40/wyf/data/openpi \
XLA_FLAGS=--xla_gpu_autotune_level=0 \
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_insert_usb_sim_abs_joint_vision_tactile_lora \
  --policy.dir=/shareNFS_40/wyf/workplace/openpi/checkpoints/pi05_insert_usb_sim_abs_joint_vision_tactile_lora/insert_usb_sim_abs_joint_vision_tactile_lora_v1/20000
```

## Swap Cup Order: 两路 Vision

Config:

```text
pi05_swap_cup_order_sim_abs_joint_vision_lora
```

Dataset:

```text
/shareNFS_40/wyf/data/lerobot/wyf/swap_cup_order_sim_abs_joint
```

Default prompt:

```text
swap the cup order
```

Recommended checkpoint:

```text
/shareNFS_40/wyf/workplace/openpi/checkpoints/pi05_swap_cup_order_sim_abs_joint_vision_lora/swap_cup_order_sim_abs_joint_vision_lora_v1/19999
```

Serve command:

```bash
cd /shareNFS_40/wyf/workplace/openpi
source ~/.bashrc

CUDA_VISIBLE_DEVICES=3 \
OPENPI_DATA_HOME=/shareNFS_40/wyf/data/openpi \
XLA_FLAGS=--xla_gpu_autotune_level=0 \
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_swap_cup_order_sim_abs_joint_vision_lora \
  --policy.dir=/shareNFS_40/wyf/workplace/openpi/checkpoints/pi05_swap_cup_order_sim_abs_joint_vision_lora/swap_cup_order_sim_abs_joint_vision_lora_v1/19999
```

## Swap Cup Order: 四路 Vision + Tactile

Config:

```text
pi05_swap_cup_order_sim_abs_joint_vision_tactile_lora
```

Dataset:

```text
/shareNFS_40/wyf/data/lerobot/wyf/swap_cup_order_sim_abs_joint
```

Default prompt:

```text
swap the cup order
```

Recommended checkpoint:

```text
/shareNFS_40/wyf/workplace/openpi/checkpoints/pi05_swap_cup_order_sim_abs_joint_vision_tactile_lora/swap_cup_order_sim_abs_joint_vision_tactile_lora_v1/19999
```

Serve command:

```bash
cd /shareNFS_40/wyf/workplace/openpi
source ~/.bashrc

CUDA_VISIBLE_DEVICES=3 \
OPENPI_DATA_HOME=/shareNFS_40/wyf/data/openpi \
XLA_FLAGS=--xla_gpu_autotune_level=0 \
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_swap_cup_order_sim_abs_joint_vision_tactile_lora \
  --policy.dir=/shareNFS_40/wyf/workplace/openpi/checkpoints/pi05_swap_cup_order_sim_abs_joint_vision_tactile_lora/swap_cup_order_sim_abs_joint_vision_tactile_lora_v1/19999
```

## Client 连接

UniVTAC 仿真侧 deploy yaml 应设置：

```yaml
policy_name: openpi

openpi:
  host: "<openpi_server_ip>"
  port: 8000
  prompt: "<task prompt>"
  image_size: 224
  send_tactile: true
  image_color_order: "rgb"
```

如果启动的是两路 vision server，`send_tactile` 可以为 `true` 或 `false`；即使发送 tactile，server 也会忽略。若启动四路 vision+tactile server，`send_tactile` 必须为 `true`。

## Debug

需要记录 server 实际收到的 observation 和输出时，可以启动时加：

```bash
--record
```

例如：

```bash
uv run scripts/serve_policy.py \
  --port 8000 \
  --record \
  policy:checkpoint \
  --policy.config=<CONFIG_NAME> \
  --policy.dir=<CHECKPOINT_DIR>
```

记录会写到：

```text
policy_records/
```
