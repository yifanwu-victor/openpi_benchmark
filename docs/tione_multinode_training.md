# TI-ONE 双机 16 卡训练指南

本文档记录在腾讯云 TI-ONE 上使用本仓库进行 JAX/OpenPI 双机 16 卡训练的完整流程。当前目标任务是使用两台 8 卡 CVM 节点，在同一个 TI-ONE 资源组内训练 `pi05_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft`。

## 总体流程

推荐流程是先在开发机中调试，再使用训练任务正式运行：

```text
8 卡开发机调通单机训练
        ↓
确认代码、数据、权重、norm stats 都在 CFS
        ↓
保存开发机镜像
        ↓
停止开发机，释放 GPU
        ↓
创建 TI-ONE 训练任务：2 节点 x 每节点 8 GPU
        ↓
训练任务挂载同一个 CFS 到 /home/tione/notebook
        ↓
运行双机训练脚本
```

开发机用于交互式调试；正式多机训练应该通过 TI-ONE 训练任务提交。不要用两个开发机手动拼多机训练，除非只是临时排查网络或框架问题。

## 当前资源与路径

训练任务依赖以下固定路径。训练任务必须挂载同一个 CFS，并保持挂载路径为 `/home/tione/notebook`。

| 类型 | 路径 | 说明 |
|---|---|---|
| 仓库 | `/home/tione/notebook/openpi_benchmark` | 代码、脚本、uv `.venv` |
| LeRobot 数据 | `/home/tione/notebook/data/lerobot/simulation/insert_block_gelsight_delta_eef` | 转换后的训练数据 |
| OpenPI cache | `/home/tione/notebook/data/openpi` | `OPENPI_DATA_HOME` |
| pi05 base 权重 | `/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base/params` | 全参微调初始权重 |
| HF cache | `/home/tione/notebook/data/hf_cache` | `HF_HOME` / datasets cache |
| 训练输出 | `/home/tione/notebook/checkpoints` | Orbax checkpoint 输出根目录 |
| JAX cache | `/home/tione/notebook/data/jax_cache` | 多机任务共享编译缓存 |

这些路径都应该位于 CFS 上。当前开发机中 `/home/tione/notebook` 挂载为 NFS/CFS；`/root` 是镜像/容器层，不是 CFS。

## 镜像与 uv 环境

本仓库使用 `uv` 管理 Python 环境。当前 `.venv` 位于：

```bash
/home/tione/notebook/openpi_benchmark/.venv
```

但 `.venv/bin/python` 指向 uv 安装的 Python runtime：

```bash
/root/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11
```

因此，在停止开发机前必须先保存开发机镜像。保存镜像会把 `/root/.local/share/uv`、`/root/.local/bin/uv` 等系统层环境一起保存下来。训练任务需要选择这个保存出来的镜像，否则可能出现 `.venv/bin/python` 找不到解释器的问题。

保存镜像前建议在开发机里确认：

```bash
cd /home/tione/notebook/openpi_benchmark
export PATH="$HOME/.local/bin:$PATH"

readlink -f .venv/bin/python
uv run python - <<'PY'
import jax, openpi, lerobot
print("env ok")
print("jax", jax.__version__)
PY
```

本训练在新 GPU 上还需要避免系统 CUDA 12.4 的旧 `ptxas`。训练脚本会优先使用 `.venv` 中 Triton 自带的 `ptxas`：

```bash
/home/tione/notebook/openpi_benchmark/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin/ptxas
```

## 代码改动概览

为支持 TI-ONE 双机训练，仓库做了以下改动：

- `scripts/train.py`
  - 从环境变量初始化 `jax.distributed.initialize(...)`。
  - 单机运行时不启用 distributed，保持原有行为。
  - 多机时只让主进程写 wandb、进度条和首 batch 图片日志。

- `src/openpi/training/data_loader.py`
  - JAX 多进程时使用 `torch.utils.data.distributed.DistributedSampler`。
  - 每个 JAX process 读取自己的数据分片，避免两个节点重复读完全相同的 batch。

- `src/openpi/training/checkpoints.py`
  - 只有主进程清理/创建 checkpoint 目录。
  - 多进程同步后再创建 Orbax `CheckpointManager`，避免两个节点同时 `--overwrite` 抢删目录。

- `scripts/train_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_local.sh`
  - 单机/多机共用的本任务训练脚本。
  - 支持通过环境变量覆盖 batch size、steps、实验名等。
  - 多机时只有主进程计算 norm stats，其他进程等待结果。

- `scripts/train_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_tione_2x8.sh`
  - TI-ONE 双机 16 卡 wrapper。
  - 将 TI-ONE 注入的 `MASTER_ADDR`、`MASTER_PORT`、`RANK` / `NODE_RANK` 转成 JAX 所需的分布式环境变量。

## 关键参数

`--batch-size` 在本仓库中表示 global batch size，不是单卡 batch。

| 场景 | 总 GPU | `BATCH_SIZE` | 约等于每卡 batch |
|---|---:|---:|---:|
| debug 双机连通性 | 16 | 64 | 4 |
| 中等显存压力 | 16 | 128 | 8 |
| 保持每卡 16 | 16 | 256 | 16 |

第一次跑双机建议先用 `BATCH_SIZE=64` 和很少的 step 验证通信、数据读取和 checkpoint 保存。正式训练再使用 `BATCH_SIZE=256`。

常用环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EXP_NAME` | `insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_tione_2x8_v1` | 实验名，决定输出目录 |
| `BATCH_SIZE` | `64` | global batch size |
| `NUM_TRAIN_STEPS` | `20000` | 训练步数 |
| `SAVE_INTERVAL` | `5000` | checkpoint 保存间隔 |
| `KEEP_PERIOD` | `5000` | Orbax 保留周期 |
| `RESUME` | `0` | `1` 时续训，`0` 时覆盖同名输出 |
| `FSDP_DEVICES` | `8` | 每个 batch replica 内用于 FSDP 的设备数 |
| `GPU_IDS` | `0,1,2,3,4,5,6,7` | 每节点可见 GPU |
| `JAX_NUM_PROCESSES` | `2` | JAX 全局进程数，双机时为 2 |
| `JAX_PROCESS_INDEX` | 从 `NODE_RANK` / `RANK` 推导 | 当前节点 rank |
| `JAX_COORDINATOR_ADDRESS` | 从 `MASTER_ADDR:MASTER_PORT` 推导 | JAX coordinator 地址 |

## 提交 TI-ONE 训练任务

在 TI-ONE 中创建训练任务时，建议这样配置：

```text
训练镜像：保存后的开发机镜像
资源组：benchmark
节点数：2
每节点 GPU：8
CFS：挂载当前开发机使用的同一个 CFS
CFS 挂载路径：/home/tione/notebook
代码路径：/home/tione/notebook/openpi_benchmark
```

启动命令先使用 debug 版本：

```bash
cd /home/tione/notebook/openpi_benchmark
BATCH_SIZE=64 \
NUM_TRAIN_STEPS=20 \
SAVE_INTERVAL=10 \
EXP_NAME=debug_2x8_bs64_v1 \
bash scripts/train_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_tione_2x8.sh
```

debug 通过后，再提交正式训练：

```bash
cd /home/tione/notebook/openpi_benchmark
BATCH_SIZE=256 \
NUM_TRAIN_STEPS=20000 \
SAVE_INTERVAL=5000 \
EXP_NAME=insert_block_2x8_bs256_v1 \
bash scripts/train_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_tione_2x8.sh
```

如果需要续训：

```bash
cd /home/tione/notebook/openpi_benchmark
RESUME=1 \
BATCH_SIZE=256 \
EXP_NAME=insert_block_2x8_bs256_v1 \
bash scripts/train_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_tione_2x8.sh
```

不要对已有重要实验名使用默认 `RESUME=0`，因为脚本会传 `--overwrite`，同名输出目录会被清理。

## 训练输出

默认输出根目录：

```bash
/home/tione/notebook/checkpoints
```

正式训练示例 `EXP_NAME=insert_block_2x8_bs256_v1` 的输出目录为：

```bash
/home/tione/notebook/checkpoints/pi05_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft/insert_block_2x8_bs256_v1
```

Orbax checkpoint 中会包含：

- `train_state`
- `params`
- `assets` / norm stats
- wandb id 等辅助文件

这些内容写在 CFS 下，训练任务结束后仍会保留。

## 任务启动日志检查

训练任务刚启动时，优先看日志中是否出现：

```text
[tione-2x8] JAX_NUM_PROCESSES=2
[tione-2x8] JAX_PROCESS_INDEX=0
[tione-2x8] JAX_COORDINATOR_ADDRESS=<master>:23456
[jax-distributed] coordinator=<master>:23456 process=0/2 local_device_ids=(0, 1, 2, 3, 4, 5, 6, 7)
```

另一个节点应显示 `process=1/2`。如果两个节点都显示 `process=0/2`，说明 TI-ONE 注入的 rank 变量不是当前脚本识别的 `RANK` 或 `NODE_RANK`，需要查看任务环境变量并调整 wrapper 中的 `JAX_PROCESS_INDEX` 推导逻辑。

## 停开发机前检查

停开发机前必须完成：

```text
1. 保存开发机镜像完成
2. 数据在 /home/tione/notebook/data 下
3. 权重在 /home/tione/notebook/data/openpi 下
4. norm stats 已存在或可以由主进程计算
5. 训练任务会挂载同一个 CFS 到 /home/tione/notebook
```

检查命令：

```bash
cd /home/tione/notebook/openpi_benchmark
export PATH="$HOME/.local/bin:$PATH"

df -hT /home/tione/notebook
test -d /home/tione/notebook/data/lerobot/simulation/insert_block_gelsight_delta_eef
test -d /home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base/params
test -d assets/pi05_insert_block_gelsight_sim_delta_eef_vision_full_ft/simulation/insert_block_gelsight_delta_eef
bash -n scripts/train_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_tione_2x8.sh
uv run python - <<'PY'
import jax, openpi, lerobot
print("ready")
PY
```

## 常见问题

### 训练任务找不到 `uv`

确认训练镜像是保存后的开发机镜像，并且启动脚本中包含：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### `.venv/bin/python` 不存在或指向无效路径

说明训练任务镜像里没有保存 `/root/.local/share/uv/python/...`。重新回到开发机，确认 uv 环境可用后保存镜像。

### 报 `ptxas too old`

确认脚本优先使用 Triton 自带 `ptxas`：

```bash
TRITON_PTXAS_DIR=/home/tione/notebook/openpi_benchmark/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin
"$TRITON_PTXAS_DIR/ptxas" --version
```

### 多机初始化卡住

检查：

- 两个节点是否都启动了。
- `MASTER_ADDR` 是否是 rank 0 节点可访问的地址。
- 两个节点的 `JAX_NUM_PROCESSES` 是否都是 2。
- 两个节点的 `JAX_PROCESS_INDEX` 是否分别为 0 和 1。
- 防火墙/安全组是否允许节点间访问 `MASTER_PORT`。

### checkpoint 目录已存在

默认 `RESUME=0` 会覆盖同名实验目录。续训时使用：

```bash
RESUME=1 EXP_NAME=<existing_exp_name> bash scripts/train_insert_block_gelsight_sim_delta_eef_vision_tactile_full_ft_tione_2x8.sh
```

新实验建议改 `EXP_NAME`，不要复用旧名字。
