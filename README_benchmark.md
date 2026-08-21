# UniVTAC 仿真训练说明

本文档说明如何把 UniVTAC 仿真 HDF5 转成 LeRobot 数据集，并完成训练和远程 serve。

相关入口：

- 数据转换脚本：`scripts/data_convert/convert_turn_gear_pair_gelsight_sim_abs_joint.sh`
- norm stats 脚本：`scripts/data_convert/compute_norm_stats.sh`
- 训练脚本：`scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh`
- serve 脚本：`scripts/serve/serve_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh`

脚本按用途分在三个目录：

- `scripts/data_convert/`
- `scripts/train/`
- `scripts/serve/`

## 路径约定

| 内容 | 默认路径 |
| --- | --- |
| 仓库根目录 | `/home/tione/notebook/openpi_benchmark` |
| UniVTAC 原始 HDF5 | `/home/tione/notebook/data/UniVTAC/sim/<task>` |
| PI 训练数据根目录 | `/run/ti/pi-training-data/lerobot` |
| `turn_gear_pair` LeRobot 数据 | `/run/ti/pi-training-data/lerobot/simulation/turn_gear_pair_gelsight_abs_joint` |
| π0.5 基础权重 | `/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base` |
| norm stats | `/home/tione/notebook/openpi_benchmark/assets/<config_name>/<repo_id>` |
| 微调 checkpoint | `/home/tione/notebook/openpi_benchmark/checkpoints/<config_name>/<exp_name>` |
| 数据转换脚本 | `/home/tione/notebook/openpi_benchmark/scripts/data_convert/` |
| 训练脚本 | `/home/tione/notebook/openpi_benchmark/scripts/train/` |
| serve 脚本 | `/home/tione/notebook/openpi_benchmark/scripts/serve/` |

## 环境准备

```bash
cd /home/tione/notebook/openpi_benchmark
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

```bash
export PI_TRAINING_DATA_ROOT=/run/ti/pi-training-data
export HF_LEROBOT_HOME=${PI_TRAINING_DATA_ROOT}/lerobot
mkdir -p "${HF_LEROBOT_HOME}/simulation"
```

## 数据转换

```bash
bash scripts/data_convert/convert_turn_gear_pair_gelsight_sim_abs_joint.sh
```

常用覆盖：

```bash
HDF5_DIR=/home/tione/notebook/data/UniVTAC/sim/turn_gear_pair \
OUTPUT_DIR=/run/ti/pi-training-data/lerobot/simulation/turn_gear_pair_gelsight_abs_joint \
bash scripts/data_convert/convert_turn_gear_pair_gelsight_sim_abs_joint.sh
```

## norm stats

```bash
CONFIG_NAME=pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft \
bash scripts/data_convert/compute_norm_stats.sh
```

如果数据语义、control mode、state/action 维度或 frame stride 变了，要重新算 norm stats。

## 训练

```bash
unset CHECKPOINT_BASE_DIR
bash scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

常用参数：

```bash
GPU_IDS=0,1,2,3 \
FSDP_DEVICES=4 \
BATCH_SIZE=64 \
NUM_TRAIN_STEPS=20000 \
SAVE_INTERVAL=5000 \
KEEP_PERIOD=5000 \
EXP_NAME=turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_v1 \
bash scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

默认行为：

- `SKIP_CONVERT=1`
- `FORCE_NORM_STATS=0`
- `RESUME=0`

## 转换并训练

```bash
SKIP_CONVERT=0 \
FORCE_NORM_STATS=1 \
bash scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

## 恢复训练

```bash
RESUME=1 \
EXP_NAME=turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_v1 \
bash scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

## serve

```bash
bash scripts/serve/serve_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

默认读取：

```text
/home/tione/notebook/openpi_benchmark/checkpoints/pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft/turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_v1/19999
```

## 扩展到其他任务

新任务通常要做三件事：

1. 复制一个 `scripts/data_convert/` 下的转换脚本。
2. 复制一个 `scripts/train/` 下的训练脚本。
3. 在 `src/openpi/training/config.py` 里补一个对应的 `TrainConfig` 条目。

如果只是改 `exp_name`、batch size、`fsdp_devices` 这类运行参数，通常可以复用已有配置；如果数据语义、模型结构、权重来源或训练超参变了，建议新增一个 `TrainConfig`，别硬复用旧条目。
