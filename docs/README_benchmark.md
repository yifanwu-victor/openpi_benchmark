# OpenPI Benchmark 训练与 Serve 操作手册

本文档说明如何在本仓库中完成 UniVTAC 仿真数据的处理、norm stats 计算、训练、checkpoint serve，以及如何新增一个任务。

仓库根目录：

```bash
cd /home/tione/notebook/openpi_benchmark
```

## 0. 最短流程

以 `turn_gear_pair` 的 `abs_joint + gelsight + pi0.5 full-ft` 配置为例，完整流程是：

```text
UniVTAC HDF5
  -> LeRobot 数据集
  -> norm stats
  -> pi0.5 full fine-tuning
  -> checkpoint serve
  -> 仿真/真实端连接 server 做评估
```

最短命令：

```bash
cd /home/tione/notebook/openpi_benchmark

export PI_TRAINING_DATA_ROOT=/run/ti/pi-training-data
export HF_LEROBOT_HOME=${PI_TRAINING_DATA_ROOT}/lerobot

bash scripts/data_convert/convert_turn_gear_pair_gelsight_sim_abs_joint.sh

CONFIG_NAME=pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft \
bash scripts/data_convert/compute_norm_stats.sh

bash scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

训练完成或已有 checkpoint 后启动 serve：

```bash
bash scripts/serve/serve_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

100 条数据调试配置：

```bash
bash scripts/data_convert/convert_100_turn_gear_pair.sh

CONFIG_NAME=pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_100 \
bash scripts/data_convert/compute_norm_stats.sh

bash scripts/train/train_turn_gear_pair_100.sh

CHECKPOINT_STEP=5000 bash scripts/serve/serve_turn_gear_pair_100.sh
```

## 1. 关键路径和命名

| 内容 | 默认路径或名称 |
| --- | --- |
| 仓库根目录 | `/home/tione/notebook/openpi_benchmark` |
| UniVTAC 原始 HDF5 根目录 | `/home/tione/notebook/data/UniVTAC/sim/<task>` |
| 临时训练数据根目录 | `/run/ti/pi-training-data` |
| LeRobot 根目录 | `/run/ti/pi-training-data/lerobot` |
| `turn_gear_pair` LeRobot 数据 | `/run/ti/pi-training-data/lerobot/simulation/turn_gear_pair_gelsight_abs_joint` |
| `turn_gear_pair_100` LeRobot 数据 | `/run/ti/pi-training-data/lerobot/simulation/turn_gear_pair_gelsight_abs_joint_100` |
| pi0.5 基础权重 | `/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base/params` |
| norm stats | `/home/tione/notebook/openpi_benchmark/assets/<CONFIG_NAME>/<REPO_ID>` |
| checkpoint | `/home/tione/notebook/openpi_benchmark/checkpoints/<CONFIG_NAME>/<EXP_NAME>/<STEP>` |

`HF_LEROBOT_HOME` 和 `REPO_ID` 必须拼成真实数据目录：

```text
${HF_LEROBOT_HOME}/${REPO_ID}
```

例如：

```text
HF_LEROBOT_HOME=/run/ti/pi-training-data/lerobot
REPO_ID=simulation/turn_gear_pair_gelsight_abs_joint

实际目录=/run/ti/pi-training-data/lerobot/simulation/turn_gear_pair_gelsight_abs_joint
```

`/run/ti` 是实例本地临时盘，适合放训练时频繁读取的 LeRobot 数据，但不适合作为唯一长期副本。原始 HDF5、代码、基础权重、norm stats 和 checkpoint 建议保留在数据盘。

## 2. 环境准备

进入仓库：

```bash
cd /home/tione/notebook/openpi_benchmark
```

安装或更新依赖：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

确认 GPU 和基础权重：

```bash
nvidia-smi
test -d /home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base/params
```

设置 LeRobot 数据根目录：

```bash
export PI_TRAINING_DATA_ROOT=/run/ti/pi-training-data
export HF_LEROBOT_HOME=${PI_TRAINING_DATA_ROOT}/lerobot
mkdir -p "${HF_LEROBOT_HOME}/simulation"
```

### 2.1 ptxas 版本

当前 GPU 是 `sm_120`，JAX 需要 CUDA 12.6.3 以上的 `ptxas`。系统默认 `/usr/local/cuda/bin/ptxas` 是 CUDA 12.4，不能放在优先级最高的位置。

训练脚本和新增的 100 条数据 serve 脚本已经内置以下逻辑：

```bash
source ~/.bashrc
unset VIRTUAL_ENV

export TRITON_PTXAS_DIR=/home/tione/notebook/openpi_benchmark/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin
export PATH="${TRITON_PTXAS_DIR}:$PATH"

which ptxas
ptxas --version
```

期望输出：

```text
/home/tione/notebook/openpi_benchmark/.venv/lib/python3.11/site-packages/triton/backends/nvidia/bin/ptxas
Cuda compilation tools, release 12.8, V12.8.93
```

如果日志里出现 `ptxas too old`，先检查 `which ptxas` 是否仍然命中了 `/usr/local/cuda/bin/ptxas`。

## 3. 数据要求

转换脚本递归查找 `HDF5_DIR` 下的 `.hdf5` 文件。`HDF5_DIR` 可以是直接包含 HDF5 文件的目录，也可以是任务根目录。

仿真 HDF5 至少需要包含：

```text
step
embodiment/joint
observation/head/rgb
observation/wrist/rgb
tactile/left_tactile/rgb_marker
tactile/right_tactile/rgb_marker
```

如果使用 `delta_eef` 控制模式，还需要：

```text
embodiment/ee
```

每个 HDF5 应在根属性中提供 `instruction`，或者在 `episode_context_json` 中提供 `instruction`。也可以用 `TASK_PROMPT` 为所有 episode 强制指定统一指令。

当前转换器输出四路图像：

```text
image                <- observation/head/rgb
wrist_image          <- observation/wrist/rgb
left_tactile_image   <- tactile/left_tactile/rgb_marker
right_tactile_image  <- tactile/right_tactile/rgb_marker
```

## 4. 数据处理：HDF5 转 LeRobot

### 4.1 标准 turn_gear_pair 数据

```bash
cd /home/tione/notebook/openpi_benchmark

HDF5_DIR=/home/tione/notebook/data/UniVTAC/sim/turn_gear_pair \
REPO_ID=simulation/turn_gear_pair_gelsight_abs_joint \
OUTPUT_DIR=/run/ti/pi-training-data/lerobot/simulation/turn_gear_pair_gelsight_abs_joint \
bash scripts/data_convert/convert_turn_gear_pair_gelsight_sim_abs_joint.sh
```

### 4.2 100 条数据调试集

```bash
cd /home/tione/notebook/openpi_benchmark

HDF5_DIR=/home/tione/notebook/data/UniVTAC/sim/turn_gear_pair \
REPO_ID=simulation/turn_gear_pair_gelsight_abs_joint_100 \
OUTPUT_DIR=/run/ti/pi-training-data/lerobot/simulation/turn_gear_pair_gelsight_abs_joint_100 \
bash scripts/data_convert/convert_100_turn_gear_pair.sh
```

### 4.3 常用环境变量

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `HDF5_DIR` | 输入 HDF5 目录 | `/home/tione/notebook/data/UniVTAC/sim/turn_gear_pair` |
| `PI_TRAINING_DATA_ROOT` | 训练数据根目录 | `/run/ti/pi-training-data` |
| `HF_LEROBOT_HOME` | LeRobot 根目录 | `${PI_TRAINING_DATA_ROOT}/lerobot` |
| `REPO_ID` | LeRobot 数据集相对路径和元数据 id | `simulation/turn_gear_pair_gelsight_abs_joint` |
| `OUTPUT_DIR` | 实际写入的数据集目录 | `${HF_LEROBOT_HOME}/${REPO_ID}` |
| `CONTROL_MODE` | state/action 表示方式 | `abs_joint` |
| `GRIPPER_MODE` | 夹爪标量来源 | `left` |
| `OVERWRITE` | 是否删除并重建 `OUTPUT_DIR` | `1` |
| `DRY_RUN` | 只打印计划，不写数据 | `0` |
| `MAX_EPISODES` | 只转换前 N 个 episode | 空 |
| `EPISODE_IDS` | 只转换指定 episode | 空 |
| `FRAME_STRIDE` | 每 N 帧保留一帧 | 空，转换器默认 1 |
| `TARGET_OFFSET` | action 目标相对当前帧的原始帧偏移 | 空，转换器默认 1 |
| `TASK_PROMPT` | 覆盖所有 episode 的 prompt | 空 |
| `IMAGE_WRITER_PROCESSES` | LeRobot 图像写进程数 | 空，转换器默认 5 |
| `IMAGE_WRITER_THREADS` | 每个图像写进程的线程数 | 空，转换器默认 10 |

`OVERWRITE=1` 会删除已有 `OUTPUT_DIR` 后重新生成。确认路径无误后再执行，尤其不要把 `OUTPUT_DIR` 指到上层目录。

### 4.4 控制模式语义

| `CONTROL_MODE` | state | action |
| --- | --- | --- |
| `abs_joint` | `[joint7, gripper]` | `[target_joint7, target_gripper]` |
| `delta_joint` | `[joint7, gripper]` | `[delta_joint7, target_gripper]` |
| `delta_eef` | `[xyz, quat_xyzw, gripper]` | `[dxyz, drotvec, target_gripper]` |

当前 `turn_gear_pair` 配置使用 `abs_joint` 数据。训练配置里启用了 `delta_joint_actions=True`，训练前会把 joint action 转为：

```text
模型 actions[:7] = target_joint[:7] - observation/state[:7]
模型 actions[7]  = target_gripper
```

serve 端会按对应 transform 把模型输出转回 absolute joint target。不要把 `delta_eef` 数据直接交给这个 `abs_joint + delta_joint_actions=True` 的训练配置。

### 4.5 检查转换结果

```bash
DATASET_DIR=/run/ti/pi-training-data/lerobot/simulation/turn_gear_pair_gelsight_abs_joint

test -d "${DATASET_DIR}"
du -sh "${DATASET_DIR}"
find "${DATASET_DIR}" -type f | wc -l
find "${DATASET_DIR}" -maxdepth 2 -type f | sort | head -n 30
```

典型 LeRobot 数据集应包含 metadata、episode 表格以及视频或图像数据。不要只确认目录存在，还要确认文件数量和总大小合理。

## 5. 计算 norm stats

训练脚本会在 norm stats 不存在时自动计算。建议首次跑新数据时手动计算一次，便于把问题定位在数据阶段。

标准配置：

```bash
cd /home/tione/notebook/openpi_benchmark
export HF_LEROBOT_HOME=/run/ti/pi-training-data/lerobot

CONFIG_NAME=pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft \
bash scripts/data_convert/compute_norm_stats.sh
```

100 条数据配置：

```bash
cd /home/tione/notebook/openpi_benchmark
export HF_LEROBOT_HOME=/run/ti/pi-training-data/lerobot

CONFIG_NAME=pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_100 \
bash scripts/data_convert/compute_norm_stats.sh
```

输出目录：

```text
assets/<CONFIG_NAME>/<REPO_ID>
```

例如：

```text
assets/pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft/simulation/turn_gear_pair_gelsight_abs_joint
```

可选参数：

| 变量 | 作用 |
| --- | --- |
| `CONFIG_NAME` | 选择 `src/openpi/training/config.py` 里的训练配置 |
| `MAX_FRAMES` | 只用前 N 帧计算 stats，适合快速检查 |

以下情况必须重新计算 norm stats：

- 换了 `REPO_ID` 或数据目录。
- 改了 `CONTROL_MODE`、`GRIPPER_MODE`、state/action 维度。
- 改了 `FRAME_STRIDE`、`TARGET_OFFSET` 或 action 语义。
- 换了数据版本，或者重新转换覆盖了原数据。

强制重新计算：

```bash
FORCE_NORM_STATS=1 \
bash scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

## 6. 训练

### 6.1 标准四卡训练

```bash
cd /home/tione/notebook/openpi_benchmark
unset CHECKPOINT_BASE_DIR

bash scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

显式指定常用参数：

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

### 6.2 100 条数据训练

```bash
cd /home/tione/notebook/openpi_benchmark

bash scripts/train/train_turn_gear_pair_100.sh
```

默认输出：

```text
checkpoints/pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_100/
└── turn_gear_pair_gelsight_abs_joint_100_v1/
    ├── 5000/
    └── 9999/
```

### 6.3 训练脚本参数

| 变量 | 作用 | 标准默认值 | 100 条默认值 |
| --- | --- | --- | --- |
| `CONFIG_NAME` | 训练配置名 | `pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft` | `pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_100` |
| `EXP_NAME` | 实验名，也是 checkpoint 二级目录 | `turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_v1` | `turn_gear_pair_gelsight_abs_joint_100_v1` |
| `GPU_IDS` | 可见 GPU | `0,1,2,3` | `0,1,2,3` |
| `FSDP_DEVICES` | FSDP 使用 GPU 数量 | `4` | `4` |
| `BATCH_SIZE` | batch size | `64` | `64` |
| `NUM_TRAIN_STEPS` | 训练步数 | `20000` | `10000` |
| `SAVE_INTERVAL` | 保存间隔 | `5000` | `5000` |
| `KEEP_PERIOD` | checkpoint 保留间隔 | `5000` | `5000` |
| `SKIP_CONVERT` | 是否跳过数据转换 | `1` | `1` |
| `FORCE_NORM_STATS` | 是否强制重算 stats | `0` | `0` |
| `RESUME` | 是否恢复训练 | `0` | `0` |
| `DRY_RUN` | 只打印检查信息，不启动训练 | `0` | `0` |
| `CHECKPOINT_BASE_DIR` | checkpoint 根目录 | `${REPO_ROOT}/checkpoints` | `${REPO_ROOT}/checkpoints` |

pi0.5 full fine-tuning 单卡通常需要超过 70 GB 显存。`FSDP_DEVICES` 应与 `GPU_IDS` 里 GPU 数量一致。

### 6.4 转换并立即训练

如果临时盘没有 LeRobot 数据，可以让训练脚本先转换再训练：

```bash
SKIP_CONVERT=0 \
FORCE_NORM_STATS=1 \
bash scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

注意：`SKIP_CONVERT=0` 会让训练脚本以 `OVERWRITE=1` 调用转换脚本，默认会删除并重建 `OUTPUT_DIR`。

### 6.5 恢复训练

使用同一个 `CONFIG_NAME`、`EXP_NAME` 和 `CHECKPOINT_BASE_DIR`：

```bash
RESUME=1 \
EXP_NAME=turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_v1 \
bash scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

非恢复模式会向训练程序传 `--overwrite`。重复使用同一个 `EXP_NAME` 启动新训练前，先确认是否允许覆盖旧实验。

## 7. 启动 serve

OpenPI serve 只负责 `observation -> policy inference -> action_chunk`，不启动 IsaacLab、UniVTAC 或 ViTaForge 环境。

### 7.1 标准 checkpoint serve

```bash
cd /home/tione/notebook/openpi_benchmark

CHECKPOINT_STEP=19999 \
PORT=8000 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/serve/serve_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

默认 checkpoint：

```text
checkpoints/pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft/
└── turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_v1/
    └── 19999/
```

### 7.2 100 条数据 checkpoint serve

```bash
cd /home/tione/notebook/openpi_benchmark

CHECKPOINT_STEP=9999 \
PORT=8000 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/serve/serve_turn_gear_pair_100.sh
```

如果训练还没结束，可以先 serve 已有 checkpoint：

```bash
CHECKPOINT_STEP=5000 bash scripts/serve/serve_turn_gear_pair_100.sh
```

默认 checkpoint：

```text
checkpoints/pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft_100/
└── turn_gear_pair_gelsight_abs_joint_100_v1/
    └── 9999/
```

### 7.3 serve 参数

| 变量 | 作用 |
| --- | --- |
| `CONFIG_NAME` | serve 使用的训练配置名，必须和 checkpoint 对应 |
| `EXP_NAME` | 实验名，参与拼接默认 checkpoint 路径 |
| `CHECKPOINT_STEP` | checkpoint step |
| `CHECKPOINT_DIR` | 直接指定 checkpoint 目录；设置后优先级最高 |
| `CHECKPOINT_BASE_DIR` | checkpoint 根目录 |
| `PORT` | server 端口 |
| `RECORD` | 是否开启 `scripts/serve_policy.py --record` |
| `CUDA_VISIBLE_DEVICES` | serve 使用的 GPU |
| `OPENPI_DATA_HOME` | OpenPI 数据目录 |
| `HF_HOME` | Hugging Face cache |
| `HF_DATASETS_CACHE` | Hugging Face datasets cache |
| `HF_LEROBOT_HOME` | LeRobot 数据目录 |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | 是否让 JAX 预占显存，默认 `false` |
| `TRITON_PTXAS_DIR` | CUDA 12.8 `ptxas` 所在目录 |

如果 checkpoint 不在默认目录，直接传绝对路径：

```bash
CHECKPOINT_DIR=/home/tione/notebook/openpi_benchmark/checkpoints/<CONFIG_NAME>/<EXP_NAME>/<STEP> \
bash scripts/serve/serve_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

## 8. 仿真部署评估

远端 serve 启动后，在 ViTaForge 机器建立端口转发：

```bash
ssh -N -L 8000:127.0.0.1:8000 -p 10383 root@42.192.34.154
```

如果远端 serve 使用了其他端口，本地 tunnel 和 deploy yaml 中的 `openpi.port` 必须同步修改。

在 ViTaForge 仓库执行：

```bash
cd /data/home/liqin/VLA/ViTaForge
conda activate UniVTAC

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  turn_gear_pair \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_turn_gear_pair_pi05_relative_action_vision.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100
```

当前 deploy 配置的关键语义：

```text
control_mode=abs_joint
send_tactile=false
open_loop_horizon=1
temporal_ensemble=true
chunk_first_n=20
ensemble_K=0.01
```

pi0.5 server 每次推理返回 `action_horizon=50` 的 action chunk。本地 ViTaForge client 只使用前 20 步做 temporal ensemble，并且每个 env step 执行 1 个 absolute qpos action。

## 9. 新建任务怎么做

新增任务不要只复制命令改任务名。需要保证四件事一致：

```text
HDF5 数据语义
  == 转换脚本 CONTROL_MODE / REPO_ID
  == config.py 里的 DataConfig / state_dim / action_dim / transform
  == norm stats
  == train 和 serve 的 CONFIG_NAME / checkpoint
```

### 9.1 选择任务命名

先确定以下变量：

```bash
TASK=<task_name>
CONTROL_MODE=abs_joint
REPO_ID=simulation/${TASK}_gelsight_abs_joint
CONFIG_NAME=pi05_${TASK}_gelsight_sim_abs_joint_relative_action_vision_full_ft
EXP_NAME=${TASK}_gelsight_sim_abs_joint_relative_action_vision_full_ft_v1
HDF5_DIR=/home/tione/notebook/data/UniVTAC/sim/${TASK}
OUTPUT_DIR=/run/ti/pi-training-data/lerobot/${REPO_ID}
```

如果不是 `abs_joint`，把 `CONTROL_MODE`、`REPO_ID`、`CONFIG_NAME` 和 deploy 配置里的控制模式一起改掉。

### 9.2 新建数据转换脚本

从最接近的脚本复制：

```bash
cp scripts/data_convert/convert_turn_gear_pair_gelsight_sim_abs_joint.sh \
  scripts/data_convert/convert_${TASK}_gelsight_sim_abs_joint.sh
```

至少修改：

```text
HDF5_DIR
REPO_ID
OUTPUT_DIR
CONTROL_MODE
GRIPPER_MODE
TASK_PROMPT，如 HDF5 中没有 instruction
```

先 dry run：

```bash
DRY_RUN=1 bash scripts/data_convert/convert_${TASK}_gelsight_sim_abs_joint.sh
```

再正式转换：

```bash
bash scripts/data_convert/convert_${TASK}_gelsight_sim_abs_joint.sh
```

### 9.3 注册训练配置

在 `src/openpi/training/config.py` 的 `_CONFIGS` 中新增一个 `TrainConfig`。通常可以从 `pi05_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft` 复制。

必须检查：

| 字段 | 说明 |
| --- | --- |
| `name` | 必须等于训练脚本里的 `CONFIG_NAME` |
| `data.repo_id` | 必须等于转换脚本里的 `REPO_ID` |
| `data.state_dim` | 必须匹配转换后的 state 维度 |
| `data.output_action_dim` | 必须匹配转换后的 action 维度 |
| `data.delta_joint_actions` | 只适用于 `abs_joint` 数据转 relative joint action 的配置 |
| `weight_loader` | 建议指向本地 pi0.5 基础权重 |
| `batch_size`、`num_train_steps`、`fsdp_devices` | 可按资源调整 |

`abs_joint + relative_action` 示例：

```python
TrainConfig(
    name="pi05_<task>_gelsight_sim_abs_joint_relative_action_vision_full_ft",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=50,
        discrete_state_input=False,
    ),
    data=LeRobotSimVisionDataConfig(
        repo_id="simulation/<task>_gelsight_abs_joint",
        base_config=DataConfig(prompt_from_task=True),
        state_dim=sim_policy.ABS_JOINT_STATE_DIM,
        output_action_dim=sim_policy.ABS_JOINT_ACTION_DIM,
        delta_joint_actions=True,
    ),
    batch_size=64,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=20_000,
    save_interval=5_000,
    keep_period=5_000,
    fsdp_devices=4,
)
```

如果新任务使用 `delta_eef`，不能复用这段配置，需要使用对应的 state/action 维度和 action transform。

### 9.4 新建训练脚本

```bash
cp scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh \
  scripts/train/train_${TASK}_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

至少修改：

```text
HDF5_DIR
REPO_ID
OUTPUT_DIR
CONFIG_NAME
EXP_NAME
NUM_TRAIN_STEPS
```

建议先检查：

```bash
DRY_RUN=1 bash scripts/train/train_${TASK}_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

再训练：

```bash
bash scripts/train/train_${TASK}_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

### 9.5 计算新任务 norm stats

```bash
export HF_LEROBOT_HOME=/run/ti/pi-training-data/lerobot

CONFIG_NAME=pi05_${TASK}_gelsight_sim_abs_joint_relative_action_vision_full_ft \
bash scripts/data_convert/compute_norm_stats.sh
```

确认输出存在：

```bash
test -d assets/pi05_${TASK}_gelsight_sim_abs_joint_relative_action_vision_full_ft/${REPO_ID}
```

### 9.6 新建 serve 脚本

```bash
cp scripts/serve/serve_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh \
  scripts/serve/serve_${TASK}_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

至少修改：

```text
CONFIG_NAME
EXP_NAME
CHECKPOINT_STEP
CHECKPOINT_DIR 默认拼接逻辑
PORT，如需要
CUDA_VISIBLE_DEVICES，如需要
```

启动：

```bash
CHECKPOINT_STEP=<step> bash scripts/serve/serve_${TASK}_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

### 9.7 新建任务检查清单

执行训练前逐项确认：

```text
[ ] HDF5_DIR 下能找到 .hdf5
[ ] HDF5 中有 instruction，或转换脚本设置了 TASK_PROMPT
[ ] OUTPUT_DIR = ${HF_LEROBOT_HOME}/${REPO_ID}
[ ] CONTROL_MODE 与 config.py 的 state/action 维度一致
[ ] config.py 注册了 CONFIG_NAME
[ ] config.py 的 data.repo_id 等于转换脚本 REPO_ID
[ ] norm stats 已计算，目录为 assets/<CONFIG_NAME>/<REPO_ID>
[ ] 训练脚本 EXP_NAME 是新的，不会覆盖旧实验
[ ] FSDP_DEVICES 等于 GPU_IDS 中的 GPU 数量
[ ] serve 脚本 CONFIG_NAME/EXP_NAME/CHECKPOINT_STEP 指向真实 checkpoint
[ ] deploy/eval 端 control_mode 与模型输出语义一致
```

## 10. 常见问题

### 找不到 LeRobot 数据

检查：

```bash
echo "${HF_LEROBOT_HOME}"
test -d "${HF_LEROBOT_HOME}/simulation/turn_gear_pair_gelsight_abs_joint"
```

最常见原因是 `HF_LEROBOT_HOME`、`REPO_ID`、`OUTPUT_DIR` 不在同一个目录树下。

### norm stats 与数据不匹配

只要数据版本或 action 语义改变，就重新计算：

```bash
FORCE_NORM_STATS=1 bash scripts/train/train_turn_gear_pair_gelsight_sim_abs_joint_relative_action_vision_full_ft.sh
```

不要复用其他任务、其他控制模式或旧数据版本的 stats。

### GPU OOM

优先减小 `BATCH_SIZE`，并确认 `FSDP_DEVICES` 与可见 GPU 数量一致。不要把同一组 GPU 同时分配给多个训练或 serve 进程。

### 训练覆盖旧实验

非恢复模式默认 `--overwrite`。新实验应使用新的 `EXP_NAME`；恢复已有实验才使用：

```bash
RESUME=1 EXP_NAME=<old_exp_name> bash scripts/train/<train_script>.sh
```

### checkpoint 目录找不到

默认 checkpoint 路径是：

```text
checkpoints/<CONFIG_NAME>/<EXP_NAME>/<STEP>
```

如果你手动传的是 `checkpoints/<CONFIG_NAME>/<STEP>`，多数训练脚本找不到，因为缺少 `EXP_NAME` 这一层。必要时用 `CHECKPOINT_DIR` 传完整绝对路径。

### serve 端口连不上

确认三处一致：

```text
serve 脚本 PORT
ssh -L 本地端口:127.0.0.1:远端端口
deploy yaml 中的 openpi.port
```

### 临时盘数据丢失

`/run/ti` 中的 LeRobot 数据可以从原始 HDF5 重新转换。不要把 `/run/ti` 当作唯一长期副本。
