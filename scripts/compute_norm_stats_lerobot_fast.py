#!/usr/bin/env python3
"""Fast LeRobot norm stats for configs whose stats only depend on state/actions.

The standard `compute_norm_stats.py` goes through the full LeRobot dataloader,
which materializes images and tactile clips before discarding them. This script
reads only the `state` and `actions` parquet columns and is intended for large
vision/tactile datasets where image features dominate IO.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def _repo_root_from_env(repo_id: str) -> Path:
    import os

    lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
    return lerobot_home / repo_id


def _list_episode_parquets(dataset_root: Path) -> list[Path]:
    paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_root / 'data'}")
    return paths


def _fixed_list_column_to_numpy(table, column: str) -> np.ndarray:
    values = table.column(column).combine_chunks().values.to_numpy(zero_copy_only=False)
    width = table.schema.field(column).type.list_size
    return values.reshape(-1, width).astype(np.float32, copy=False)


def _action_chunks(actions: np.ndarray, horizon: int) -> np.ndarray:
    timesteps = np.arange(actions.shape[0])[:, None] + np.arange(horizon)[None, :]
    timesteps = np.minimum(timesteps, actions.shape[0] - 1)
    return actions[timesteps]


def main(config_name: str, dataset_root: Path | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    root = dataset_root.expanduser().resolve() if dataset_root is not None else _repo_root_from_env(data_config.repo_id)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    horizon = int(config.model.action_horizon)
    delta_joint_actions = bool(getattr(config.data, "delta_joint_actions", False))
    output_action_dim = int(getattr(config.data, "output_action_dim", 0) or 0)

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    parquet_files = _list_episode_parquets(root)

    print(f"Config: {config_name}")
    print(f"Repo id: {data_config.repo_id}")
    print(f"Dataset root: {root}")
    print(f"Action horizon: {horizon}")
    print(f"Delta joint actions: {delta_joint_actions}")

    for parquet_path in tqdm.tqdm(parquet_files, desc="Reading state/actions parquet"):
        table = pq.read_table(parquet_path, columns=["state", "actions"])
        state = _fixed_list_column_to_numpy(table, "state")
        actions = _fixed_list_column_to_numpy(table, "actions")

        if output_action_dim:
            actions = actions[..., :output_action_dim]

        chunks = _action_chunks(actions, horizon)
        if delta_joint_actions:
            if state.shape[-1] < 8 or chunks.shape[-1] < 8:
                raise ValueError(
                    f"delta_joint_actions expects state/actions dim >= 8, got {state.shape[-1]}/{chunks.shape[-1]}"
                )
            chunks = chunks.copy()
            chunks[..., :7] -= state[:, None, :7]

        stats["state"].update(state)
        stats["actions"].update(chunks)

    norm_stats = {key: running.get_statistics() for key, running in stats.items()}
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
