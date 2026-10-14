import dataclasses
import os
import pathlib

import jax
import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


class _PartialCheckpointLoader:
    def load(self, params):
        del params
        return {
            "PaliGemma": {
                "img": {
                    "kernel": np.ones((2,), dtype=np.float32),
                }
            }
        }


def test_partial_checkpoint_allows_missing_tactile_encoder():
    params_shape = {
        "PaliGemma": {
            "img": {"kernel": jax.ShapeDtypeStruct((2,), np.float32)},
            "tactile_img": {"kernel": jax.ShapeDtypeStruct((2,), np.float32)},
        }
    }

    loaded = train._load_weights_and_validate(  # noqa: SLF001
        _PartialCheckpointLoader(),
        params_shape,
        allow_missing_regex=".*tactile_img.*",
    )

    np.testing.assert_array_equal(loaded["PaliGemma"]["img"]["kernel"], np.ones((2,), dtype=np.float32))
    assert "tactile_img" not in loaded["PaliGemma"]


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
