import numpy as np
import pytest

from openpi.models import model as model_lib
from openpi.policies import sim_policy


def test_sim_vision_inputs_maps_insert_block_base_and_wrist_images():
    base_image = np.full((2, 2, 3), 1, dtype=np.uint8)
    wrist_image = np.full((2, 2, 3), 2, dtype=np.uint8)
    transform = sim_policy.SimVisionInputs(model_type=model_lib.ModelType.PI05, state_dim=8)

    output = transform(
        {
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "observation/state": np.zeros(8, dtype=np.float32),
            "actions": np.ones((50, 8), dtype=np.float32),
            "prompt": "Insert the blue cube into the matching hole in the yellow box.",
        }
    )

    assert tuple(output["image"]) == sim_policy.VISION_IMAGE_KEYS
    assert np.array_equal(output["image"]["base_0_rgb"], base_image)
    assert np.array_equal(output["image"]["left_wrist_0_rgb"], wrist_image)
    assert np.array_equal(output["image"]["right_wrist_0_rgb"], np.zeros_like(base_image))
    assert output["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    assert output["prompt"] == "Insert the blue cube into the matching hole in the yellow box."
    assert output["actions"].shape == (50, 8)


def test_sim_vision_tactile_inputs_preserves_tactile_first_order():
    images = {
        "image": np.full((2, 2, 3), 1, dtype=np.uint8),
        "wrist_image": np.full((2, 2, 3), 2, dtype=np.uint8),
        "left_tactile_image": np.full((2, 2, 3), 3, dtype=np.uint8),
        "right_tactile_image": np.full((2, 2, 3), 4, dtype=np.uint8),
    }
    transform = sim_policy.SimVisionTactileInputs(
        model_type=model_lib.ModelType.PI05,
        state_dim=8,
        image_keys=sim_policy.TACTILE_FIRST_VISION_TACTILE_IMAGE_KEYS,
    )

    output = transform(
        {
            "observation/image": images["image"],
            "observation/wrist_image": images["wrist_image"],
            "observation/left_tactile_image": images["left_tactile_image"],
            "observation/right_tactile_image": images["right_tactile_image"],
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "pull the drawer open",
        }
    )

    assert tuple(output["image"]) == sim_policy.TACTILE_FIRST_VISION_TACTILE_IMAGE_KEYS
    assert np.array_equal(output["image"]["left_tactile_0_rgb"], images["left_tactile_image"])
    assert np.array_equal(output["image"]["right_tactile_0_rgb"], images["right_tactile_image"])
    assert np.array_equal(output["image"]["base_0_rgb"], images["image"])
    assert np.array_equal(output["image"]["wrist_0_rgb"], images["wrist_image"])


def test_sim_vision_tactile_inputs_rejects_incomplete_image_keys():
    transform = sim_policy.SimVisionTactileInputs(
        model_type=model_lib.ModelType.PI05,
        image_keys=("base_0_rgb", "wrist_0_rgb"),
    )

    with pytest.raises(ValueError, match="exactly the four"):
        transform(
            {
                "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
                "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "observation/left_tactile_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "observation/right_tactile_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "observation/state": np.zeros(8, dtype=np.float32),
                "prompt": "pull the drawer open",
            }
        )
