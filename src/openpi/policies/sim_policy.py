import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

DELTA_EEF_STATE_DIM = 8
DELTA_EEF_ACTION_DIM = 7
DELTA_JOINT_STATE_DIM = 8
DELTA_JOINT_ACTION_DIM = 8
ABS_EEF_STATE_DIM = 10
ABS_EEF_ACTION_DIM = 10
ABS_JOINT_STATE_DIM = 8
ABS_JOINT_ACTION_DIM = 8

VISION_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
VISION_TACTILE_IMAGE_KEYS = ("base_0_rgb", "wrist_0_rgb", "left_tactile_0_rgb", "right_tactile_0_rgb")


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_prompt(data: dict) -> str:
    if "prompt" not in data:
        raise ValueError("Sim policy requires a top-level 'prompt' field in every observation.")

    prompt = data["prompt"]
    if isinstance(prompt, np.ndarray):
        if prompt.size != 1:
            raise ValueError(f"Expected scalar prompt, got array with shape {prompt.shape}.")
        prompt = prompt.item()

    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")

    if not isinstance(prompt, str):
        raise TypeError(f"Expected prompt to be str or bytes, got {type(prompt).__name__}.")
    if not prompt.strip():
        raise ValueError("Sim policy requires a non-empty prompt.")

    return prompt


def _parse_state(state, expected_dim: int | None) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if expected_dim is not None and state.shape[-1] != expected_dim:
        raise ValueError(f"Expected state dim {expected_dim}, got shape {state.shape}.")
    return state


@dataclasses.dataclass(frozen=True)
class SimVisionInputs(transforms.DataTransformFn):
    """Map UniVTAC sim vision observations to pi0/pi0.5 input names."""

    model_type: _model.ModelType
    state_dim: int | None = None

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = VISION_IMAGE_KEYS
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _:
                raise ValueError(f"Unsupported model type for sim vision policy: {self.model_type}")

        inputs = {
            "state": _parse_state(data["observation/state"], self.state_dim),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "prompt": _parse_prompt(data),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        return inputs


@dataclasses.dataclass(frozen=True)
class SimVisionTactileInputs(transforms.DataTransformFn):
    """Map UniVTAC sim visual and tactile observations to pi0/pi0.5 input names."""

    model_type: _model.ModelType
    state_dim: int | None = None

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        left_tactile_image = _parse_image(data["observation/left_tactile_image"])
        right_tactile_image = _parse_image(data["observation/right_tactile_image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = VISION_TACTILE_IMAGE_KEYS
                images = (base_image, wrist_image, left_tactile_image, right_tactile_image)
                image_masks = (np.True_, np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type for sim vision+tactile policy: {self.model_type}")

        inputs = {
            "state": _parse_state(data["observation/state"], self.state_dim),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "prompt": _parse_prompt(data),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        return inputs


@dataclasses.dataclass(frozen=True)
class SimOutputs(transforms.DataTransformFn):
    """Return only the task action dimensions after model padding."""

    action_dim: int = DELTA_EEF_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
