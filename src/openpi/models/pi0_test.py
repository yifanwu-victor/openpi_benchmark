import types

from flax import traverse_util
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_pi0_dual_siglip_has_tactile_encoder():
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        image_keys=("left_tactile_0_rgb", "right_tactile_0_rgb", "base_0_rgb", "wrist_0_rgb"),
        separate_tactile_encoder=True,
        tactile_image_keys=("left_tactile_0_rgb", "right_tactile_0_rgb"),
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    state = traverse_util.flatten_dict(nnx.state(abstract_model).to_pure_dict())

    assert any(path[:2] == ("PaliGemma", "img") for path in state)
    assert any(path[:2] == ("PaliGemma", "tactile_img") for path in state)
    assert tuple(config.inputs_spec()[0].images) == config.image_keys


def test_pi0_dual_siglip_dispatches_tactile_first():
    calls = []

    class RecordingEncoder:
        def __init__(self, name):
            self.name = name

        def __call__(self, image, *, train):
            calls.append((self.name, int(image[0, 0, 0, 0])))
            return jnp.zeros((image.shape[0], 2, 4), dtype=jnp.float32), None

    image_keys = ("left_tactile", "right_tactile", "base", "wrist")
    model = types.SimpleNamespace(
        image_keys=image_keys,
        tactile_image_keys=frozenset(image_keys[:2]),
        PaliGemma=types.SimpleNamespace(
            img=RecordingEncoder("vision"),
            tactile_img=RecordingEncoder("tactile"),
        ),
    )
    observation = _model.Observation(
        images={key: jnp.full((1, 2, 2, 3), value, dtype=jnp.float32) for value, key in enumerate(image_keys, 1)},
        image_masks={key: jnp.ones((1,), dtype=jnp.bool_) for key in image_keys},
        state=jnp.zeros((1, 8), dtype=jnp.float32),
    )

    tokens, input_mask, autoregressive_mask = _pi0.Pi0.embed_prefix(model, observation)

    assert calls == [("tactile", 1), ("tactile", 2), ("vision", 3), ("vision", 4)]
    assert tokens.shape == (1, 8, 4)
    assert input_mask.shape == (1, 8)
    assert autoregressive_mask.shape == (8,)


def test_pi0_initializes_tactile_encoder_from_loaded_vision_weights():
    class TinyEncoder(nnx.Module):
        def __init__(self, value):
            self.kernel = nnx.Param(jnp.array([value], dtype=jnp.float32))

    vision_encoder = TinyEncoder(3.0)
    tactile_encoder = TinyEncoder(8.0)
    model = types.SimpleNamespace(
        separate_tactile_encoder=True,
        PaliGemma=types.SimpleNamespace(img=vision_encoder, tactile_img=tactile_encoder),
    )

    _pi0.Pi0.initialize_tactile_encoder_from_vision(model)

    np.testing.assert_array_equal(tactile_encoder.kernel[...], vision_encoder.kernel[...])
