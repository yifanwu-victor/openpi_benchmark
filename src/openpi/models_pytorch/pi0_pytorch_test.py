import types

import pytest
import torch
from torch import nn

from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def test_tactile_expert_copy_has_equal_values_and_independent_storage():
    action_expert = nn.Linear(3, 2)
    tactile_expert = nn.Linear(3, 2)
    container = types.SimpleNamespace(
        use_tactile_expert=True,
        gemma_expert=action_expert,
        gemma_tactile_expert=tactile_expert,
    )

    PaliGemmaWithExpertModel.init_tactile_expert_from_action_expert(container)

    for action_parameter, tactile_parameter in zip(
        action_expert.parameters(), tactile_expert.parameters(), strict=True
    ):
        torch.testing.assert_close(action_parameter, tactile_parameter)
        assert action_parameter.data_ptr() != tactile_parameter.data_ptr()


def test_tactile_tokens_are_left_then_right_and_projected_to_expert_width():
    calls = []

    def encoder(clip):
        calls.append(int(clip[0, 0, 0, 0, 0]))
        return torch.full((clip.shape[0], 36, 1024), calls[-1], dtype=torch.float32)

    model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            use_tactile_expert=True,
            anytouch_keep_cls_sensor_tokens=True,
            anytouch_patch_tokens_per_side=30,
        ),
        anytouch2_encoder=encoder,
        _apply_checkpoint=lambda function, value: function(value),
    )
    left = torch.ones(2, 4, 1, 1, 1)
    right = torch.full((2, 4, 1, 1, 1), 2.0)

    tokens = PI0Pytorch.embed_tactile_tokens(model, [("left", left), ("right", right)])

    assert calls == [1, 2]
    assert tokens.shape == (2, 72, 1024)
    torch.testing.assert_close(tokens[:, :36], torch.ones(2, 36, 1024))
    torch.testing.assert_close(tokens[:, 36:], torch.full((2, 36, 1024), 2.0))


def test_tactile_time_is_limited_to_late_flow_interval():
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(flow_num_steps=10, action_expert_steps=6),
    )
    model.action_tactile_split_time = PI0Pytorch.action_tactile_split_time.__get__(model)

    samples = PI0Pytorch.sample_tactile_time(model, 4096, torch.device("cpu"))

    assert torch.all(samples > 0)
    assert torch.all(samples <= 0.4)


def test_cascaded_sampling_dispatches_four_tactile_steps_after_boundary():
    tactile_times = []
    config = types.SimpleNamespace(
        action_horizon=2,
        action_dim=1,
        flow_num_steps=10,
        action_expert_steps=6,
        use_tactile_expert=True,
    )
    model = types.SimpleNamespace(config=config)
    model._preprocess_observation = lambda observation, train: (  # noqa: SLF001
        [("left", None), ("right", None), ("base", None), ("wrist", None)],
        {},
        None,
        None,
        observation.state,
    )
    model._split_image_items = lambda images: (images[2:], images[:2])  # noqa: SLF001
    model.embed_prefix = lambda *args: (
        torch.zeros(1, 3, 4),
        torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    model._build_prefix_cache = lambda *args: object()  # noqa: SLF001
    model._action_partial_sample = lambda *args: (  # noqa: SLF001
        torch.zeros(1, 2, 1),
        (object(), torch.ones(1, 5, dtype=torch.bool)),
    )
    model.embed_tactile_tokens = lambda images: torch.zeros(1, 72, 4)

    def tactile_step(boundary_masks, boundary_cache, tokens, state, actions, timestep):
        tactile_times.append(float(timestep[0]))
        return torch.ones_like(actions)

    model._tactile_denoise_step = tactile_step  # noqa: SLF001
    observation = types.SimpleNamespace(state=torch.zeros(1, 8))

    actions = PI0Pytorch.sample_actions(
        model,
        torch.device("cpu"),
        observation,
        noise=torch.zeros(1, 2, 1),
    )

    assert tactile_times == pytest.approx([0.4, 0.3, 0.2, 0.1])
    torch.testing.assert_close(actions, torch.full((1, 2, 1), -0.4))


def test_action_only_fallback_runs_all_ten_steps():
    action_times = []
    config = types.SimpleNamespace(
        action_horizon=2,
        action_dim=1,
        flow_num_steps=10,
        action_expert_steps=6,
        use_tactile_expert=True,
    )
    model = types.SimpleNamespace(config=config)
    model._preprocess_observation = lambda observation, train: (  # noqa: SLF001
        [("left", None), ("right", None), ("base", None), ("wrist", None)],
        {},
        None,
        None,
        observation.state,
    )
    model._split_image_items = lambda images: (images[2:], images[:2])  # noqa: SLF001
    model.embed_prefix = lambda *args: (
        torch.zeros(1, 3, 4),
        torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    model._build_prefix_cache = lambda *args: object()  # noqa: SLF001

    def action_step(state, prefix_masks, prefix_cache, actions, timestep):
        action_times.append(float(timestep[0]))
        return torch.ones_like(actions)

    model.denoise_step = action_step
    observation = types.SimpleNamespace(state=torch.zeros(1, 8))

    actions = PI0Pytorch.sample_actions(
        model,
        torch.device("cpu"),
        observation,
        noise=torch.zeros(1, 2, 1),
        disable_tactile_expert=True,
    )

    assert action_times == pytest.approx([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    torch.testing.assert_close(actions, torch.full((1, 2, 1), -1.0))
