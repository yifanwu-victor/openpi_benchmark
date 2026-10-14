import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import msgpack_numpy
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.policy import TactileClipPolicy
from openpi.training import config as _config


class _RecordingPolicy:
    def __init__(self):
        self.calls = []

    def infer(self, obs, *, noise=None):
        del noise
        self.calls.append(obs)
        return {"actions": np.zeros((1, 8), dtype=np.float32)}

    def create_session(self):
        return self

    @property
    def metadata(self):
        return {"test": True}

    def reset(self):
        self.calls.clear()


def _valid_tactile_payload():
    return {
        "observation/left_tactile_clip": np.zeros((4, 224, 224, 3), dtype=np.uint8),
        "observation/right_tactile_clip": np.ones((4, 224, 224, 3), dtype=np.uint8),
    }


def test_tactile_clip_policy_accepts_msgpack_payload_and_strips_reset():
    backing = _RecordingPolicy()
    policy = TactileClipPolicy(backing)
    payload = {
        **_valid_tactile_payload(),
        "reset": True,
        "episode_reset": np.array([True]),
    }

    decoded_payload = msgpack_numpy.unpackb(msgpack_numpy.packb(payload))
    policy.infer(decoded_payload)

    inputs = backing.calls[-1]
    assert "reset" not in inputs
    assert "episode_reset" not in inputs
    assert inputs["observation/left_tactile_clip"].dtype == np.uint8
    assert inputs["observation/left_tactile_clip"].shape == (4, 224, 224, 3)
    np.testing.assert_array_equal(
        inputs["observation/right_tactile_clip"], payload["observation/right_tactile_clip"]
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "observation/left_tactile_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/right_tactile_image": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        {"observation/left_tactile_clip": np.zeros((4, 224, 224, 3), dtype=np.uint8)},
    ],
)
def test_tactile_clip_policy_requires_both_clips(payload):
    policy = TactileClipPolicy(_RecordingPolicy())

    with pytest.raises(ValueError, match="direct tactile clips only|Missing required AnyTouch tactile clip"):
        policy.infer(payload)


def test_tactile_clip_policy_rejects_single_frames_alongside_clips():
    payload = _valid_tactile_payload()
    payload["observation/left_tactile_image"] = np.zeros((224, 224, 3), dtype=np.uint8)
    policy = TactileClipPolicy(_RecordingPolicy())

    with pytest.raises(ValueError, match="direct tactile clips only"):
        policy.infer(payload)


@pytest.mark.parametrize(
    ("invalid_clip", "error_type", "error_match"),
    [
        (np.zeros((4, 224, 224, 3), dtype=np.float32), TypeError, "dtype uint8"),
        (np.zeros((3, 224, 224, 3), dtype=np.uint8), ValueError, "must have shape"),
        (np.zeros((4, 3, 224, 224), dtype=np.uint8), ValueError, "must have shape"),
        (np.zeros((4, 224, 224, 3), dtype=np.uint8)[:, :, ::-1, :], ValueError, "C-contiguous"),
    ],
)
def test_tactile_clip_policy_rejects_invalid_clip(invalid_clip, error_type, error_match):
    payload = _valid_tactile_payload()
    payload["observation/left_tactile_clip"] = invalid_clip
    policy = TactileClipPolicy(_RecordingPolicy())

    with pytest.raises(error_type, match=error_match):
        policy.infer(payload)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
