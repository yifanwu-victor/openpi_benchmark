import torch
from torch import nn

from openpi.models_pytorch.anytouch2 import encoder as encoder_module


class _FakeTactileEncoder(nn.Module):
    def forward(self, clip, sensor_type):
        del sensor_type
        return torch.zeros(clip.shape[0], 36, 512)


def test_random_init_does_not_load_anytouch_checkpoint(monkeypatch):
    loaded_paths = []
    monkeypatch.setattr(
        encoder_module.TactileVideoMAE,
        "from_config_path",
        lambda config_path, num_frames, stride: _FakeTactileEncoder(),
    )
    monkeypatch.setattr(
        encoder_module,
        "load_anytouch2_checkpoint",
        lambda checkpoint_path, model: loaded_paths.append(checkpoint_path),
    )

    encoder = encoder_module.AnyTouch2TactileEncoder(
        checkpoint_path=None,
        random_init=True,
        output_dim=16,
        dtype=torch.float32,
    )

    assert loaded_paths == []
    assert encoder.projector.in_features == 512
    assert encoder.projector.out_features == 16
