import flax.nnx as nnx
import numpy as np
import pytest

import openpi.policies.sim_policy as sim_policy
import openpi.training.config as config
import openpi.training.optimizer as optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as transforms


def test_anytouch_serving_config_disables_torch_compile_and_cuda_graphs():
    train_config = config.get_config(
        "pi05_insert_usb_v3_gelsight_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft"
    )

    assert train_config.model.tactile_encoder_type == "anytouch2"
    assert train_config.model.image_keys == sim_policy.TACTILE_FIRST_VISION_TACTILE_IMAGE_KEYS
    assert train_config.model.anytouch_num_frames == 4
    assert train_config.model.pytorch_compile_mode is None


@pytest.mark.parametrize(
    ("task_name", "repo_id", "norm_stats_config_name"),
    [
        (
            "insert_block_v1",
            "simulation/insert_block_v1_gelsight_abs_joint",
            "pi05_insert_block_v1_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft",
        ),
        (
            "hardness_classify",
            "simulation/hardness_classify_gelsight_abs_joint",
            "pi05_hardness_classify_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora",
        ),
        (
            "pull_drawer",
            "simulation/pull_drawer_gelsight_abs_joint",
            "pi05_pull_drawer_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora",
        ),
        (
            "turn_gear_pair",
            "simulation/turn_gear_pair_gelsight_abs_joint",
            "pi05_turn_gear_pair_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora",
        ),
        (
            "weight_classify",
            "simulation/weight_classify_gelsight_abs_joint",
            "pi05_weight_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft",
        ),
        (
            "roughness_regrasp",
            "simulation/roughness_regrasp_gelsight_abs_joint",
            "pi05_roughness_regrasp_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora",
        ),
        (
            "roughness_classify",
            "simulation/roughness_classify_gelsight_abs_joint",
            "pi05_roughness_classify_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft",
        ),
    ],
)
def test_sim_anytouch_full_ft_configs_reuse_existing_data_and_stats(task_name, repo_id, norm_stats_config_name):
    train_config = config.get_config(
        f"pi05_{task_name}_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft"
    )

    assert train_config.project_name == "openpi-anytouch"
    assert train_config.data.repo_id == repo_id
    assert train_config.data.assets.assets_dir == f"./assets/{norm_stats_config_name}"
    assert train_config.data.state_dim == sim_policy.ABS_JOINT_STATE_DIM
    assert train_config.data.output_action_dim == sim_policy.ABS_JOINT_ACTION_DIM
    assert train_config.data.delta_joint_actions is True
    assert train_config.model.image_keys == sim_policy.TACTILE_FIRST_VISION_TACTILE_IMAGE_KEYS
    assert train_config.model.tactile_encoder_type == "anytouch2"
    assert train_config.model.anytouch_num_frames == 4
    assert train_config.model.anytouch_stride == 2
    assert train_config.model.pytorch_compile_mode is None
    assert train_config.batch_size == 64
    assert train_config.num_train_steps == 10_000


def test_insert_block_v2_anytouch_full_ft_config_uses_20k_steps():
    train_config = config.get_config(
        "pi05_insert_block_v2_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft"
    )

    assert train_config.project_name == "openpi-anytouch"
    assert train_config.data.repo_id == "simulation/insert_block_v2_gelsight_abs_joint"
    assert train_config.data.assets.assets_dir == (
        "./assets/pi05_insert_block_v2_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft"
    )
    assert train_config.data.state_dim == sim_policy.ABS_JOINT_STATE_DIM
    assert train_config.data.output_action_dim == sim_policy.ABS_JOINT_ACTION_DIM
    assert train_config.data.delta_joint_actions is True
    assert train_config.model.image_keys == sim_policy.TACTILE_FIRST_VISION_TACTILE_IMAGE_KEYS
    assert train_config.model.tactile_encoder_type == "anytouch2"
    assert train_config.batch_size == 64
    assert train_config.num_train_steps == 20_000


@pytest.mark.parametrize(
    ("config_task_name", "repo_task_name", "num_train_steps"),
    [
        ("weight_classify", "weight_classify", 10_000),
        ("hardness_classify", "hardness_classify", 10_000),
        ("roughness_classify", "roughness_classify", 10_000),
        ("roughness_regrasp", "roughness_regrasp", 10_000),
        ("insert_usb_v3_gelsight", "insert_usb_v3", 10_000),
        ("insert_block_v1", "insert_block_v1", 10_000),
        ("pull_drawer", "pull_drawer", 10_000),
        ("turn_gear_pair", "turn_gear_pair", 10_000),
        ("insert_block_v2", "insert_block_v2", 20_000),
    ],
)
def test_random_anytouch_full_ft_configs(config_task_name, repo_task_name, num_train_steps):
    train_config = config.get_config(
        f"pi05_{config_task_name}_sim_abs_joint_relative_action_vision_tactile_"
        "anytouch2_random_torch_full_ft"
    )

    assert train_config.project_name == "openpi-anytouch"
    assert train_config.data.repo_id == f"simulation/{repo_task_name}_gelsight_abs_joint"
    assert train_config.data.state_dim == sim_policy.ABS_JOINT_STATE_DIM
    assert train_config.data.output_action_dim == sim_policy.ABS_JOINT_ACTION_DIM
    assert train_config.data.delta_joint_actions is True
    assert train_config.model.tactile_encoder_type == "anytouch2"
    assert train_config.model.anytouch_random_init is True
    assert train_config.model.anytouch_checkpoint_path is None
    assert train_config.model.anytouch_trainable is True
    assert train_config.model.anytouch_projector_trainable is True
    assert train_config.model.use_tactile_expert is False
    assert train_config.model.pytorch_compile_mode is None
    assert train_config.batch_size == 64
    assert train_config.fsdp_devices == 4
    assert train_config.checkpoint_base_dir == "./checkpoints/random_anytouch"
    assert train_config.num_train_steps == num_train_steps


def test_pretrained_anytouch_config_still_requires_checkpoint():
    train_config = config.get_config(
        "pi05_insert_block_v1_sim_abs_joint_relative_action_vision_tactile_anytouch2_torch_full_ft"
    )

    assert train_config.model.anytouch_random_init is False
    assert train_config.model.anytouch_checkpoint_path is not None


@pytest.mark.parametrize(
    ("config_task_name", "repo_task_name", "num_train_steps"),
    [
        ("insert_usb_v3_gelsight", "insert_usb_v3", 10_000),
        ("insert_block_v1", "insert_block_v1", 10_000),
        ("insert_block_v2", "insert_block_v2", 20_000),
        ("hardness_classify", "hardness_classify", 10_000),
        ("pull_drawer", "pull_drawer", 10_000),
        ("turn_gear_pair", "turn_gear_pair", 10_000),
        ("weight_classify", "weight_classify", 10_000),
        ("roughness_regrasp", "roughness_regrasp", 10_000),
        ("roughness_classify", "roughness_classify", 10_000),
    ],
)
def test_anytouch_tactile_expert_configs(config_task_name, repo_task_name, num_train_steps):
    train_config = config.get_config(
        f"pi05_{config_task_name}_sim_abs_joint_relative_action_vision_tactile_"
        "anytouch2_tactile_expert_torch_full_ft"
    )

    assert train_config.project_name == "openpi-anytouch-trex"
    assert train_config.data.repo_id == f"simulation/{repo_task_name}_gelsight_abs_joint"
    assert train_config.data.state_dim == sim_policy.ABS_JOINT_STATE_DIM
    assert train_config.data.output_action_dim == sim_policy.ABS_JOINT_ACTION_DIM
    assert train_config.data.delta_joint_actions is True
    assert train_config.model.image_keys == sim_policy.TACTILE_FIRST_VISION_TACTILE_IMAGE_KEYS
    assert train_config.model.tactile_encoder_type == "anytouch2"
    assert train_config.model.use_tactile_expert is True
    assert train_config.model.tactile_expert_variant == "gemma_300m"
    assert train_config.model.flow_num_steps == 10
    assert train_config.model.action_expert_steps == 6
    assert train_config.model.tactile_loss_weight == 1.0
    assert train_config.model.anytouch_trainable is True
    assert train_config.model.anytouch_projector_trainable is True
    assert train_config.model.pytorch_compile_mode is None
    assert train_config.batch_size == 64
    assert train_config.fsdp_devices == 4
    assert train_config.num_train_steps == num_train_steps
    assert train_config.policy_metadata["flow_dispatch"] == "6+4"


def test_pull_drawer_abs_joint_configs():
    vision = config.get_config("pi05_pull_drawer_sim_abs_joint_relative_action_vision_lora")
    tactile = config.get_config("pi05_pull_drawer_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora")

    assert vision.data.repo_id == "simulation/pull_drawer_gelsight_abs_joint"
    assert vision.model.image_keys == sim_policy.VISION_IMAGE_KEYS
    assert vision.data.state_dim == sim_policy.ABS_JOINT_STATE_DIM
    assert vision.data.output_action_dim == sim_policy.ABS_JOINT_ACTION_DIM
    assert vision.data.delta_joint_actions is True
    assert vision.batch_size == 64
    assert vision.num_train_steps == 10_000
    assert vision.save_interval == 5_000
    assert vision.keep_period == 5_000

    assert tactile.data.repo_id == vision.data.repo_id
    assert tactile.model.image_keys == sim_policy.TACTILE_FIRST_VISION_TACTILE_IMAGE_KEYS
    assert tactile.model.separate_tactile_encoder is True
    assert tactile.model.tactile_image_keys == ("left_tactile_0_rgb", "right_tactile_0_rgb")
    assert tactile.data.state_dim == sim_policy.ABS_JOINT_STATE_DIM
    assert tactile.data.output_action_dim == sim_policy.ABS_JOINT_ACTION_DIM
    assert tactile.data.delta_joint_actions is True


@pytest.mark.parametrize(
    ("task_name", "repo_id"),
    [
        ("pour_ball_to_cup", "simulation/pour_ball_to_cup_gelsight_abs_joint"),
        ("roughness_regrasp", "simulation/roughness_regrasp_gelsight_abs_joint"),
        ("turn_gear_pair", "simulation/turn_gear_pair_gelsight_abs_joint"),
    ],
)
def test_sim_abs_joint_lora_configs(task_name, repo_id):
    vision = config.get_config(f"pi05_{task_name}_sim_abs_joint_relative_action_vision_lora")
    tactile = config.get_config(f"pi05_{task_name}_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora")

    assert vision.data.repo_id == repo_id
    assert vision.model.image_keys == sim_policy.VISION_IMAGE_KEYS
    assert vision.model.action_dim == 32
    assert vision.data.state_dim == sim_policy.ABS_JOINT_STATE_DIM
    assert vision.data.output_action_dim == sim_policy.ABS_JOINT_ACTION_DIM
    assert vision.data.delta_joint_actions is True

    assert tactile.data.repo_id == repo_id
    assert tactile.model.image_keys == sim_policy.TACTILE_FIRST_VISION_TACTILE_IMAGE_KEYS
    assert tactile.model.separate_tactile_encoder is True
    assert tactile.model.tactile_image_keys == ("left_tactile_0_rgb", "right_tactile_0_rgb")
    assert tactile.data.state_dim == sim_policy.ABS_JOINT_STATE_DIM
    assert tactile.data.output_action_dim == sim_policy.ABS_JOINT_ACTION_DIM
    assert tactile.data.delta_joint_actions is True


@pytest.mark.parametrize(
    ("task_name", "repo_id", "num_train_steps"),
    [
        ("insert_block_v1", "simulation/insert_block_v1_gelsight_abs_joint", 10_000),
        ("insert_block_v2", "simulation/insert_block_v2_gelsight_abs_joint", 20_000),
    ],
)
def test_insert_block_abs_joint_full_ft_configs(task_name, repo_id, num_train_steps):
    vision = config.get_config(f"pi05_{task_name}_sim_abs_joint_relative_action_vision_full_ft")
    tactile = config.get_config(
        f"pi05_{task_name}_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft"
    )

    for train_config in (vision, tactile):
        assert train_config.model.paligemma_variant == "gemma_2b"
        assert train_config.model.action_expert_variant == "gemma_300m"
        assert isinstance(train_config.freeze_filter, nnx.Nothing)
        assert train_config.data.repo_id == repo_id
        assert train_config.data.state_dim == sim_policy.ABS_JOINT_STATE_DIM
        assert train_config.data.output_action_dim == sim_policy.ABS_JOINT_ACTION_DIM
        assert train_config.data.delta_joint_actions is True
        assert train_config.batch_size == 64
        assert train_config.fsdp_devices == 4
        assert train_config.num_train_steps == num_train_steps
        assert train_config.ema_decay == 0.999
        assert train_config.save_interval == 5_000
        assert train_config.keep_period == 5_000
        assert isinstance(train_config.lr_schedule, optimizer.CosineDecaySchedule)
        assert train_config.lr_schedule.warmup_steps == 5_000
        assert train_config.lr_schedule.peak_lr == 5e-5
        assert train_config.lr_schedule.decay_steps == 1_000_000
        assert train_config.lr_schedule.decay_lr == 5e-5
        assert isinstance(train_config.optimizer, optimizer.AdamW)
        assert train_config.optimizer.clip_gradient_norm == 1.0
        assert isinstance(train_config.weight_loader, weight_loaders.CheckpointWeightLoader)
        assert train_config.weight_loader.params_path == (
            "/home/tione/notebook/data/openpi/openpi-assets/checkpoints/pi05_base/params"
        )

    assert vision.model.image_keys == sim_policy.VISION_IMAGE_KEYS
    assert vision.model.separate_tactile_encoder is False
    assert tactile.model.image_keys == sim_policy.TACTILE_FIRST_VISION_TACTILE_IMAGE_KEYS
    assert tactile.model.separate_tactile_encoder is True
    assert tactile.model.tactile_image_keys == ("left_tactile_0_rgb", "right_tactile_0_rgb")


@pytest.mark.parametrize(
    ("task_name", "repo_id", "num_train_steps"),
    [
        ("insert_block_v1", "simulation/insert_block_v1_gelsight_abs_joint", 10_000),
        ("insert_block_v2", "simulation/insert_block_v2_gelsight_abs_joint", 20_000),
    ],
)
def test_insert_block_abs_joint_lora_configs_reuse_full_ft_norm_stats(task_name, repo_id, num_train_steps):
    vision = config.get_config(f"pi05_{task_name}_sim_abs_joint_relative_action_vision_lora")
    tactile = config.get_config(
        f"pi05_{task_name}_sim_abs_joint_relative_action_vision_tactile_dual_siglip_lora"
    )

    for train_config in (vision, tactile):
        assert train_config.model.paligemma_variant == "gemma_2b_lora"
        assert train_config.model.action_expert_variant == "gemma_300m_lora"
        assert not isinstance(train_config.freeze_filter, nnx.Nothing)
        assert train_config.data.repo_id == repo_id
        assert train_config.batch_size == 64
        assert train_config.fsdp_devices == 2
        assert train_config.num_train_steps == num_train_steps
        assert train_config.ema_decay is None

    assert vision.data.assets.assets_dir == (
        f"./assets/pi05_{task_name}_sim_abs_joint_relative_action_vision_full_ft"
    )
    assert tactile.data.assets.assets_dir == (
        f"./assets/pi05_{task_name}_sim_abs_joint_relative_action_vision_tactile_dual_siglip_full_ft"
    )


def test_abs_joint_relative_action_mask_preserves_absolute_gripper():
    state = np.arange(8, dtype=np.float32)
    absolute_actions = np.stack((state + 10, state + 20))
    mask = transforms.make_bool_mask(7, -1)

    relative = transforms.DeltaActions(mask)({"state": state, "actions": absolute_actions.copy()})

    np.testing.assert_array_equal(relative["actions"][:, :7], np.array([[10] * 7, [20] * 7]))
    np.testing.assert_array_equal(relative["actions"][:, 7], absolute_actions[:, 7])

    restored = transforms.AbsoluteActions(mask)(relative)
    np.testing.assert_array_equal(restored["actions"], absolute_actions)
