import dataclasses
import functools
import logging
import os
import platform
import re
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_triton_ptxas_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".venv",
    "lib",
    "python3.11",
    "site-packages",
    "triton",
    "backends",
    "nvidia",
    "bin",
)
if os.path.isfile(os.path.join(_triton_ptxas_dir, "ptxas")):
    os.environ["PATH"] = f"{_triton_ptxas_dir}:{os.environ.get('PATH', '')}"

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)

def _env_int(*names: str, default: int | None = None) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


def _visible_device_count() -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() in ("", "NoDevFiles"):
        return None
    return len([device for device in visible.split(",") if device.strip()])


def init_jax_distributed_from_env() -> None:
    """Initialize JAX distributed training from TI-ONE or explicit env vars.

    Single-node runs remain unchanged. For multi-node TI-ONE tasks, set:
      JAX_NUM_PROCESSES=2
      JAX_PROCESS_INDEX=0 or 1
      JAX_COORDINATOR_ADDRESS=<master_addr>:<master_port>
      JAX_LOCAL_DEVICE_COUNT=8
    """
    num_processes = _env_int("JAX_NUM_PROCESSES", "TIONE_NUM_PROCESSES", "WORLD_SIZE", "NNODES", default=1)
    if num_processes is None or num_processes <= 1:
        return

    process_id = _env_int("JAX_PROCESS_INDEX", "TIONE_PROCESS_INDEX", "NODE_RANK", "RANK")
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if not coordinator_address:
        master_addr = os.environ.get("MASTER_ADDR") or os.environ.get("JAX_COORDINATOR_HOST")
        master_port = os.environ.get("MASTER_PORT") or os.environ.get("JAX_COORDINATOR_PORT") or "23456"
        if master_addr:
            coordinator_address = f"{master_addr}:{master_port}"

    if process_id is None:
        raise ValueError(
            "Multi-node JAX training requires JAX_PROCESS_INDEX, TIONE_PROCESS_INDEX, NODE_RANK, or RANK."
        )
    if not coordinator_address:
        raise ValueError(
            "Multi-node JAX training requires JAX_COORDINATOR_ADDRESS or MASTER_ADDR/MASTER_PORT."
        )

    local_device_count = _env_int("JAX_LOCAL_DEVICE_COUNT", "GPU_NUM_PER_NODE", "LOCAL_WORLD_SIZE")
    if local_device_count is None:
        local_device_count = _visible_device_count()
    local_device_ids = tuple(range(local_device_count)) if local_device_count else None

    kwargs: dict[str, Any] = {
        "coordinator_address": coordinator_address,
        "num_processes": num_processes,
        "process_id": process_id,
        "local_device_ids": local_device_ids,
        "initialization_timeout": _env_int("JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT", default=900),
    }
    if os.environ.get("JAX_COORDINATOR_BIND_ADDRESS"):
        kwargs["coordinator_bind_address"] = os.environ["JAX_COORDINATOR_BIND_ADDRESS"]

    print(
        "[jax-distributed] "
        f"coordinator={coordinator_address} process={process_id}/{num_processes} "
        f"local_device_ids={local_device_ids}",
        flush=True,
    )
    jax.distributed.initialize(**kwargs)


def is_primary_process() -> bool:
    return jax.process_index() == 0



def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader,
    params_shape: at.Params,
    *,
    allow_missing_regex: str | None = None,
) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    flat_expected = traverse_util.flatten_dict(params_shape)
    flat_loaded_all = traverse_util.flatten_dict(loaded_params)
    flat_loaded = {
        key: value
        for key, value in flat_loaded_all.items()
        if not isinstance(value, jax.ShapeDtypeStruct)
    }

    if allow_missing_regex is None:
        at.check_pytree_equality(
            expected=params_shape,
            got=loaded_params,
            check_shapes=True,
            check_dtypes=True,
        )
    else:
        missing = set(flat_expected) - set(flat_loaded_all)
        unexpected = set(flat_loaded_all) - set(flat_expected)
        missing_pattern = re.compile(allow_missing_regex)
        invalid_missing = [key for key in missing if not missing_pattern.fullmatch("/".join(key))]
        if invalid_missing or unexpected:
            raise ValueError(
                "Unexpected checkpoint parameter mismatch: "
                f"invalid_missing={invalid_missing[:20]}, unexpected={list(unexpected)[:20]}"
            )
        expected_loaded = traverse_util.unflatten_dict({key: flat_expected[key] for key in flat_loaded_all})
        at.check_pytree_equality(
            expected=expected_loaded,
            got=loaded_params,
            check_shapes=True,
            check_dtypes=True,
        )

    return traverse_util.unflatten_dict(flat_loaded)


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
            if hasattr(model, "initialize_tactile_encoder_from_vision"):
                loaded_paths = traverse_util.flatten_dict(partial_params)
                tactile_loaded = any("tactile_img" in path for path in loaded_paths)
                if not tactile_loaded:
                    model.initialize_tactile_encoder_from_vision()

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    allow_missing_regex = ".*tactile_img.*" if getattr(config.model, "separate_tactile_encoder", False) else None
    partial_params = _load_weights_and_validate(
        config.weight_loader,
        train_state_shape.params.to_pure_dict(),
        allow_missing_regex=allow_missing_regex,
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_jax_distributed_from_env()
    init_logging()
    logging.info(
        "Running on: %s; JAX process %s/%s; local devices=%s; global devices=%s",
        platform.node(),
        jax.process_index(),
        jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get("JAX_COMPILATION_CACHE_DIR", str(epath.Path("~/.cache/jax").expanduser())),
    )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled and is_primary_process())

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    if (
        is_primary_process()
        and config.wandb_enabled
        and os.environ.get("WANDB_MODE", "").lower() != "disabled"
        and os.environ.get("OPENPI_SKIP_WANDB_IMAGES", "0") != "1"
    ):
        try:
            images = jax.device_get(batch[0].images)
            image_keys = [key for key in getattr(config.model, "image_keys", tuple(images)) if key in images]
            image_keys.extend(key for key in images if key not in image_keys)
            images_to_log = [
                wandb.Image(
                    np.concatenate([np.asarray(images[key][i]) for key in image_keys], axis=1),
                    caption=" | ".join(image_keys),
                )
                for i in range(min(5, len(next(iter(images.values())))))
            ]
            wandb.log({"camera_views": images_to_log}, step=0)
        except Exception:
            logging.exception("Failed to log camera views to wandb; continuing training without image logging.")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
        disable=not is_primary_process(),
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            if is_primary_process():
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
