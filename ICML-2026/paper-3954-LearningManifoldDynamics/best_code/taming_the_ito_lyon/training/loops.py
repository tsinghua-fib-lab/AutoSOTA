from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from cyreal.loader import DataLoader, _LoaderState
from tqdm.auto import tqdm

from taming_the_ito_lyon.config.config_options import Datasets, TrainingMode
from taming_the_ito_lyon.models import Model
from taming_the_ito_lyon.training.io import format_loss
from taming_the_ito_lyon.training.results_gathering_fns import ResultsDict
from taming_the_ito_lyon.training.runtime import (
    ExperimentRuntime,
    align_predictions_to_targets,
    trim_time_aligned_batch,
)


def run_train_epoch(
    runtime: ExperimentRuntime,
    model: Model,
    opt_state: optax.OptState,
    epoch_key: jax.Array | None,
    epoch_idx: int,
    train_step: Callable[
        [jax.Array, jax.Array, jax.Array, Model, optax.OptState],
        tuple[jax.Array, Model, optax.OptState],
    ],
    train_loader_state: _LoaderState,
) -> tuple[float, Model, optax.OptState, _LoaderState]:
    # Keep accumulation on-device to avoid forcing a device sync every step.
    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    tqdm_every = int(runtime.config.experiment_config.tqdm_update_interval)
    pbar = tqdm(
        range(runtime.train_loader.steps_per_epoch),
        desc="Training batches",
        position=1,
        leave=False,
    )
    for step_idx in pbar:
        batch, train_loader_state, _ = runtime.train_iterate(train_loader_state)

        # Get control values based on mode
        if runtime.mode == TrainingMode.UNCONDITIONAL:
            if epoch_key is None or runtime.unconditional_control_sampler is None:
                raise ValueError(
                    "epoch_key and unconditional_control_sampler required for UNCONDITIONAL"
                )
            step_key = jr.fold_in(jr.fold_in(epoch_key, epoch_idx), step_idx)
            control_values_b = runtime.unconditional_control_sampler(
                runtime.ts_full, step_key, runtime.batch_size
            )
        else:
            control_values_b = batch["driver"]

        control_values_b, target_b, gt_driver_b = trim_time_aligned_batch(
            runtime,
            control_values_b,
            batch["solution"],
            batch["driver"],
        )
        loss_value, model, opt_state = train_step(
            control_values_b,
            target_b,
            gt_driver_b,
            model,
            opt_state,
        )
        total_loss = total_loss + loss_value

        # Only sync to host occasionally; converting device arrays to Python
        # scalars every step can dominate wall time.
        if tqdm_every > 0 and (step_idx % tqdm_every == 0):
            loss_host = float(jax.device_get(loss_value))
            pbar.set_postfix(
                {
                    f"train_{runtime.loss_label}": format_loss(
                        runtime.loss_label, loss_host
                    )
                }
            )

    steps = max(1, int(runtime.train_loader.steps_per_epoch))
    avg_loss = float(jax.device_get(total_loss)) / float(steps)
    return avg_loss, model, opt_state, train_loader_state


def run_eval_epoch(
    runtime: ExperimentRuntime,
    model: Model,
    loader: DataLoader,
    iterate: Callable[
        [_LoaderState], tuple[dict[str, jax.Array], _LoaderState, jax.Array]
    ],
    loader_state: _LoaderState,
    epoch_key: jax.Array | None,
    epoch_idx: int,
    split_label: str,
) -> tuple[float, ResultsDict, _LoaderState]:
    # Keep accumulation on-device to avoid syncing every step.
    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    tqdm_every = int(runtime.config.experiment_config.tqdm_update_interval)
    # Keep all batches for diagnostics (KS over all eval batches).
    preds_batches: list[jax.Array] = []
    targets_batches: list[jax.Array] = []
    controls_batches: list[jax.Array] = []

    pbar = tqdm(
        range(loader.steps_per_epoch),
        desc="Evaluating batches",
        position=1,
        leave=False,
    )
    for step_idx in pbar:
        batch, loader_state, _ = iterate(loader_state)

        # Get control values based on mode
        if runtime.mode == TrainingMode.UNCONDITIONAL:
            if epoch_key is None or runtime.unconditional_control_sampler is None:
                raise ValueError(
                    "epoch_key and unconditional_control_sampler required for UNCONDITIONAL"
                )
            # Hold eval drivers fixed across epochs by not folding in epoch_idx.
            step_key = jr.fold_in(epoch_key, step_idx)
            control_values_b = runtime.unconditional_control_sampler(
                runtime.ts_full, step_key, runtime.batch_size
            )
        else:
            control_values_b = batch["driver"]

        control_values_b, target_b, gt_driver_b = trim_time_aligned_batch(
            runtime,
            control_values_b,
            batch["solution"],
            batch["driver"],
        )
        # Single forward pass per batch; reuse preds for loss and metrics.
        preds = runtime.predict_batch(control_values_b, model)
        preds = align_predictions_to_targets(runtime, preds, target_b)
        loss_value = runtime.loss_on_preds_fn(
            preds, target_b, control_values_b, gt_driver_b
        )
        total_loss = total_loss + loss_value

        if tqdm_every > 0 and (step_idx % tqdm_every == 0):
            loss_host = float(jax.device_get(loss_value))
            pbar.set_postfix(
                {
                    f"{split_label}_{runtime.loss_label}": format_loss(
                        runtime.loss_label, loss_host
                    )
                }
            )

        preds_batches.append(preds)
        targets_batches.append(target_b)
        controls_batches.append(control_values_b)

    steps = max(1, int(loader.steps_per_epoch))
    avg_loss = float(jax.device_get(total_loss)) / float(steps)

    # Compute diagnostics (plots/KS/etc.) using all eval batches.
    results_dict = runtime.results_gathering_fn(
        preds_batches,
        targets_batches,
        None
        if runtime.config.experiment_config.dataset_name == Datasets.PPG_DALIA
        else controls_batches,
        epoch_idx,
        runtime.config.experiment_config.model_type.value,
        n_plot=min(8, int(runtime.batch_size)),
        config=runtime.config,
    )
    return avg_loss, results_dict, loader_state
