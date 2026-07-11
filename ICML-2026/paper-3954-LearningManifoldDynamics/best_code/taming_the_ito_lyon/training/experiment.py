import atexit
import os
import signal
import time
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from tqdm.auto import tqdm

from taming_the_ito_lyon.config import Config, load_toml_config
from taming_the_ito_lyon.config.config_options import Datasets, TrainingMode
from taming_the_ito_lyon.models import Model
from taming_the_ito_lyon.training.factories import (
    configure_jax,
    create_model,
    create_optimizer,
)
from taming_the_ito_lyon.training.io import (
    finalize_training_run,
    format_loss,
    get_run_dirname,
    write_test_metrics,
)
from taming_the_ito_lyon.training.loops import (
    run_eval_epoch,
    run_train_epoch,
)
from taming_the_ito_lyon.training.runtime import (
    build_runtime,
    trim_time_aligned_batch,
)

SAVED_MODELS_DIR = "saved_models"

T = TypeVar("T")


def _zero_trainable_leaves(tree: T) -> T:
    return jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x,
        tree,
    )


def _mask_updates_for_trainable_subtree(updates: T) -> T:
    mode = os.environ.get("TIL_TRAINABLE_SUBTREE", "").strip().lower()
    if not mode:
        return updates
    if mode == "vector_field":
        return eqx.tree_at(
            lambda u: (u.initial_cond_mlp, u.readout_layer),
            updates,
            replace=(
                _zero_trainable_leaves(updates.initial_cond_mlp),
                _zero_trainable_leaves(updates.readout_layer),
            ),
        )
    if mode == "dynamics":
        return eqx.tree_at(
            lambda u: u.readout_layer,
            updates,
            replace=_zero_trainable_leaves(updates.readout_layer),
        )
    if mode == "readout":
        return eqx.tree_at(
            lambda u: (u.initial_cond_mlp, u.vector_field),
            updates,
            replace=(
                _zero_trainable_leaves(updates.initial_cond_mlp),
                _zero_trainable_leaves(updates.vector_field),
            ),
        )
    raise ValueError(f"Unknown TIL_TRAINABLE_SUBTREE={mode!r}")


def _eval_metric_name(config: Config, loss_label: str) -> str:
    dataset_name = config.experiment_config.dataset_name
    if dataset_name in (
        Datasets.BLACK_SCHOLES,
        Datasets.BERGOMI,
        Datasets.ROUGH_BERGOMI,
        Datasets.SIMPLE_RBERGOMI,
    ):
        return "median_ks"
    if dataset_name in (
        Datasets.SG_SO3_SIMULATION,
        Datasets.OXFORD_MULTIMOTION_STATIC,
        Datasets.OXFORD_MULTIMOTION_TRANSLATIONAL,
        Datasets.OXFORD_MULTIMOTION_UNCONSTRAINED,
    ):
        if loss_label == "rge":
            return "rge"
        return "frobenius"
    if dataset_name == Datasets.SPD_WISHART_DIFFUSION:
        return "median_eigenvalue_w1"
    return loss_label


def _compiled_xla_scratch_mib(fn: object, *args: object) -> float | None:
    try:
        compiled_wrapper = fn.lower(*args).compile()
        compiled = getattr(compiled_wrapper, "compiled", compiled_wrapper)
        return float(compiled.memory_analysis().temp_size_in_bytes) / (1024.0**2)
    except Exception:
        return None


def experiment(
    config: Config,
    config_path: str | None = None,
    return_metrics: bool = False,
) -> dict[str, float | str] | None:
    configure_jax()
    model_name = config.experiment_config.model_type.value
    model_key, loader_key = jr.split(jr.PRNGKey(config.experiment_config.seed), 2)

    runtime = build_runtime(config, loader_key)

    model: Model = create_model(
        config=config,
        input_path_dim=runtime.input_path_dim,
        output_path_dim=runtime.output_head_dim,
        key=model_key,
    )
    init_checkpoint = os.environ.get("TIL_INIT_CHECKPOINT")
    if init_checkpoint:
        model = eqx.tree_deserialise_leaves(init_checkpoint, model)
        tqdm.write(f"Loaded initial checkpoint from {init_checkpoint}.")

    trainable_leaves = jax.tree_util.tree_leaves(
        eqx.filter(model, eqx.is_inexact_array)
    )
    num_params = int(sum(int(x.size) for x in trainable_leaves))
    tqdm.write(
        f"Instantiated model '{model_name}' with {num_params:,} trainable params "
        f"(input_path_dim={runtime.input_path_dim}, output_head_dim={runtime.output_head_dim})."
    )

    optim = create_optimizer(
        optimizer_name=config.experiment_config.optimizer,
        learning_rate=config.experiment_config.learning_rate,
        weight_decay=config.experiment_config.weight_decay,
        max_grad_norm=config.experiment_config.max_grad_norm,
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # NOTE: Avoid buffer donation here. With our dataloader + warmup flow we may
    # otherwise hit "Buffer has been deleted or donated" errors when arrays are
    # (re)used across steps.
    @eqx.filter_jit
    def train_step(
        control_values_b: jax.Array,
        target_b: jax.Array,
        gt_driver_b: jax.Array,
        model: Model,
        opt_state: optax.OptState,
    ) -> tuple[jax.Array, Model, optax.OptState]:
        loss_value, grads = runtime.grad_fn(
            model, control_values_b, target_b, gt_driver_b
        )
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, new_opt_state = optim.update(grads, opt_state, params)
        updates = _mask_updates_for_trainable_subtree(updates)
        updated_model: Model = eqx.apply_updates(model, updates)
        return loss_value, updated_model, new_opt_state

    train_key = None
    val_key = None
    test_key = None
    if runtime.mode == TrainingMode.UNCONDITIONAL:
        train_key = jr.PRNGKey(config.experiment_config.seed + 1)
        val_key = jr.PRNGKey(config.experiment_config.seed + 2)
        test_key = jr.PRNGKey(config.experiment_config.seed + 3)

    # Setup temporary checkpoint path
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    temp_best_path = os.path.join(SAVED_MODELS_DIR, "best.eqx")

    # Cleanup handler for temp file
    def cleanup_temp() -> None:
        if os.path.exists(temp_best_path):
            os.remove(temp_best_path)

    def sigint_handler(signum: int, frame: object) -> None:
        tqdm.write("\nInterrupted. Cleaning up temporary checkpoint...")
        cleanup_temp()
        raise SystemExit(1)

    signal.signal(signal.SIGINT, sigint_handler)
    atexit.register(cleanup_temp)

    min_val_metric = float("inf")
    min_train_loss = float("inf")
    best_epoch = -1
    time_to_best_epoch: float | None = None
    epochs_since_improve = 0
    patience = int(config.experiment_config.early_stopping_patience)
    checkpoint_on_val_loss = os.environ.get("TIL_CHECKPOINT_ON_VAL_LOSS", "0") == "1"
    checkpoint_qv_under_ks = (
        os.environ.get("TIL_CHECKPOINT_QV_UNDER_KS", "0") == "1"
    )
    checkpoint_qv_key = os.environ.get(
        "TIL_CHECKPOINT_QV_KEY", "ito_qv_x_mean_gap_abs"
    )
    checkpoint_ks_max = float(os.environ.get("TIL_CHECKPOINT_KS_MAX", "inf"))
    if checkpoint_qv_under_ks:
        checkpoint_metric_label = f"{checkpoint_qv_key}@metric<={checkpoint_ks_max:g}"
    elif checkpoint_on_val_loss:
        checkpoint_metric_label = runtime.loss_label
    else:
        checkpoint_metric_label = "metric"

    epochs = int(config.experiment_config.epochs)
    final_epoch = 0
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_metric_history: list[float] = []

    train_loader_state = runtime.train_loader_state
    val_loader_state = runtime.val_loader_state
    test_loader_state = runtime.test_loader_state

    # Warm up JITs to exclude compile time from training timer.
    warmup_batch, _, _ = runtime.train_iterate(train_loader_state)
    if runtime.mode == TrainingMode.UNCONDITIONAL:
        if runtime.unconditional_control_sampler is None:
            raise ValueError("unconditional_control_sampler required for UNCONDITIONAL")
        warmup_key = jr.PRNGKey(config.experiment_config.seed + 12345)
        warmup_controls = runtime.unconditional_control_sampler(
            runtime.ts_full, warmup_key, runtime.batch_size
        )
    else:
        warmup_controls = warmup_batch["driver"]

    warmup_controls, warmup_targets, warmup_driver = trim_time_aligned_batch(
        runtime,
        warmup_controls,
        warmup_batch["solution"],
        warmup_batch["driver"],
    )
    warmup_preds = runtime.predict_batch(warmup_controls, model)
    warmup_eval = runtime.loss_on_preds_fn(
        warmup_preds, warmup_targets, warmup_controls, warmup_driver
    )
    jax.block_until_ready(warmup_eval)

    # Warm up the (JIT-compiled) training step as well.
    warmup_loss, _, _ = train_step(
        warmup_controls,
        warmup_targets,
        warmup_driver,
        model,
        opt_state,
    )
    jax.block_until_ready(warmup_loss)

    xla_scratch_size_mib = _compiled_xla_scratch_mib(
        train_step,
        warmup_controls,
        warmup_targets,
        warmup_driver,
        model,
        opt_state,
    )

    training_start = time.perf_counter()

    epoch_bar = tqdm(
        range(epochs),
        desc="Epochs",
        position=0,
        leave=True,
    )
    for epoch_idx in epoch_bar:
        final_epoch = epoch_idx
        train_loss, model, opt_state, train_loader_state = run_train_epoch(
            runtime,
            model,
            opt_state,
            train_key,
            epoch_idx,
            train_step,
            train_loader_state,
        )
        min_train_loss = min(min_train_loss, train_loss)
        train_loss_history.append(float(train_loss))
        val_loss, val_results_dict, val_loader_state = run_eval_epoch(
            runtime,
            model,
            runtime.val_loader,
            runtime.val_iterate,
            val_loader_state,
            val_key,
            epoch_idx,
            split_label="val",
        )

        eval_metric = (
            val_results_dict.eval_metric
            if val_results_dict.eval_metric is not None
            else val_loss
        )
        val_loss_history.append(float(val_loss))
        val_metric_history.append(float(eval_metric))
        if checkpoint_qv_under_ks:
            extra_metrics = val_results_dict.extra_scalar_metrics or {}
            qv_metric = extra_metrics.get(checkpoint_qv_key)
            checkpoint_metric = (
                float(qv_metric)
                if qv_metric is not None and float(eval_metric) <= checkpoint_ks_max
                else float("inf")
            )
        elif checkpoint_on_val_loss:
            checkpoint_metric = val_loss
        else:
            checkpoint_metric = eval_metric

        if checkpoint_metric < min_val_metric:
            min_val_metric = checkpoint_metric
            best_epoch = epoch_idx
            time_to_best_epoch = time.perf_counter() - training_start
            eqx.tree_serialise_leaves(temp_best_path, model)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        epoch_bar.set_postfix(
            {
                f"train_{runtime.loss_label}": format_loss(
                    runtime.loss_label, train_loss
                ),
                f"best_val_{runtime.loss_label}": (
                    f"{format_loss(runtime.loss_label, min_val_metric)} at epoch {best_epoch}"
                ),
                "checkpoint_by": checkpoint_metric_label,
                "val_metric": format_loss(runtime.loss_label, eval_metric),
            }
        )

        if epochs_since_improve >= patience:
            tqdm.write(
                f"Early stopping at epoch {epoch_idx} (no val improvement for {patience} epochs)."
            )
            break

    training_elapsed = time.perf_counter() - training_start

    # Evaluate best model on test set with KS test
    best_model: Model = eqx.tree_deserialise_leaves(temp_best_path, model)
    inference_start = time.perf_counter()
    test_loss, test_results_dict, test_loader_state = run_eval_epoch(
        runtime,
        best_model,
        runtime.test_loader,
        runtime.test_iterate,
        test_loader_state,
        test_key,
        0,
        split_label="test",
    )
    inference_elapsed = time.perf_counter() - inference_start

    test_eval_metric = (
        test_results_dict.eval_metric
        if test_results_dict.eval_metric is not None
        else test_loss
    )
    eval_metric_name = _eval_metric_name(config, runtime.loss_label)

    run_dirname = get_run_dirname(model_name)
    run_dir = finalize_training_run(
        run_dirname=run_dirname,
        model_name=model_name,
        model=model,
        temp_best_path=temp_best_path,
        config_path=config_path,
        num_params=num_params,
        final_epoch=final_epoch,
        best_epoch=best_epoch,
        training_elapsed=training_elapsed,
        time_to_best_epoch=time_to_best_epoch,
        inference_elapsed=inference_elapsed,
        loss_label=runtime.loss_label,
        eval_metric_name=eval_metric_name,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        val_metric_history=val_metric_history,
        test_loss=test_loss,
        test_eval_metric=test_eval_metric,
        test_results_dict=test_results_dict,
        xla_scratch_size_mib=xla_scratch_size_mib,
    )
    # Unregister cleanup since we moved the file
    atexit.unregister(cleanup_temp)

    tqdm.write(
        f"\nResults: {num_params:,} params | "
        f"train {training_elapsed:.1f}s | "
        f"inference {inference_elapsed * 1000:.1f}ms | "
        f"test metric {format_loss(runtime.loss_label, test_eval_metric)}"
    )
    tqdm.write(f"Saved to: {run_dir}/")

    result = {
        "run_dir": run_dir,
        "min_val_metric": float(min_val_metric),
        "test_eval_metric": float(test_eval_metric),
        "min_train_loss": float(min_train_loss),
        "best_epoch": int(best_epoch),
    }
    if return_metrics:
        return result
    return None


def run_test(
    config: Config,
    checkpoint_path: str,
    run_dir: str | None = None,
    metrics_name: str = "test_metrics.json",
    metrics_seed: int | None = None,
) -> None:
    configure_jax()
    model_name = config.experiment_config.model_type.value
    model_key, loader_key = jr.split(jr.PRNGKey(config.experiment_config.seed), 2)

    runtime = build_runtime(config, loader_key)

    model: Model = create_model(
        config=config,
        input_path_dim=runtime.input_path_dim,
        output_path_dim=runtime.output_head_dim,
        key=model_key,
    )
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)

    trainable_leaves = jax.tree_util.tree_leaves(
        eqx.filter(model, eqx.is_inexact_array)
    )
    num_params = int(sum(int(x.size) for x in trainable_leaves))
    tqdm.write(
        f"Instantiated model '{model_name}' with {num_params:,} trainable params "
        f"(input_path_dim={runtime.input_path_dim}, output_head_dim={runtime.output_head_dim})."
    )

    test_key = None
    if runtime.mode == TrainingMode.UNCONDITIONAL:
        test_key = jr.PRNGKey(config.experiment_config.seed + 3)

    inference_start = time.perf_counter()
    test_loss, test_results_dict, _ = run_eval_epoch(
        runtime,
        model,
        runtime.test_loader,
        runtime.test_iterate,
        runtime.test_loader_state,
        test_key,
        0,
        split_label="test",
    )
    inference_elapsed = time.perf_counter() - inference_start

    test_eval_metric = (
        test_results_dict.eval_metric
        if test_results_dict.eval_metric is not None
        else test_loss
    )
    eval_metric_name = _eval_metric_name(config, runtime.loss_label)
    scratch_batch, _, _ = runtime.test_iterate(runtime.test_loader_state)
    if runtime.mode == TrainingMode.UNCONDITIONAL:
        if test_key is None or runtime.unconditional_control_sampler is None:
            raise ValueError("test_key and sampler required for UNCONDITIONAL")
        scratch_controls = runtime.unconditional_control_sampler(
            runtime.ts_full,
            jr.fold_in(test_key, 0),
            runtime.batch_size,
        )
    else:
        scratch_controls = scratch_batch["driver"]
    scratch_controls, _, _ = trim_time_aligned_batch(
        runtime,
        scratch_controls,
        scratch_batch["solution"],
        scratch_batch["driver"],
    )
    xla_scratch_size_mib = _compiled_xla_scratch_mib(
        runtime.predict_batch,
        scratch_controls,
        model,
    )

    if run_dir is not None:
        write_test_metrics(
            run_dir=run_dir,
            model_name=model_name,
            num_params=num_params,
            inference_elapsed=inference_elapsed,
            loss_label=runtime.loss_label,
            eval_metric_name=eval_metric_name,
            test_loss=test_loss,
            test_eval_metric=test_eval_metric,
            test_results_dict=test_results_dict,
            checkpoint_path=checkpoint_path,
            metrics_name=metrics_name,
            metrics_seed=metrics_seed,
            xla_scratch_size_mib=xla_scratch_size_mib,
        )

    tqdm.write(
        f"\nResults: {num_params:,} params | "
        f"inference {inference_elapsed * 1000:.1f}ms | "
        f"test metric {format_loss(runtime.loss_label, test_eval_metric)}"
    )
    if run_dir is not None:
        tqdm.write(f"Saved to: {run_dir}/")


def run_test_from_run_dir(run_dir: str) -> None:
    config_path = os.path.join(run_dir, "config.toml")
    checkpoint_path = os.path.join(run_dir, "best.eqx")
    if os.path.exists(config_path) and os.path.exists(checkpoint_path):
        config = load_toml_config(config_path)
        run_test(config, checkpoint_path=checkpoint_path, run_dir=run_dir)
        return

    if os.path.exists(config_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    subdirs = [
        os.path.join(run_dir, name)
        for name in sorted(os.listdir(run_dir))
        if os.path.isdir(os.path.join(run_dir, name))
    ]
    if len(subdirs) == 0:
        raise FileNotFoundError(
            f"No run subdirectories found under {run_dir} (expected config.toml + best.eqx)."
        )

    suppress_prev = os.environ.get("SG_SO3_SUPPRESS_INDIVIDUAL")
    os.environ["SG_SO3_SUPPRESS_INDIVIDUAL"] = "1"
    for subdir in subdirs:
        sub_config_path = os.path.join(subdir, "config.toml")
        sub_checkpoint_path = os.path.join(subdir, "best.eqx")
        if os.path.exists(sub_config_path) and os.path.exists(sub_checkpoint_path):
            config = load_toml_config(sub_config_path)
            run_test(config, checkpoint_path=sub_checkpoint_path, run_dir=subdir)

    _save_sg_so3_combined_plots(run_dir, subdirs)
    if suppress_prev is None:
        os.environ.pop("SG_SO3_SUPPRESS_INDIVIDUAL", None)
    else:
        os.environ["SG_SO3_SUPPRESS_INDIVIDUAL"] = suppress_prev


def _save_sg_so3_combined_plots(base_run_dir: str, subdirs: list[str]) -> None:
    from taming_the_ito_lyon.training.results_plotting import save_sg_so3_sphere_plot

    batch_files: list[tuple[str, str]] = []
    base_out_dir = os.environ.get(
        "SG_SO3_SPHERE_PLOT_DIR", "z_paper_content/sg_so3_sphere_by_epoch"
    )
    chosen: dict[str, tuple[str, str]] = {}
    for subdir in subdirs:
        config_path = os.path.join(subdir, "config.toml")
        if not os.path.exists(config_path):
            continue
        try:
            config = load_toml_config(config_path)
        except Exception:
            continue
        model_name = config.experiment_config.model_type.value
        batch_path = os.path.join(base_out_dir, model_name, "sg_so3_batch.npz")
        if not os.path.exists(batch_path):
            continue
        prev = chosen.get(model_name)
        if prev is None:
            chosen[model_name] = (subdir, batch_path)
            continue
        prev_subdir, _ = prev
        if (
            os.path.basename(prev_subdir) != model_name
            and os.path.basename(subdir) == model_name
        ):
            chosen[model_name] = (subdir, batch_path)

    for model_name, (_, batch_path) in sorted(chosen.items()):
        batch_files.append((model_name, batch_path))

    if len(batch_files) == 0:
        return

    batches: list[dict[str, np.ndarray]] = []
    model_names: list[str] = []
    for model_name, batch_path in batch_files:
        with np.load(batch_path) as data:
            preds = np.asarray(data["preds"])
            targets = np.asarray(data["targets"])
        if preds.ndim == 3:
            preds = preds[None, ...]
        if targets.ndim == 3:
            targets = targets[None, ...]
        if preds.ndim != 4 or targets.ndim != 4:
            continue
        batches.append({"preds": preds, "targets": targets})
        model_names.append(model_name)

    if len(batches) == 0:
        return

    min_batch = min(int(batch["preds"].shape[0]) for batch in batches)
    if min_batch <= 0:
        return

    out_dir = os.path.join(base_out_dir, "_combined")
    os.makedirs(out_dir, exist_ok=True)

    num_plots = int(os.environ.get("SG_SO3_NUM_PLOTS", "10"))
    rng = np.random.default_rng(0)
    for plot_idx in range(num_plots):
        sample_idx = int(rng.integers(0, min_batch))
        preds_stack = np.stack(
            [batch["preds"][sample_idx] for batch in batches], axis=0
        )
        targets_ref = batches[0]["targets"][sample_idx : sample_idx + 1]
        filename = f"sphere_epoch_00000_sample_{plot_idx:02d}.pdf"
        if num_plots == 1:
            filename = "sphere_epoch_00000.pdf"
        save_sg_so3_sphere_plot(
            preds=preds_stack,
            targets=targets_ref,
            out_file=os.path.join(out_dir, filename),
            n_plot=int(preds_stack.shape[0]),
            labels=model_names,
        )
