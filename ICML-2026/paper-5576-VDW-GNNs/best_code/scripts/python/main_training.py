#!/usr/bin/env python3

"""
Main training script. Note that in this version, we have added 
special support for the VDW-family models and comparisons.
"""
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import TensorDataset
import yaml
import time
import pickle
from datetime import datetime
from accelerate import Accelerator
from accelerate.utils import (
    broadcast_object_list, 
    DistributedDataParallelKwargs
)
# Import guard for wandb
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    wandb = None

# Add the code directory to sys.path (go up 2 levels from scripts/python/)
_script_dir = Path(__file__).resolve().parent
_code_dir = _script_dir.parent.parent
sys.path.insert(0, str(_code_dir))

from train_utils import *
from config.arg_parsing import get_clargs
from config.config_manager import ConfigManager
from config.train_config import TrainingConfig
import training.train as train
from models.base_module import test_nn, MetricDefinitions
from models.nn_utilities import count_parameters
from training.prep_dataset import load_dataset
from training.prep_dataloaders import create_dataloaders
from training.metrics_utils import metric_to_str
from training.prep_optimizer import prepare_optimizer, prepare_scheduler
from training.kfold_results import compute_kfold_results
from os_utilities import (
    create_experiment_dir, 
    smart_pickle, 
    ensure_dir_exists,
)
from data_processing.data_utilities import (
    get_random_splits,
    get_kfold_splits,
    process_kfold_splits,
    # multi_strat_multi_fold_idx_sample,  # not yet implemented
)
from data_processing.ellipsoids_utils import attach_tnn_operators

from training.prep_model import (
    prepare_vdw_model, 
    prepare_comparison_model,
)
from data_processing.macaque_reaching import (
    macaque_prepare_kth_fold,
    macaque_prepare_spatial_graph_only_for_fold,
)
from training.macaque_inductive_utils import (
    build_train_graph,
    evaluate_supcon2_with_svm,
    get_processed_dataset_path,
    load_processed_train_graph,
    resolve_processed_dataset_override,
    save_embeddings,
    save_processed_train_graph,
    _run_cluster_probes_on_embeddings, 
    _maybe_save_fold_embeddings,
)


def run_one_fold(
    config: TrainingConfig,
    dataloader_dict: Dict[str, Any],
    fold_idx: int = 0,
    save_results: bool = True,
    accelerator: Optional[Accelerator] = None,
    post_fold_hook: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run a single fold of training, validation, and testing for a model.
    
    This function handles the complete training pipeline for one fold, including:
    1. Preparing the optimizer and learning rate scheduler classes and kwargs
    2. Initializing the model
    3. Training the model
    4. Testing the model on the appropriate evaluation set
    5. Saving results if specified ('smart_pickle' prevents overwriting existing 
       files)
    
    Args:
        config: TrainingConfig object containing all training parameters and configurations
        dataloader_dict: Dictionary mapping set names ('train', 'valid', 'test') to their
            respective DataLoader objects
        fold_idx: Index of the current fold (default: 0, used for k-fold cross-validation)
        save_results: Whether to save the results to disk (default: True)
        accelerator: Optional Accelerator object for distributed training
        
    Returns:
        Dictionary containing:
        - 'model': Name of the model used
        - 'dataset': Name of the dataset used
        - 'fold_i': Index of the current fold
        - 'epoch_times': List of times taken for each epoch up to the best epoch
        - Metric scores from testing (keys depend on the task and metrics used)
        
    Raises:
        ValueError: If the model key in config is not supported
    """
    
    # Use the shared main_print function
    def _main_print(*args, **kwargs):
        main_print(*args, acc=accelerator, config=config, **kwargs)
    
    # Prepare optimizer and (optional) scheduler 
    # prep_start_time = time.time()
    # _main_print(f"\n{'=' * 80}")
    # _main_print(f"STARTING OPTIMIZER/SCHEDULER PREPARATION - FOLD {fold_idx}", timestamp=True)
    optimizer_class, optimizer_kwargs = prepare_optimizer(config.optimizer_config)
    if config.scheduler_config.scheduler_key:
        scheduler_class, scheduler_kwargs = prepare_scheduler(
            config.scheduler_config,
            config.validate_every
        )
    else:
        scheduler_class, scheduler_kwargs = None, None

    # prep_elapsed = time.time() - prep_start_time
    # prep_min, prep_sec = prep_elapsed // 60, prep_elapsed % 60
    # _main_print(f"Complete.")
    # _main_print(f"Total optimizer/scheduler preparation time: {prep_min}m {prep_sec:.2f}s")
    # _main_print(f"{'=' * 80}")

    # Prepare model
    model_start_time = time.time()
    _main_print(f"\n{'=' * 80}")
    _main_print(f"FOLD {fold_idx}: STARTING MODEL PREPARATION ['{config.model_config.model_key}']", timestamp=True)

    key_lower = config.model_config.model_key.lower()
    if 'vdw' in key_lower:
        # VDW-family models (vdw, vdw_radial, vdw_modular, etc.)
        config, model, dataloader_dict = prepare_vdw_model(
            config, 
            dataloader_dict, 
            acc=accelerator
        )
    elif key_lower in ('egnn', 'tfn', 'legs', 'gin', 'gat', 'gcn'):
        # Comparison models prepared via dedicated wrapper
        config, model, dataloader_dict = prepare_comparison_model(
            config, 
            dataloader_dict, 
            acc=accelerator
        )
    else:
        raise ValueError(
            f"Model key '{config.model_config.model_key}' not supported"
        )

    supcon_context = getattr(config, "_current_supcon2_context", None)
    if (
        config.model_config.model_key == 'vdw_supcon_2'
        and supcon_context is not None
    ):
        real_model = model.module if hasattr(model, 'module') else model
        if hasattr(real_model, "set_supcon2_context"):
            real_model.set_supcon2_context(
                context=supcon_context,
                accelerator=accelerator,
                eval_fn=evaluate_supcon2_with_svm,
            )

    if config.verbosity > 0:
        if config.dataset_config.target_preprocessing_type is not None:
            _main_print(f"Target rescaling stats set in model: center: {model._target_center}, scale: {model._target_scale}")
        else:
            _main_print(f"No target rescaling stats set in model")
    model_elapsed = time.time() - model_start_time
    model_min, model_sec = model_elapsed // 60, model_elapsed % 60
    _main_print(f"Complete.")
    _main_print(f"Total model preparation time: {int(model_min)}m {model_sec:.2f}s")

    # TRAINING START
    train_start_time = time.time()
    _main_print(f"\n{'=' * 80}")
    _main_print(f"FOLD {fold_idx}: STARTING TRAINING", timestamp=True)
        
    # Train model
    trained_model, records, epoch_ctr = train.train_model(
        config,
        dataloader_dict,
        model,
        optimizer_class,
        optimizer_kwargs,
        scheduler_class,
        scheduler_kwargs,
        accelerator=accelerator
    )

    best_epoch = None
    try:
        if epoch_ctr is not None:
            best_epoch = int(epoch_ctr.best.get(config.main_metric, {}).get('epoch', 0))
    except Exception:
        best_epoch = None

    best_epoch_fallback = 0
    if records:
        best_epoch_fallback = int(records[-1].get('epoch', 0))
    best_epoch_value = best_epoch if (best_epoch is not None and best_epoch > 0) else best_epoch_fallback

    mean_train_time = None
    mean_infer_time = None
    if records and best_epoch_value > 0:
        subset = [r for r in records if r.get('epoch', 0) <= best_epoch_value]
        if subset:
            mean_train_time = sum(float(r.get('train_time_sec', 0.0)) for r in subset) / len(subset)
            mean_infer_time = sum(float(r.get('valid_inference_time_sec', 0.0)) for r in subset) / len(subset)
    if mean_train_time is None:
        mean_train_time = float("nan")
    if mean_infer_time is None:
        mean_infer_time = float("nan")
    
    # Expose timing and best-epoch stats to downstream hooks (if a mutable context exists)
    propagate_training_summary_to_context(
        getattr(config, "_current_supcon2_context", None),
        best_epoch=best_epoch_value,
        mean_train_time=mean_train_time,
        mean_infer_time=mean_infer_time,
    )

    # TRAINING COMPLETE
    train_elapsed = time.time() - train_start_time
    _main_print(f"Complete.")
    train_min, train_sec = train_elapsed // 60, train_elapsed % 60
    _main_print(f"Training time: {int(train_min)}m {train_sec:.2f}s")
    # _main_print(f"{'=' * 80}")

    # Optional: visualize embeddings on the training set (main process only)
    if accelerator is not None and accelerator.is_main_process:
        try:
            model_for_vis = trained_model.module if hasattr(trained_model, 'module') else trained_model
            if config.model_save_dir is not None:
                fold_root = os.path.dirname(config.model_save_dir)
                embeddings_vis_dir = os.path.join(fold_root, 'embeddings')
                try:
                    ensure_dir_exists(embeddings_vis_dir, raise_exception=True)
                except Exception as err:
                    _main_print(
                        f"[WARNING] Could not create embeddings visualization directory '{embeddings_vis_dir}': {err}",
                        timestamp=True
                    )
                    embeddings_vis_dir = config.model_save_dir

                labels_attr = config.dataset_config.target_key
                if isinstance(labels_attr, str) and ('+' in labels_attr):
                    parts = [p.strip() for p in labels_attr.split('+') if p.strip()]
                    labels_attr = next(
                        (p for p in parts if 'vel' in p),
                        parts[0] if parts else labels_attr
                    )
                model_for_vis.visualize_embeddings(
                    dataloader_dict=dataloader_dict,
                    save_dir=embeddings_vis_dir,
                    labels_attr=labels_attr,
                )
        except Exception as e:
            _main_print(f"[WARNING] visualize_embeddings() raised an error: {e}", timestamp=True)

    # Ensure we have an accelerator for DDP-aware testing and results saving
    if accelerator is None:
        raise ValueError("accelerator parameter is required for DDP-aware testing and results saving")

    extra_fold_results = {
        "best_epoch": int(best_epoch_value) if best_epoch_value is not None else 0,
        "mean_train_time": float(mean_train_time),
        "mean_infer_time": float(mean_infer_time),
    }

    # -------------------------------------------------
    # EVALUATION
    # -------------------------------------------------
    eval_start_time = time.time()
    _main_print(f"\n{'=' * 80}")
    _main_print(f"FOLD {fold_idx}: STARTING EVALUATION", timestamp=True)

    # Test model (DDP-aware: all processes evaluate their subset)
    eval_set_key = 'test'
    # The evaluation set is 'test' unless we're using k-fold cross-validation
    # and we are not using a test set.
    if (('kfold' in config.experiment_type) \
         and (not config.use_cv_test_set)):
        eval_set_key = 'valid'
    metrics_kwargs = {}
    metrics_kwargs['num_outputs'] = config.dataset_config.target_dim
    if 'class' in config.dataset_config.task.lower():
        metrics_kwargs['num_classes'] = config.dataset_config.target_dim
    
    if 'cluster' in config.dataset_config.task.lower():
        metrics_kwargs['eval_cluster_metrics'] = config.model_config.eval_cluster_metrics

    # Set model to eval mode for all processes
    trained_model.eval()
    
    # Ensure all processes are synchronized before evaluation
    accelerator.wait_for_everyone()

    # Call the new evaluation function
    _eval_target_name = config.dataset_config.target_key
    if isinstance(_eval_target_name, str) and ('+' in _eval_target_name):
        parts = [p.strip() for p in _eval_target_name.split('+') if p.strip()]
        _eval_target_name = next((p for p in parts if 'vel' in p), parts[0] if parts else _eval_target_name)

    fold_results_dict = run_evaluation(
        config=config,
        dataloader_dict=dataloader_dict,
        trained_model=trained_model,
        accelerator=accelerator,
        fold_idx=fold_idx,
        eval_set_key=eval_set_key,
        metrics_kwargs=metrics_kwargs,
        eval_start_time=eval_start_time,
        extra_fold_results=extra_fold_results,
    )

    # Ensure all processes wait for testing and saving to complete
    accelerator.wait_for_everyone()

    _maybe_save_fold_embeddings(
        config=config,
        dataloader_dict=dataloader_dict,
        trained_model=trained_model,
        accelerator=accelerator,
        fold_idx=fold_idx,
    )

    # [Special case] Macaque: Run regression diagnostics and print detailed report
    if 'macaque' in config.dataset_config.dataset.lower() \
    and 'regression' in config.dataset_config.task.lower():
        try:
            from tests.macaque_diagnostics import run_diagnostics
            _ = run_diagnostics(
                model=trained_model,
                train_loader=dataloader_dict['train'],
                val_loader=dataloader_dict['valid'],
                device=accelerator.device,
                verbose=config.verbosity > -1
            )
        except ImportError as e:
            main_print(f"[WARNING] Could not import macaque diagnostics: {e}", acc=accelerator, config=config)
            main_print(f"[WARNING] Skipping diagnostics. Check that tests/__init__.py exists.", acc=accelerator, config=config)

    if callable(post_fold_hook):
        real_model = trained_model.module if hasattr(trained_model, 'module') else trained_model
        post_fold_hook(real_model, accelerator)

    return fold_results_dict





def run_evaluation(
    config: TrainingConfig,
    dataloader_dict: Dict[str, Any],
    trained_model: torch.nn.Module,
    accelerator: Accelerator,
    fold_idx: int = 0,
    eval_set_key: str = 'test',
    metrics_kwargs: Optional[Dict[str, Any]] = None,
    eval_start_time: Optional[float] = None,
    extra_fold_results: Optional[Dict[str, Any]] = None,
):
    """
    Run evaluation of a model and dataloader_dict using accelerator; print and save results.
    """

    def _sanitize_for_yaml(results_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep only YAML-friendly scalars; drop multi-element arrays/tensors (e.g., predictions).
        """
        sanitized = {}
        for k, v in results_dict.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    sanitized[k] = v.item()
                # skip multi-element tensors to avoid truncation and YAML errors
                continue
            if isinstance(v, np.ndarray):
                if v.size == 1:
                    sanitized[k] = v.item()
                # skip multi-element arrays (e.g., preds) for YAML
                continue
            if isinstance(v, (float, int, str, bool)):
                sanitized[k] = v
                continue
            # Leave other types out of YAML
        return sanitized
    if metrics_kwargs is None:
        metrics_kwargs = {}
    if extra_fold_results is None:
        extra_fold_results = {}
    if eval_start_time is None:
        eval_start_time = time.time()
    
    # All processes evaluate their subset of data
    main_print(f"Running evaluation on {eval_set_key.upper()} set...", acc=accelerator, config=config)
    log_to_train_print(f"Running evaluation on {eval_set_key.upper()} set...", acc=accelerator, config=config)
    
    # Resolve evaluation target name: support compound target keys like 'vel_xy+pos_xy'
    _eval_target_name = config.dataset_config.target_key
    if isinstance(_eval_target_name, str) and ('+' in _eval_target_name):
        parts = [p.strip() for p in _eval_target_name.split('+') if p.strip()]
        _eval_target_name = next((p for p in parts if 'vel' in p), parts[0] if parts else _eval_target_name)
    
    # Derive num_outputs from the chosen target tensor shape on a sample batch
    mk = dict(metrics_kwargs) if isinstance(metrics_kwargs, dict) else {}
    eval_cluster_metrics = mk.pop('eval_cluster_metrics', None)
    eval_cluster_metric_kwargs = mk.pop('eval_cluster_metric_kwargs', {})
    try:
        loader = dataloader_dict.get(eval_set_key)
        if loader is not None:
            first_batch = next(iter(loader))
            if hasattr(first_batch, _eval_target_name):
                tgt = getattr(first_batch, _eval_target_name)
                if isinstance(tgt, torch.Tensor) and tgt.dim() >= 2:
                    mk['num_outputs'] = int(tgt.shape[1])
                else:
                    mk['num_outputs'] = 1
    except Exception:
        # Fallback: keep provided num_outputs or default
        if 'num_outputs' not in mk:
            mk['num_outputs'] = 1
    
    metric_scores_dict = test_nn(
        trained_model=trained_model,
        data_container=dataloader_dict,
        task=config.dataset_config.task,
        device=accelerator.device,  # Use accelerator device instead of config.device
        target_name=_eval_target_name,
        set_key=eval_set_key,
        metrics_kwargs=mk,
        using_pytorch_geo=config.using_pytorch_geo,
        accelerator=accelerator  # Pass accelerator for DDP synchronization
    )

    # Ensure all processes are synchronized after evaluation
    accelerator.wait_for_everyone()

    # Only main process handles results and printing
    if accelerator.is_main_process:
        # Parameter counts
        try:
            _model_for_count = trained_model.module if hasattr(trained_model, 'module') else trained_model
            num_params_total = sum(p.numel() for p in _model_for_count.parameters())
            num_params_trainable = sum(p.numel() for p in _model_for_count.parameters() if p.requires_grad)
        except Exception:
            num_params_total, num_params_trainable = None, None

        fold_results_dict = {
            'model': config.model_config.model_key,
            'dataset': config.dataset_config.dataset,
            'fold_i': fold_idx,
            'num_params_total': num_params_total,
            'num_params_trainable': num_params_trainable,
        }
        if extra_fold_results:
            fold_results_dict.update(extra_fold_results)
        
        # Print evaluation set metric scores
        if config.dataset_config.target_dim > 1:
            main_print(f"[Target dim: {config.dataset_config.target_dim}]", acc=accelerator, config=config)
            log_to_train_print(f"[Target dim: {config.dataset_config.target_dim}]", acc=accelerator, config=config)
        printable_metrics = MetricDefinitions.get_printable_metrics_for_task(
            config.dataset_config.task
        )
        # Heading for test metrics per fold
        log_to_train_print(f"[TEST METRICS] Fold {fold_idx} results:", acc=accelerator, config=config)
        for metric, score in metric_scores_dict.items():
            score = score.detach().cpu().numpy()
            # Preserve array vs scalar in results dict as-is
            fold_results_dict[metric] = score

            if (config.verbosity > -1) and (metric in printable_metrics):
                _line = '\t' + metric_to_str(metric, score)
                main_print(_line, acc=accelerator, config=config)
                log_to_train_print(_line, acc=accelerator, config=config)

        # Save results (only main process)
        if hasattr(config, 'results_save_dir') and hasattr(config, 'results_filename'):
            import os
            from os_utilities import smart_pickle
            results_filepath = os.path.join(
                config.results_save_dir,
                config.results_filename
            )
            smart_pickle(results_filepath, fold_results_dict, overwrite=False)
            # Also write a YAML summary for downstream aggregation
            try:
                yaml_path = os.path.join(
                    config.results_save_dir,
                    "results.yaml"
                )
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(_sanitize_for_yaml(fold_results_dict), f)
            except Exception as exc:
                main_print(f"[WARNING] Could not write results.yaml: {exc}", acc=accelerator, config=config)
        
        # EVALUATION COMPLETE
        import time
        eval_elapsed = time.time() - eval_start_time
        eval_min, eval_sec = eval_elapsed // 60, eval_elapsed % 60
        main_print(f"Complete.", acc=accelerator, config=config)
        log_to_train_print("Complete.", acc=accelerator, config=config)
        main_print(f"Evaluation time: {int(eval_min)}m {eval_sec:.2f}s", acc=accelerator, config=config)
        log_to_train_print(f"Evaluation time: {int(eval_min)}m {eval_sec:.2f}s", acc=accelerator, config=config)
    else:
        # Non-main processes: don't need to save results
        fold_results_dict = None
        
    return fold_results_dict


# --------------------------------------------------------------
# Helper – warn if global batch size exceeds split sizes
# (this could lead to cryptic errors like "UnboundLocalError: cannot access 
# local variable 'current_batch' where it is not associated with a value")
# --------------------------------------------------------------
# def _batch_size_sanity_check(split_sizes: dict, train_dl, acc, fold_idx=None):
#     """Print an all-caps warning if global batch size > any split size."""
#     try:
#         local_bs = getattr(train_dl, 'batch_size', None)
#         global_bs = getattr(train_dl, 'total_batch_size', None)
#         if global_bs is None and local_bs is not None:
#             global_bs = local_bs * acc.num_processes \
#                 if not acc.split_batches else local_bs

#         if global_bs is None:
#             return

#         for split_name, split_size in split_sizes.items():
#             if split_size > 0 and split_size < global_bs:
#                 fold_str = f" IN FOLD {fold_idx}" if fold_idx is not None else ""
#                 acc.print(
#                     f"*** WARNING: GLOBAL BATCH SIZE ({global_bs}) IS LARGER THAN {split_name.upper()} SPLIT SIZE ({split_size}){fold_str}. "
#                     f"THIS CAN CREATE EMPTY SHARDS AND RUNTIME ERRORS. REDUCE batch_size OR GPU COUNT.***"
#                 )
#                 break
#     except Exception as _e:
#         acc.print(f"[DEBUG] Batch-size validation skipped: {_e}")


# def _batch_remainder_sanity_check(
#     split_sizes: dict,
#     dataloader_dict: Dict[str, Any],
#     acc,
#     fold_idx: Optional[int] = None,
#     min_multiplier: int = 4,
#     min_remainder_ratio: float = 0.5
# ) -> None:
#     """
#     Warn in ALL CAPS when, for small splits, drop_last would drop a large remainder.

#     For each available split dataloader:
#       - compute effective global batch size
#       - if split_size < min_multiplier * global_bs and drop_last is True, 
#         ensure remainder < (min_remainder_ratio * global_bs)
#       - otherwise, warn to adjust batch size
#     """
#     try:
#         for split_name, split_size in split_sizes.items():
#             dl = dataloader_dict.get(split_name)
#             if dl is None:
#                 continue

#             local_bs = getattr(dl, 'batch_size', None)
#             global_bs = getattr(dl, 'total_batch_size', None)
#             if global_bs is None and local_bs is not None:
#                 global_bs = local_bs * acc.num_processes if not acc.split_batches else local_bs

#             # Respect user's preference: avoid getattr with non-None defaults
#             if hasattr(dl, 'drop_last'):
#                 drop_last = dl.drop_last
#             else:
#                 drop_last = None

#             try:
#                 global_bs = int(global_bs) if global_bs is not None else None
#             except Exception:
#                 global_bs = None

#             if (global_bs is None) or (global_bs <= 0) or (split_size is None) or (split_size <= 0):
#                 continue

#             if (split_size < min_multiplier * global_bs) and bool(drop_last):
#                 remainder = split_size % global_bs
#                 if remainder >= (min_remainder_ratio * global_bs):
#                     fold_str = f" IN FOLD {fold_idx}" if fold_idx is not None else ""
#                     acc.print(
#                         f"*** WARNING: {split_name.upper()} SPLIT SIZE ({split_size}) IS < 4x GLOBAL BATCH ({global_bs}){fold_str}. "
#                         f"WITH drop_last=True THIS WILL DROP {remainder} SAMPLES (>= HALF A BATCH). CONSIDER ADJUSTING batch_size. ***"
#                     )
#     except Exception as _e:
#         acc.print(f"[DEBUG] Batch remainder validation skipped: {_e}")


def _batching_sanity_check(
    split_sizes: dict,
    dataloader_dict: Dict[str, Any],
    acc,
    fold_idx: Optional[int] = None,
    # min_multiplier: int = 4,
    min_remainder_ratio: float = 0.5
) -> None:
    """
    Validate batching across ALL splits:
      1) Warn if global batch size > split size (empty shards risk)
      2) When drop_last=True and split is small, warn if dropped remainder >= threshold
    """
    min_multiplier = acc.num_processes
    try:
        for split_name, split_size in split_sizes.items():
            dl = dataloader_dict.get(split_name)
            if dl is None:
                continue

            local_bs = getattr(dl, 'batch_size', None)
            global_bs = getattr(dl, 'total_batch_size', None)
            if global_bs is None and local_bs is not None:
                global_bs = local_bs * acc.num_processes if not acc.split_batches else local_bs

            # Respect preference: avoid getattr with non-None default
            drop_last = dl.drop_last if hasattr(dl, 'drop_last') else None

            try:
                global_bs = int(global_bs) if global_bs is not None else None
            except Exception:
                global_bs = None

            if (global_bs is None) or (global_bs <= 0) or (split_size is None) or (split_size <= 0):
                continue

            # Check 1: batch size vs split size
            if split_size > 0 and split_size < global_bs:
                fold_str = f" IN FOLD {fold_idx}" if fold_idx is not None else ""
                acc.print(
                    f"*** WARNING: GLOBAL BATCH SIZE ({global_bs}) IS LARGER THAN {split_name.upper()} SPLIT SIZE ({split_size}){fold_str}. "
                    f"THIS CAN CREATE EMPTY SHARDS AND RUNTIME ERRORS. REDUCE batch_size OR GPU COUNT.***"
                )
                # Do not break; check remainder too for this split

            # Check 2: drop_last remainder magnitude
            if (split_size < (min_multiplier * global_bs)) and bool(drop_last):
                remainder = split_size % global_bs
                if remainder >= (min_remainder_ratio * global_bs):
                    fold_str = f" IN FOLD {fold_idx}" if fold_idx is not None else ""
                    # Use variables in message to match thresholds
                    ratio_pct = int(min_remainder_ratio * 100)
                    acc.print(
                        f"*** WARNING: {split_name.upper()} SPLIT SIZE ({split_size}) IS < {min_multiplier} x GLOBAL BATCH ({global_bs}){fold_str}. "
                        f"WITH drop_last=True THIS WILL DROP {remainder} SAMPLES (>= {ratio_pct}% OF A BATCH). CONSIDER ADJUSTING batch_size. ***"
                    )
    except Exception as _e:
        acc.print(f"[DEBUG] Batching validation skipped: {_e}")


def print_first_batch_summary(
    dataloader_dict: Dict[str, Any],
    set_key: str,
    acc: Accelerator,
    config: TrainingConfig,
) -> None:
    """
    Print a brief summary of the first batch from a specified dataloader set.

    Runs only on the main process. Handles PyG `Batch`/`Data`, tuple/list of
    tensors, and generic objects by type.
    """
    try:
        if not acc.is_main_process:
            return
        if (set_key not in dataloader_dict) or (dataloader_dict[set_key] is None):
            return

        first_batch = next(iter(dataloader_dict[set_key]))
        heading = f"[INFO] First {set_key.upper()} batch:"

        # PyG Batch/Data prints shapes via __repr__
        cls_name = first_batch.__class__.__name__
        if hasattr(first_batch, 'to_data_list') or cls_name in ('Batch', 'Data'):
            main_print(f"{heading} object:", acc=acc, config=config)
            main_print(f"{first_batch}", acc=acc, config=config)
            return

        # Tuple/list batches
        if isinstance(first_batch, (tuple, list)):
            shape_strs: List[str] = []
            for i, item in enumerate(first_batch):
                if isinstance(item, torch.Tensor):
                    shape_strs.append(f"arg{i} shape: {tuple(item.shape)}")
                else:
                    shape_strs.append(f"arg{i} type: {type(item).__name__}")
            main_print(f"{heading} (non-PyG) shapes:", acc=acc, config=config)
            main_print("; ".join(shape_strs), acc=acc, config=config)
            return

        # Fallback
        main_print(f"{heading} type: {type(first_batch)}", acc=acc, config=config)
    except Exception as e:
        main_print(f"[WARNING] Could not sample first {set_key.upper()} batch: {e}", acc=acc, config=config)


def main(clargs):
    # EXPERIMENT START
    experiment_start_time = time.time()
    
    # Load configuration, overriding with clargs
    config_manager = ConfigManager(clargs)
    config = config_manager.config
    
    # Set environment variable to suppress device warnings
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    # Set up wandb offline mode if requested or on SLURM
    if getattr(config, 'use_wandb_logging', False):
        if getattr(config, 'wandb_offline', False) \
        or os.environ.get('SLURM_JOB_ID'):
            os.environ['WANDB_MODE'] = 'offline'

    # Initialize accelerator object
    # Note: we disable automatic device placement to avoid issues with batching PyG Data objects
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=config.ddp_find_unused_parameters
    )

    acc = Accelerator(
        device_placement=False \
            if config.using_pytorch_geo else config.device,  
        # cpu=(not torch.cuda.is_available()),
        mixed_precision='no' \
            if config.mixed_precision == 'none' else config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        split_batches=config.dataloader_split_batches,
        log_with='wandb' \
            if (getattr(config, 'use_wandb_logging', False) \
                and _WANDB_AVAILABLE) \
            else None,
        kwargs_handlers=[ddp_kwargs] \
            if config.ddp_find_unused_parameters else None
    )

    # If using wandb, initialize tracker (only once, after Accelerator is created)
    if getattr(config, 'use_wandb_logging', False):
        if _WANDB_AVAILABLE:
            project_name = getattr(config, 'experiment_id', 'vdw')
            wandb_init_kwargs = {}

            # Set wandb mode to offline if requested or on SLURM    
            if getattr(config, 'wandb_offline', False) or os.environ.get('SLURM_JOB_ID'):
                wandb_init_kwargs['mode'] = 'offline'

            # Initialize wandb tracker
            acc.init_trackers(
                project_name=project_name,
                config=config.__dict__,
                init_kwargs={'wandb': wandb_init_kwargs}
            )
        else:
            acc.print('[WARNING] wandb not installed, but use_wandb_logging=True')

    # Print PyTorch version
    main_print(f"PyTorch version: {torch.__version__}", acc=acc)
    
    # Print CUDA info
    if torch.cuda.is_available():
        try:
            main_print(f"CUDA info:", acc=acc)
            main_print(f"   - version: {torch.version.cuda}", acc=acc)
            main_print(f"   - device count: {torch.cuda.device_count()}", acc=acc)
        except Exception as e:
            main_print(f"Error getting CUDA info: {e}", acc=acc)
    
    # Print accelerator info
    main_print(f"Accelerator info:", acc=acc)
    main_print(f"   - num. processes: {acc.num_processes}", acc=acc)
    main_print(f"   - distributed type: {acc.distributed_type}", acc=acc)

    # Print warning if running in single-process mode with multiple GPUs
    if torch.cuda.device_count() > 1 and acc.num_processes == 1:
        main_print("WARNING: Multiple GPUs detected but running in single-process mode!", acc=acc)
        main_print("To use multiple GPUs, launch with: accelerate launch --num_processes=N --multi_gpu script.py", acc=acc)
    # elif acc.num_processes > 1:
    #     main_print(f"Multi-GPU training enabled with {acc.num_processes} processes", acc=acc)
    
    # Ensure experiment_id is deterministic for DDP (all processes use same ID)
    if config.experiment_id is None:
        # Generate a deterministic experiment ID based on current time
        # All processes will generate the same timestamp since they start simultaneously
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_id = f"exp_{timestamp}"
    
    # Only main process creates directories, then broadcast exact paths to all ranks
    shared_dirs_list = [None]
    if acc.is_main_process and config.snapshot_name is None:
        # Use a stable experiment directory name for VDW-family models
        mk_lower = str(getattr(config.model_config, "model_key", "")).lower()
        if "vdw" in mk_lower:
            model_slug = "vdw"
        else:
            model_slug = config.model_config.model_key
            try:
                tgt_key = getattr(config.dataset_config, 'target_key', None)
                if isinstance(tgt_key, str) and tgt_key:
                    safe_tgt = ''.join(c if c.isalnum() or c in ('-', '_') else '-' for c in tgt_key)
                    model_slug = f"{model_slug}_{safe_tgt}"
            except Exception:
                pass

        # For CV runs, create only the parent experiment directory now; per-fold subdirs will be created later.
        dirs = create_experiment_dir(
            root_dir=config.save_dir,
            model_name=model_slug,
            dataset_name=config.dataset_config.dataset,
            experiment_id=config.experiment_id,
            config=config_manager.config.__dict__,
            config_format='yaml',
            verbosity=config.verbosity,
            create_subdirs=(config.experiment_type == 'tvt')
        )
        if dirs is None:
            main_print("Error: Could not create experiment directories. Exiting.", acc=acc)
            return
        shared_dirs_list[0] = dirs

    # Broadcast the directory map so every rank uses the same (possibly suffixed) model directory
    if config.snapshot_name is None:
        broadcast_object_list(shared_dirs_list)
        dirs = shared_dirs_list[0]
        if dirs is None:
            main_print("Error: Directory broadcast failed.", acc=acc)
            return

        # Update config paths on all ranks consistently
        if config.experiment_type == 'tvt':
            config.results_save_dir = dirs['metrics']
            config.model_save_dir = dirs['models'] if config.model_save_dir is None else config.model_save_dir
            config.train_logs_save_dir = dirs['logs']
            config.config_save_path = dirs['config_save_path']
        else:
            # For CV, leave top-level subdirs unset; will be set per fold
            config.results_save_dir = None
            config.model_save_dir = None
            config.train_logs_save_dir = None
            config.config_save_path = None

    # ------------------------------------------------------------------
    # Resume-from-snapshot adjustments
    # ------------------------------------------------------------------
    if config.snapshot_name is not None:
        main_print(
            f"Resuming from snapshot: {config.snapshot_name}: reconciling config(s)...", 
            acc=acc
        )
        # Experiment directory (parent of "models/")
        exp_dir = os.path.dirname(config.model_save_dir)

        # Path to the *original* config used in the first run
        legacy_yaml_path = os.path.join(exp_dir, 'config', 'config.yaml')

        # Decide whether to load the legacy YAML.  We do so **only** when the
        # user did *not* pass a different --config on the command line.
        cl_yaml_path = getattr(config_manager.clargs, 'config', None)
        # Load the legacy YAML only when the caller **did not pass** a
        # --config (CLI parameter omitted).  If a path was provided we
        # assume the user wants to use that file exactly as-is.
        load_legacy = (cl_yaml_path is None)

        if load_legacy and os.path.exists(legacy_yaml_path):
            with open(legacy_yaml_path, 'r') as f:
                loaded_yaml = yaml.safe_load(f)

            # Merge – CLI overrides have already been applied by ConfigManager
            for k, v in loaded_yaml.get('training', {}).items():
                if hasattr(config, k):
                    setattr(config, k, v)
            for k, v in loaded_yaml.get('model', {}).items():
                if hasattr(config.model_config, k):
                    setattr(config.model_config, k, v)
            for k, v in loaded_yaml.get('optimizer', {}).items():
                if hasattr(config.optimizer_config, k):
                    setattr(config.optimizer_config, k, v)
            for k, v in loaded_yaml.get('scheduler', {}).items():
                if hasattr(config.scheduler_config, k):
                    setattr(config.scheduler_config, k, v)

            # Keep the snapshot we are resuming from
            # (it could have been overwritten above)
            config.snapshot_name = Path(config_manager.clargs.snapshot_path).name

        # --- Paths that always point inside *exp_dir* ---
        config.train_logs_save_dir = os.path.join(exp_dir, 'logs')
        config.model_save_dir      = os.path.join(exp_dir, 'models')
        config.results_save_dir    = os.path.join(exp_dir, 'metrics')

        # Visibility prints: show which YAML sources are active and key values
        main_print("[RESUME] Config reconciliation sources:", acc=acc)
        if cl_yaml_path is not None:
            main_print(f"  - CLI --config provided: {cl_yaml_path}", acc=acc)
        else:
            main_print("  - Using legacy experiment/config.yaml from snapshot directory", acc=acc)
        main_print("[RESUME] Effective selections after reconciliation:", acc=acc)
        try:
            _mk = config.model_config.model_key
        except Exception:
            _mk = None
        try:
            _lr = config.optimizer_config.learn_rate
        except Exception:
            _lr = None
        try:
            _sched = config.scheduler_config.scheduler_key
        except Exception:
            _sched = None
        main_print(f"  - model.model_key: {_mk}", acc=acc)
        main_print(f"  - optimizer.learn_rate: {_lr}", acc=acc)
        main_print(f"  - scheduler.scheduler_key: {_sched}", acc=acc)

        # ------------------------------------------------------------------
        # Prepare a restart-config filename but delay the actual save until
        # later (guarded by `is_main_process`) so we write it only once.
        # ------------------------------------------------------------------
        config_dir = os.path.join(exp_dir, 'config')
        os.makedirs(config_dir, exist_ok=True)

        n = 1
        while True:
            restart_config_path = os.path.join(config_dir, f'config_restart_{n}.yaml')
            if not os.path.exists(restart_config_path):
                break
            n += 1

        # Remember where we will save the (possibly edited) config later
        config.config_save_path = restart_config_path
        main_print(f"\tDone.", acc=acc)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    # All processes load dataset
    # for now, this includes independently computing target rescaling stats,
    # applying them to targets, and subsetting targets if needed
    # TODO: compute stats on one process, and move these steps into a 
    # transform method of the dataset class (this requires awareness of which
    # split a data object is in though...unless all splits are transformed and
    # un-transformed)

    # Special case of macaque dataset and kfolds experiments: construct per-fold 
    # graphs on the fly
    
    is_macaque_kfolds = False
    if 'macaque' in config.dataset_config.dataset.lower():
        if 'kfold' in config.experiment_type:
            is_macaque_kfolds = True
        else:
            raise NotImplementedError(
                f"Macaque data set handling not yet implemented for "
                f"'{config.experiment_type}' experiment type. Exiting."
            )

    if not is_macaque_kfolds:
        dataset = load_dataset(
            config, 
            model_key=config.model_config.model_key
        )
        if config.model_config.model_key.lower() == 'tnn':
            # Cache sheaf Laplacians per graph; enforce single-graph batches
            dataset = attach_tnn_operators(
                dataset=dataset,
                vector_feat_key=config.dataset_config.vector_feat_key,
            )
            config.batch_size = 1
            config.valid_batch_size = 1
        # print(f"len(dataset) after load_dataset: {len(dataset)}")
        if dataset is None:
            raise RuntimeError(
                "Dataset failed to load! Check your config file and data filepaths."
            )
    else:
        # Load macaque data in main_training.py
        print(
            "[INFO] Macaque data set detected in main_training.py; "
            "bypassing usual call to 'load_dataset'."
        )
        pass

    # ------------------------------------------------------------------
    # Single train/valid/test split experiment (no cross-validation)
    # ------------------------------------------------------------------
    if config.experiment_type == 'tvt':  # 'train/valid/[test]' splits
        if acc.is_main_process:
            # Default: random proportional splits
            splits_dict = get_random_splits(
                n=len(dataset),
                seed=config.dataset_config.split_seed,
                train_prop=config.dataset_config.train_prop,
                valid_prop=config.dataset_config.valid_prop,
            )

            splits_list = [splits_dict]
        else:
            splits_list = [None]

        broadcast_object_list(splits_list)
        splits_dict = splits_list[0]
        dataloader_dict, config = create_dataloaders(
            dataset, 
            splits_dict, 
            config
        )
        main_print(f"After create_dataloaders:\n\tconfig.dataset_config.target_include_indices: {config.dataset_config.target_include_indices}", acc=acc)
        if config.dataset_config.target_preprocessing_type is not None:
            main_print(f"\tconfig.dataset_config.target_preproc_stats: {config.dataset_config.target_preproc_stats}", acc=acc)
        
        # Sanity check: print first train batch object (main process only)
        print_first_batch_summary(
            dataloader_dict=dataloader_dict, 
            set_key='train', 
            acc=acc, 
            config=config
        )
        
        # Print split and batch sizes (main process only)
        if acc.is_main_process:
            split_sizes = {k: len(v.dataset) for k, v in dataloader_dict.items()}
            acc.print(
                f"Dataset split sizes:" 
                f" train: {split_sizes.get('train', 0)},"
                f" valid: {split_sizes.get('valid', 0)},"
                f" test: {split_sizes.get('test', 0)}"
            )
            acc.print(
                f"Batch sizes:"
                f" train: {dataloader_dict.get('train', {}).batch_size},"
                f" valid: {dataloader_dict.get('valid', {}).batch_size},"
                f" test: {dataloader_dict.get('test', {}).batch_size}"
            )

            # Batching sanity-checks across all splits
            _batching_sanity_check(split_sizes, dataloader_dict, acc)
        
        # Save config after all processing is complete (main process only)
        if config.config_save_path and acc.is_main_process:
            config_manager.save_config(config.config_save_path)
            # Also save readable copies of the original model/experiment YAML files
            # alongside the aggregate config.yaml generated by ConfigManager.
            try:
                config_dir = os.path.dirname(config.config_save_path)

                # Model-level YAML (e.g., 'vdw_supcon.yaml')
                if hasattr(config_manager, 'model_yaml_path') \
                and (config_manager.model_yaml_path is not None):
                    model_yaml_src = str(config_manager.model_yaml_path)
                    if os.path.isfile(model_yaml_src):
                        model_yaml_dst = os.path.join(
                            config_dir,
                            os.path.basename(model_yaml_src),
                        )
                        if os.path.abspath(model_yaml_src) != os.path.abspath(model_yaml_dst):
                            shutil.copy2(model_yaml_src, model_yaml_dst)

                # Experiment-level YAML (e.g., 'experiment.yaml') if available
                if hasattr(config_manager, 'experiment_yaml_path') \
                and (config_manager.experiment_yaml_path is not None):
                    exp_yaml_src = str(config_manager.experiment_yaml_path)
                    if os.path.isfile(exp_yaml_src):
                        exp_yaml_dst = os.path.join(
                            config_dir,
                            os.path.basename(exp_yaml_src),
                        )
                        if os.path.abspath(exp_yaml_src) != os.path.abspath(exp_yaml_dst):
                            shutil.copy2(exp_yaml_src, exp_yaml_dst)
            except Exception as e:
                main_print(f"[WARNING] Could not copy original YAML config files: {e}", acc=acc)
        
        # Run one fold (whole experiment)
        _ = run_one_fold(
            config, 
            dataloader_dict, 
            save_results=True,
            accelerator=acc
        )
        
    # ------------------------------------------------------------------
    # k-fold cross-validation
    # ------------------------------------------------------------------
    elif 'kfold' in config.experiment_type:  # k-fold cross-validation

        # Stratified k-fold cross-validation not yet implemented
        if 'stratified' in config.experiment_type:
            raise NotImplementedError(
                "Stratified k-fold cross-validation not yet implemented. Exiting."
            )

        if not is_macaque_kfolds:
            # Create generic k-fold split indexes on main process, broadcast to all
            if acc.is_main_process:
                kfold_splits = get_kfold_splits(
                    seed=config.dataset_config.split_seed,
                    k=config.dataset_config.k_folds,
                    n=len(dataset)
                )
                # Choose validation fold indices cyclically from non-test folds
                val_folds_is = list(range(1, config.dataset_config.k_folds)) + [0]
                kfold_splits_list = [kfold_splits, val_folds_is]
            else:
                kfold_splits_list = [None, None]
            broadcast_object_list(kfold_splits_list)
            kfold_splits = kfold_splits_list[0]
            val_folds_is = kfold_splits_list[1]
    
        is_supcon2 = config.model_config.model_key == 'vdw_supcon_2'
        is_tnn = config.model_config.model_key == 'tnn'
        is_supcon_like = is_supcon2 or is_tnn
        processed_dataset_override = getattr(
            config.dataset_config,
            'processed_dataset_path',
            None,
        )
        processed_dataset_override_base_path: Optional[Path] = None
        if processed_dataset_override:
            candidate = Path(processed_dataset_override).expanduser()
            if not candidate.is_absolute():
                if getattr(config, 'root_dir', None):
                    candidate = Path(config.root_dir).expanduser() / candidate
                elif getattr(config.dataset_config, 'data_dir', None):
                    candidate = Path(config.dataset_config.data_dir).expanduser() / candidate
                else:
                    candidate = Path.cwd() / candidate
            processed_dataset_override_base_path = candidate.resolve()
        supcon2_context: Dict[int, Dict[str, Any]] = {}

        # Loop through folds
        for fold_i in range(config.dataset_config.k_folds):
            # Process k-fold splits into train/valid/test sets (all processes, deterministic)
            if not is_macaque_kfolds:
                split_dict = process_kfold_splits(
                    kfold_splits,
                    k=config.dataset_config.k_folds,
                    fold_idx=fold_i,
                    include_test_set=True,
                    valid_fold_idx=val_folds_is[fold_i]
                )
            
            # Create per-fold output directories under a common parent
            if acc.is_main_process and config.snapshot_name is None:
                parent_dir = dirs['exp_dir']
                fold_root = os.path.join(parent_dir, f"fold_{fold_i}")
                fold_dirs = {
                    'metrics': os.path.join(fold_root, 'metrics'),
                    'models': os.path.join(fold_root, 'models'),
                    'logs': os.path.join(fold_root, 'logs'),
                    'config': os.path.join(fold_root, 'config'),
                    'embeddings': os.path.join(fold_root, 'embeddings'),
                }
                for _p in fold_dirs.values():
                    ensure_dir_exists(_p, raise_exception=True)
                # Prepare a unique config save path per fold
                fold_config_path = os.path.join(fold_dirs['config'], 'config.yaml')
                fold_dirs_map = {
                    'results_save_dir': fold_dirs['metrics'],
                    'model_save_dir': fold_dirs['models'],
                    'train_logs_save_dir': fold_dirs['logs'],
                    'config_save_path': fold_config_path,
                }
            else:
                fold_dirs_map = None

            # Broadcast fold directories so all ranks agree
            shared_fold_dirs = [fold_dirs_map]
            broadcast_object_list(shared_fold_dirs)
            fold_dirs_map = shared_fold_dirs[0]

            # Update config paths for this fold
            config.results_save_dir = fold_dirs_map['results_save_dir']
            config.model_save_dir = fold_dirs_map['model_save_dir']
            config.train_logs_save_dir = fold_dirs_map['train_logs_save_dir']
            config.config_save_path = fold_dirs_map['config_save_path']

            # Create dataloaders for fold
            if is_macaque_kfolds:
                time_start = time.time()
                
                # Check scatter_path_graphs flag to decide which pipeline
                scatter_path_graphs = getattr(config.model_config, 'scatter_path_graphs', False)
                day_selection = config.dataset_config.macaque_day_index
                
                if is_supcon_like:
                    from data_processing.macaque_reaching import (  # noqa: E402
                        # MarbleFoldData,
                        macaque_prepare_marble_fold_data,
                    )
                    day_offset_seed = 0
                    try:
                        if isinstance(day_selection, (list, tuple)) and len(day_selection) > 0:
                            day_offset_seed = int(day_selection[0])
                        else:
                            day_offset_seed = int(day_selection)
                    except Exception:
                        day_offset_seed = 0
                    adjusted_split_seed = int(config.dataset_config.split_seed) - day_offset_seed
                    fold_data = macaque_prepare_marble_fold_data(
                        data_root=config.dataset_config.data_dir,
                        day_index=day_selection,
                        k_folds=int(config.dataset_config.k_folds),
                        fold_i=int(fold_i),
                        seed=adjusted_split_seed,
                        include_lever_velocity=False,
                        k_neighbors=int(config.model_config.k_neighbors),
                        cknn_dist_metric=getattr(config.model_config, "cknn_dist_metric", "euclidean"),
                    )
                    scattering_k = getattr(config.model_config, "scattering_k", None)
                    if (
                        scattering_k is not None
                        and scattering_k > int(config.model_config.k_neighbors)
                    ):
                        raise ValueError(
                            f"model.scattering_k ({scattering_k}) must be <= k_neighbors ({config.model_config.k_neighbors})."
                        )
                    learnable_p_flag = bool(getattr(config.model_config, 'learnable_P', False))
                    use_processed_override = processed_dataset_override_base_path is not None
                    should_save_processed = (
                        bool(getattr(config.dataset_config, 'save_processed_dataset', False))
                        and not use_processed_override
                    )
                    if use_processed_override:
                        override_path = resolve_processed_dataset_override(
                            processed_dataset_override_base_path,
                            fold_i,
                        )
                        if not override_path.exists():
                            raise FileNotFoundError(
                                f"Processed dataset file '{override_path}' was not found."
                            )
                        train_data = load_processed_train_graph(override_path)
                        main_print(
                            f"[INFO] Loaded processed dataset from {override_path}",
                            acc=acc,
                            config=config,
                        )
                    else:
                        import data_processing.macaque_reaching as mr  # noqa: E402

                        def _resolve_min_nbrs(cfg_val):
                            if cfg_val is True:
                                # Prefer resolved intrinsic dim from preprocessing globals
                                if isinstance(getattr(mr, "MANIFOLD_INTRINSIC_DIM", None), int):
                                    return int(mr.MANIFOLD_INTRINSIC_DIM)
                                if isinstance(getattr(mr, "GLOBAL_PCA_REDUCED_DIM", None), int):
                                    return int(mr.GLOBAL_PCA_REDUCED_DIM)
                                # Fallback: use fold_data dimensionality if available
                                if hasattr(fold_data, "state_dim"):
                                    try:
                                        return int(fold_data.state_dim)
                                    except Exception:
                                        pass
                                if hasattr(fold_data, "train_positions"):
                                    try:
                                        return int(fold_data.train_positions.shape[1])
                                    except Exception:
                                        pass
                                return None
                            if isinstance(cfg_val, int):
                                return cfg_val
                            return None

                        min_neighbors = _resolve_min_nbrs(
                            getattr(config.model_config, "cknn_min_nbrs", None)
                        )
                        dist_metric = str(
                            getattr(config.model_config, "cknn_dist_metric", "euclidean")
                        ).lower()

                        train_data = build_train_graph(
                            fold_data,
                            n_neighbors=int(config.model_config.k_neighbors),
                            delta=float(config.model_config.cknn_delta),
                            compute_q=True,
                            scattering_n_neighbors=(
                                int(scattering_k) if scattering_k is not None else None
                            ),
                            learnable_p=learnable_p_flag,
                            min_neighbors=min_neighbors,
                            dist_metric=dist_metric,
                            eval_mode=str(getattr(config.model_config, "eval_mode", "weighted_average")),
                        )
                        if should_save_processed:
                            cache_path = get_processed_dataset_path(
                                data_dir=config.dataset_config.data_dir,
                                macaque_day_index=day_selection,
                                learnable_p=learnable_p_flag,
                                fold_idx=fold_i,
                            )
                            if acc.is_main_process:
                                save_processed_train_graph(
                                    train_data,
                                    cache_path,
                                    log_fn=lambda msg: main_print(msg, acc=acc, config=config),
                                )
                            acc.wait_for_everyone()
                    if not hasattr(train_data, 'v_vel'):
                        train_data.v_vel = train_data.x.clone()
                    if not hasattr(train_data, 'y'):
                        train_data.y = train_data.condition_idx.clone()
                    split_graphs = {'train': [train_data]}
                    supcon2_context[fold_i] = {
                        'fold_data': fold_data,
                        'train_data': train_data,
                        'expected_nodes': fold_data.nodes_per_trial,
                        'embeddings_dir': Path(fold_dirs['embeddings']),
                        'metrics_dir': Path(fold_dirs['metrics']),
                        'models_dir': Path(fold_dirs['models']),
                        'fold_idx': fold_i,
                        'svm_kernel': getattr(config.model_config, 'svm_kernel', 'rbf'),
                        'svm_c': float(getattr(config.model_config, 'svm_c', 1.0)),
                        'svm_gamma': getattr(config.model_config, 'svm_gamma', 'scale'),
                    }
                elif scatter_path_graphs:
                    # Legacy: Build macaque path graphs (with Q) per split for this fold (single day)
                    # Parse task_level: 'graph' for final position/velocity targets
                    task_level = 'graph' \
                        if ('graph' in config.dataset_config.task.lower() \
                            and 'final' in config.dataset_config.target_key.lower()) \
                        else 'node'
                    split_graphs = macaque_prepare_kth_fold(
                        data_root=config.dataset_config.data_dir,
                        day_index=day_selection,
                        k_folds=int(config.dataset_config.k_folds),
                        fold_i=int(fold_i),
                        seed=int(config.dataset_config.split_seed),
                        mode=config.experiment_inference_type,
                        include_lever_velocity=True,
                        n_neighbors=int(config.model_config.k_neighbors),
                        cknn_delta=float(config.model_config.cknn_delta),
                        metric='euclidean',
                        include_self=False,
                        reweight_with_median_kernel=True,
                        apply_savgol_filter_before_pca=True,
                        target_key=config.dataset_config.target_key,
                        task_level=task_level,
                    )
                    # Patch: if present, remove 'final_' from target_key in config AFTER data prep
                    # done by macaque_prepare_kth_fold (the prefix signals to use only the final
                    # timepoint target as a graph-level target)
                    if 'final_' in config.dataset_config.target_key:
                        config.dataset_config.target_key = config.dataset_config.target_key \
                            .replace('final_', '')
                else:
                    # New: Build single spatial graph with Q for entire day (with masks)
                    spatial_graph = macaque_prepare_spatial_graph_only_for_fold(
                        data_root=config.dataset_config.data_dir,
                        day_index=day_selection,
                        k_folds=int(config.dataset_config.k_folds),
                        fold_i=int(fold_i),
                        seed=int(config.dataset_config.split_seed),
                        mode=config.experiment_inference_type,
                        include_lever_velocity=True,
                        n_neighbors=int(config.model_config.k_neighbors),
                        cknn_delta=float(config.model_config.cknn_delta),
                        metric='euclidean',
                        include_self=False,
                        reweight_with_median_kernel=True,
                        apply_savgol_filter_before_pca=True,
                        target_key=config.dataset_config.target_key,
                    )
                    # Wrap in dict with only 'train' key for compatibility
                    # (the single graph has train/valid/test masks)
                    split_graphs = {'train': [spatial_graph]}
                    
                    # Override batch size to 1 for spatial graph mode
                    config.batch_size = 1
                    config.valid_batch_size = 1
                    
                    # Remove 'final_' from target_key if present
                    if 'final_' in config.dataset_config.target_key:
                        config.dataset_config.target_key = config.dataset_config.target_key \
                            .replace('final_', '')
                
                # Create dataloaders
                dataloader_dict, config = create_dataloaders(
                    split_graphs,
                    None,
                    config,
                )

                prep_time = time.time() - time_start
                print(
                    f"Dataset preparation time (fold {fold_i}): "
                    f"{prep_time:.1f} sec"
                )
            else:  # k-folds, not macaque dataset
                dataloader_dict, config = create_dataloaders(
                    dataset, 
                    split_dict, 
                    config
                )
            
            # Sanity check: print first TRAIN PyG Batch object (main process only)
            print_first_batch_summary(
                dataloader_dict=dataloader_dict, 
                set_key='train', 
                acc=acc, 
                config=config
            )
            
            # Print split sizes (main process only)
            if acc.is_main_process:
                # try:
                split_sizes = {
                    k: len(v.dataset) for k, v in dataloader_dict.items()
                }
                acc.print(
                    f"[Dataset split sizes]" \
                    f" train: {split_sizes.get('train', 0)},"\
                    f" valid: {split_sizes.get('valid', split_sizes.get('val', 0))},"\
                    f" test: {split_sizes.get('test', 0)}"\
                    f" (fold {fold_i})"
                )

                # Batching sanity-checks across all splits (fold-aware)
                _batching_sanity_check(split_sizes, dataloader_dict, acc, fold_i)
            
            # Save config for this fold (main process only)
            if config.config_save_path and acc.is_main_process:
                config_manager.save_config(config.config_save_path)
                # Also save readable copies of the original model/experiment YAML files
                # alongside the aggregate config.yaml generated by ConfigManager.
                try:
                    config_dir = os.path.dirname(config.config_save_path)

                    # Model-level YAML (e.g., 'vdw_supcon.yaml')
                    if hasattr(config_manager, 'model_yaml_path') and (config_manager.model_yaml_path is not None):
                        model_yaml_src = str(config_manager.model_yaml_path)
                        if os.path.isfile(model_yaml_src):
                            model_yaml_dst = os.path.join(
                                config_dir,
                                os.path.basename(model_yaml_src),
                            )
                            if os.path.abspath(model_yaml_src) != os.path.abspath(model_yaml_dst):
                                shutil.copy2(model_yaml_src, model_yaml_dst)

                    # Experiment-level YAML (e.g., 'experiment.yaml') if available
                    if hasattr(config_manager, 'experiment_yaml_path') \
                    and (config_manager.experiment_yaml_path is not None):
                        exp_yaml_src = str(config_manager.experiment_yaml_path)
                        if os.path.isfile(exp_yaml_src):
                            exp_yaml_dst = os.path.join(
                                config_dir,
                                os.path.basename(exp_yaml_src),
                            )
                            if os.path.abspath(exp_yaml_src) != os.path.abspath(exp_yaml_dst):
                                shutil.copy2(exp_yaml_src, exp_yaml_dst)
                except Exception as e:
                    main_print(f"[WARNING] Could not copy original YAML config files: {e}", acc=acc)
            
            post_hook = None
            if is_supcon_like:
                ctx = supcon2_context[fold_i]

                def _make_supcon2_hook(context, cfg):
                    def _hook(model_obj, accelerator_obj):
                        if not accelerator_obj.is_main_process:
                            return

                        embeddings_np, svm_stats, split_payloads = evaluate_supcon2_with_svm(
                            model=model_obj,
                            train_data=context['train_data'],
                            fold_data=context['fold_data'],
                            expected_nodes=context['expected_nodes'],
                            svm_kernel=context['svm_kernel'],
                            svm_C=context['svm_c'],
                            svm_gamma=context['svm_gamma'],
                            include_test=True,
                            return_split_features=True,
                        )

                        msg = (
                            f"[SupCon2] Fold {context['fold_idx']}: "
                            f"SVM train={svm_stats['train_accuracy']:.4f} "
                            f"valid={svm_stats['val_accuracy']:.4f} "
                            f"test={svm_stats['test_accuracy']:.4f}"
                        )
                        main_print(msg, acc=accelerator_obj, config=cfg)
                        log_to_train_print(msg, acc=accelerator_obj, config=cfg)

                        try:
                            parameter_count = count_parameters(model_obj)
                        except Exception as exc:
                            parameter_count = None
                            main_print(
                                f"[WARNING] Failed to count parameters: {exc}",
                                acc=accelerator_obj,
                                config=cfg,
                            )

                        embeddings_dir = context['embeddings_dir']
                        metrics_dir = context['metrics_dir']
                        models_dir = context['models_dir']

                        for split_name, (feats, labels, trial_ids) in split_payloads.items():
                            save_embeddings(embeddings_dir, split_name, feats, labels, trial_ids)

                        ensure_dir_exists(models_dir, raise_exception=True)
                        with open(models_dir / "svm.pkl", "wb") as f:
                            pickle.dump(svm_stats["svm"], f)
                        smart_pickle(
                            str(metrics_dir / "test_probabilities.pkl"),
                            svm_stats["test_prob_records"],
                            overwrite=True,
                        )

                        numeric_svm_stats = {
                            k: float(v)
                            for k, v in svm_stats.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)
                        }
                        results_payload = {
                            "fold": context.get("fold_idx"),
                            "parameter_count": int(parameter_count) if parameter_count is not None else None,
                            **numeric_svm_stats,
                        }
                        merge_results_artifacts(
                            metrics_dir=metrics_dir,
                            new_metrics=results_payload,
                            backfill_if_missing=context if isinstance(context, dict) else None,
                            log_fn=accelerator_obj.print,
                        )

                    return _hook

                post_hook = _make_supcon2_hook(ctx, config)

            # Train model on fold and save results
            if is_supcon_like:
                config._current_supcon2_context = supcon2_context.get(fold_i)
            try:
                _ = run_one_fold(
                    config, 
                    dataloader_dict, 
                    fold_i,
                    save_results=True,
                    accelerator=acc,
                    post_fold_hook=post_hook,
                )
            finally:
                if hasattr(config, "_current_supcon2_context"):
                    delattr(config, "_current_supcon2_context")
            print("\n\n")
            print("="*64)

        if acc.is_main_process and getattr(config.dataset_config, "k_folds", 1) > 1:
            parent_metrics_dir = Path(parent_dir) / "metrics"
            compute_kfold_results(
                parent_metrics_dir=parent_metrics_dir,
                timing_keys=("mean_train_time", "mean_infer_time"),
                weight_key="best_epoch",
            )
    else:
        raise ValueError(
            f"Experiment type '{config.experiment_type}' not supported"
        )
        
    # ------------------------------------------------------------------
    # EXPERIMENT COMPLETE (tvt or k-folds experiments)
    # ------------------------------------------------------------------
    # Print experiment summary (main process only)
    if acc.is_main_process:
        total_experiment_elapsed = time.time() - experiment_start_time
        hours = int(total_experiment_elapsed // 3600)
        minutes = int((total_experiment_elapsed % 3600) // 60)
        seconds = total_experiment_elapsed % 60
        
        main_print(f"\n{'=' * 80}")
        main_print(f"EXPERIMENT COMPLETE")
        main_print(f"Total experiment time: {hours:02d}h {minutes:02d}m {seconds:05.2f}s")
        # main_print(f"{'=' * 80}\n")

    # End Accelerator tracking (e.g., wandb) after all work
    if getattr(config, 'use_wandb_logging', False):
        acc.end_training()

    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
if __name__ == '__main__':
    clargs = get_clargs()
    # torch.autograd.set_detect_anomaly(True)
    main(clargs)

