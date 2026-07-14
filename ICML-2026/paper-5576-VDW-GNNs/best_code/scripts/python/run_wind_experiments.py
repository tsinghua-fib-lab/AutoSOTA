#!/usr/bin/env python3
"""
Run wind reconstructions across sample_n and mask_prop combinations with replications.

Outputs:
- /root/experiments/model/<combo>/results_<rep>.yaml for each replication
- /root/experiments/model/<combo>/reps_results.yaml aggregated over replications

Example usage:
python3 run_wind_experiments.py \
    --config=../config/yaml_files/wind/vdw.yaml \
    --root_dir=../ \
    --replications=5 \
    --knn_k=3 \
    --local_pca_k=10 \
    --sample_n=400 \
    --mask_prop=0.1
"""

from __future__ import annotations

import os
import argparse
import copy
import time
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Optional

import numpy as np
import torch
from accelerate import Accelerator
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch

import sys
DIR_PARENTS = Path(__file__).resolve().parents
PROJECT_ROOT, CODE_DIR = DIR_PARENTS[3], DIR_PARENTS[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from config.config_manager import ConfigManager
from training.prep_model import (
    prepare_vdw_model,
    prepare_comparison_model,
    _load_pretrained_weights,
)
from training.prep_optimizer import prepare_optimizer, prepare_scheduler
import training.train as train
from models.nn_utilities import count_parameters
from models import nn_utilities as nnu
from data_processing.wind import (
    WIND_ROT_MASK_PROP,
    WIND_ROT_SAMPLE_N,
    apply_random_2d_rotation,
    apply_random_3d_rotation,
    create_dataset,
    create_wind_rot_datasets,
    wind_rot_seed_rng,
)
from data_processing import process_pyg_data as pproc
import results_utils
from os_utilities import get_unique_path_with_suffix
from tests.wind_equivariance_helpers import generate_equivariance_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wind reconstruction experiments with replications.")
    parser.add_argument("--config", type=Path, default=Path("config/yaml_files/wind/vdw.yaml"))
    parser.add_argument("--root_dir", type=Path, default=None)
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument(
        "--local_pca_k",
        type=int,
        default=10,
        help="Neighbor count for local PCA O-frame construction (default: 10).",
    )
    parser.add_argument(
        "--sample_n", 
        type=str, 
        default=None, 
        help="Comma-separated ints, e.g., 100,200,300"
    )
    parser.add_argument(
        "--mask_prop", 
        type=str, 
        default=None, 
        help="Comma-separated floats, e.g., 0.5,0.3,0.1"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset key override (e.g., wind or wind_rot). Defaults to config.",
    )
    parser.add_argument("--vector_feat_key", type=str, default=None)
    parser.add_argument(
        "--rotation_seed", 
        type=int, 
        default=None, 
        help="Seed for rotation equivariance eval."
    )
    parser.add_argument(
        "--do_rotation_eval",
        action="store_true",
        help="If set, run post-training rotation evaluation; disabled by default.",
    )
    parser.add_argument(
        "--reconstruct_Q_rot",
        action="store_true",
        help=(
            "If set, reconstruct Q on the rotated geometry (legacy behavior). "
            "If omitted (default), rotate the precomputed Q via conjugation: "
            "Q_rot = (I⊗R) Q (I⊗R)^T."
        ),
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Folder under experiments/wind for all combos (defaults to model_key).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, run debug mode, e.g., print local PCA singular value ratio stats.",
    )
    args = parser.parse_args()
    return args


def rotate_sparse_block_operator(
    Q: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate a block-sparse vector diffusion operator by conjugation:

        Q_rot = (I⊗R) Q (I⊗R)^T

    When Q is block-structured with d×d blocks (one per node-pair) and R is a d×d
    rotation matrix, this is equivalent to rotating each block:

        (Q_rot)_{ij} = R (Q_{ij}) R^T

    Assumes Q stores full d×d blocks in COO format (as produced by process_pyg_data).
    """
    if not (isinstance(Q, torch.Tensor) and Q.is_sparse):
        raise ValueError("rotate_sparse_block_operator expects a sparse torch.Tensor Q.")
    if not (isinstance(R, torch.Tensor) and (R.ndim == 2) and (R.shape[0] == R.shape[1])):
        raise ValueError("rotate_sparse_block_operator expects a square 2D torch.Tensor R.")

    d = int(R.shape[0])
    if Q.shape[0] % d != 0 or Q.shape[1] % d != 0:
        raise ValueError("Q shape must be divisible by d for block rotation.")
    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square.")

    Qc = Q.coalesce()
    idx = Qc._indices()
    vals = Qc._values()
    nnz = int(vals.numel())
    if nnz == 0:
        return Qc

    N = int(Q.shape[0] // d)
    # Block indices and within-block indices
    block_i = torch.div(idx[0], d, rounding_mode="floor")
    block_j = torch.div(idx[1], d, rounding_mode="floor")
    in_i = idx[0] - block_i * d
    in_j = idx[1] - block_j * d
    within_id = in_i * d + in_j  # row-major
    block_id = block_i * N + block_j

    # Sort to get contiguous full blocks for reshape
    combined = block_id * (d * d) + within_id
    perm = torch.argsort(combined)
    vals_sorted = vals[perm]

    if nnz % (d * d) != 0:
        raise ValueError("Q nnz is not divisible by d^2; expected full d×d blocks.")
    num_blocks = nnz // (d * d)
    blocks = vals_sorted.view(num_blocks, d, d)

    # Batched block rotation: R @ block @ R^T
    R = R.to(device=blocks.device, dtype=blocks.dtype)
    Rt = R.transpose(0, 1)
    blocks_rot = torch.matmul(torch.matmul(R, blocks), Rt)
    vals_rot_sorted = blocks_rot.reshape(-1)

    # Scatter back to original ordering
    vals_rot = torch.empty_like(vals)
    vals_rot[perm] = vals_rot_sorted

    return torch.sparse_coo_tensor(idx, vals_rot, Q.shape).coalesce()


def resolve_config_path(config_path: Path, dataset_key: str) -> Path:
    """
    Resolve a config path, defaulting to config/yaml_files/<dataset>/ when only
    a filename is provided.
    """
    # If absolute and exists, return directly
    if config_path.is_absolute() and config_path.exists():
        return config_path

    candidates = []

    # Filename only: prefer config/yaml_files/<dataset>/<name>
    if len(config_path.parts) == 1:
        candidates.append(CODE_DIR / "config" / "yaml_files" / dataset_key / config_path.name)
    else:
        # If already under config/, anchor to PROJECT_ROOT
        if config_path.parts[0] == "config":
            candidates.append((PROJECT_ROOT / config_path).resolve())
        # Otherwise, try relative to CODE_DIR
        candidates.append((CODE_DIR / config_path).resolve())

    # Fallback: if dataset-specific path missing, try wind as a last resort
    if len(config_path.parts) == 1:
        candidates.append(CODE_DIR / "config" / "yaml_files" / "wind" / config_path.name)

    for cand in candidates:
        if cand.exists():
            return cand

    # If nothing found, return the first candidate to surface a clear error
    return candidates[0]


def make_clargs(config_path: Path, args: argparse.Namespace) -> SimpleNamespace:
    """
    Minimal CLI namespace covering attributes ConfigManager accesses directly.
    """
    return SimpleNamespace(
        config=str(config_path),
        root_dir=str(args.root_dir) if args.root_dir is not None else None,
        dataset=args.dataset,
        processed_dataset_path=None,
        model_key=None,
        model_mode=None,
        wavelet_scales_type=None,
        J=None,
        use_dirac_nodes=None,
        invariant_pred=None,
        scalar_feat_key=None,
        vector_feat_key=args.vector_feat_key,
        validate_every_n_epochs=None,
        burn_in=None,
        patience=None,
        experiment_id=None,
        batch_size=None,
        verbosity=None,
        device=None,
        save_best_model_state=None,
        save_final_model_state=None,
        dataloader_split_batches=True,
        subsample_n=None,
        subsample_seed=None,
        ablate_vector_track=False,
        ablate_scalar_track=False,
        task_key=None,
        target_key=None,
        target_dim=None,
        k_folds=None,
        experiment_type="tvt",
        experiment_dir=None,
        use_wandb_logging=False,
        wandb_offline=False,
        wandb_log_freq=None,
        checkpoint_every=None,
        snapshot_path=None,
        snapshot_name=None,
        dataloader_num_workers=None,
        precompute_scatter_tensors=None,
        save_processed_dataset=None,
        processed_dataset_path_override=None,
        warmup_epochs=None,
        n_epochs=None,
        scheduler_burnin=None,
        cknn_dist_metric=None,
        cknn_delta=None,
        k_neighbors=None,
        model_save_dir=None,
        results_save_dir=None,
        train_logs_save_dir=None,
        config_save_path=None,
        slurm=False,
        learn_rate=None,
    )


def prep_config(args: argparse.Namespace) -> Tuple[ConfigManager, object]:
    """
    Build a config via ConfigManager with required attributes set for wind runs.
    """
    dataset_key_for_path = args.dataset if args.dataset is not None else "wind"
    resolved_config = resolve_config_path(Path(args.config), dataset_key_for_path)
    clargs = make_clargs(resolved_config, args)
    config_manager = ConfigManager(clargs)
    config = config_manager.config

    # Override or enforce key selections for wind experiments
    if args.root_dir is not None:
        config.root_dir = str(args.root_dir)
    if args.debug:
        config.verbosity = 1
    dataset_key = args.dataset if args.dataset is not None else getattr(config.dataset_config, "dataset", "wind")
    config.dataset_config.dataset = dataset_key
    if args.vector_feat_key is not None:
        config.dataset_config.vector_feat_key = args.vector_feat_key
    if getattr(config.dataset_config, "vector_feat_dim", None) is None:
        config.dataset_config.vector_feat_dim = 2
    # Map legacy/short model key to the wind-optimized VDW layer stack.
    if getattr(config.model_config, "model_key", None) == "vdw":
        config.model_config.model_key = "vdw_layer"
    # Make sure multi-fold mechanisms don't run
    config.dataset_config.k_folds = 1
    config.experiment_type = "tvt"

    print(
        "[CONFIG] ablate_scalar_track="
        f"{getattr(config, 'ablate_scalar_track')}, "
        "[CONFIG] ablate_vector_track="
        f"{getattr(config, 'ablate_vector_track')}, "
        f"model_yaml={config_manager.model_yaml_path}, "
        f"experiment_yaml={config_manager.experiment_yaml_path}"
    )

    return config_manager, config


def parse_list(arg: str, cast_fn):
    return [cast_fn(x.strip()) for x in arg.split(",") if x.strip()]


def format_combo(sample_n: int, mask_prop: float) -> str:
    pct = int(round(mask_prop * 100))
    return f"n{sample_n}p{pct:02d}"


def seed_rng(base_seed: int, combo_idx: int, rep_idx: int) -> np.random.RandomState:
    return np.random.RandomState(base_seed + combo_idx * 1000 + rep_idx)


def is_wind_rot_dataset(dataset_key: str | None) -> bool:
    """
    Helper to identify the wind_rot dataset selection.
    """
    if dataset_key is None:
        return False
    return dataset_key.lower() == "wind_rot"


def attach_pq(
    data: torch.utils.data.Dataset,
    vector_feat_key: str = "v",
    geom_feat_key: Optional[str] = None,
    local_pca_k: Optional[int] = None,
    equivar_or_invar: str = "equivariant",
    debug: bool = False,
):
    """
    Attach P and Q operators using the existing process_pyg_data pipeline.
    Keeps the data as an VDWData object with operator_keys=('P', 'Q').
    """
    # Remove pre-existing operator_keys to avoid duplicate kwargs when constructing VDWData
    if hasattr(data, "operator_keys"):
        try:
            delattr(data, "operator_keys")
        except Exception as e:
            print(f"Error deleting operator_keys from data: {e}")
    data = pproc.process_pyg_data(
        data=data,
        vector_feat_key=vector_feat_key,
        geom_feat_key=geom_feat_key,
        graph_construction=None,
        return_data_object=True,
        use_mean_recentering=False,
        row_normalize=True,
        local_pca_k=local_pca_k,
        equivar_or_invar=equivar_or_invar,
        debug=debug,
    )
    _check_for_nans_in_data(data, vector_feat_key=vector_feat_key)
    return data


def attach_q_only(
    data: torch.utils.data.Dataset,
    vector_feat_key: str = "v",
    geom_feat_key: Optional[str] = None,
    local_pca_k: Optional[int] = None,
    equivar_or_invar: str = "equivariant",
    debug: bool = False,
):
    """
    Compute PQ via process_pyg_data, then drop P to keep only Q as requested.
    """
    data = attach_pq(
        data,
        vector_feat_key=vector_feat_key,
        geom_feat_key=geom_feat_key,
        local_pca_k=local_pca_k,
        equivar_or_invar=equivar_or_invar,
        debug=debug,
    )
    if hasattr(data, "P"):
        delattr(data, "P")
    return data


def _check_for_nans_in_data(
    data: torch.utils.data.Dataset,
    *,
    vector_feat_key: str,
) -> None:
    if hasattr(data, vector_feat_key):
        nnu.raise_if_nonfinite_tensor(
            data[vector_feat_key],
            name=f"attach_pq: data[{vector_feat_key}]",
        )
    if hasattr(data, "y"):
        nnu.raise_if_nonfinite_tensor(
            data.y,
            name="attach_pq: data.y",
        )
    if hasattr(data, "Q"):
        Q = data.Q
        if isinstance(Q, torch.Tensor) and Q.is_sparse:
            nnu.raise_if_nonfinite_tensor(
                Q._values(),
                name="attach_pq: data.Q values",
            )
        elif isinstance(Q, torch.Tensor):
            nnu.raise_if_nonfinite_tensor(
                Q,
                name="attach_pq: data.Q",
            )


def attach_operators(
    data: torch.utils.data.Dataset,
    vector_feat_key: str,
    geom_feat_key: Optional[str],
    ablate_scalar_track: bool,
    local_pca_k: Optional[int] = None,
    equivar_or_invar: str = "equivariant",
    debug: bool = False,
):
    """
    Attach Q (and optionally P) based on scalar-track ablation.
    """
    if ablate_scalar_track:
        return attach_q_only(
            data,
            vector_feat_key=vector_feat_key,
            geom_feat_key=geom_feat_key,
            local_pca_k=local_pca_k,
            equivar_or_invar=equivar_or_invar,
            debug=debug,
        )
    return attach_pq(
        data,
        vector_feat_key=vector_feat_key,
        geom_feat_key=geom_feat_key,
        local_pca_k=local_pca_k,
        equivar_or_invar=equivar_or_invar,
        debug=debug,
    )


def compute_masked_mse(
    model: torch.nn.Module,
    data_obj: torch.utils.data.Dataset,
    target_key: str = "y",
    mask_key: str = "valid_mask",
) -> float:
    """
    Compute MSE on masked nodes for a single-graph dataset.
    """
    if not hasattr(data_obj, mask_key):
        raise RuntimeError(f"Data is missing {mask_key} for MSE computation.")
    model_device = next(model.parameters()).device
    if hasattr(data_obj, "batch"):
        data_dev = data_obj.to(model_device)
    else:
        data_dev = Batch.from_data_list([data_obj]).to(model_device)
    model.eval()
    with torch.no_grad():
        out = model(data_dev)
        preds = out["preds"] if isinstance(out, dict) else out
        target = data_dev[target_key]
        mask = getattr(data_dev, mask_key).bool()
        if int(mask.sum()) == 0:
            return float("nan")
        if isinstance(preds, torch.Tensor) and preds.dim() == 3 and int(preds.shape[1]) == 1:
            preds = preds.squeeze(1)
        if isinstance(target, torch.Tensor) and target.dim() == 3 and int(target.shape[1]) == 1:
            target = target.squeeze(1)
        loss = torch.mean((preds[mask] - target[mask]) ** 2)
        return float(loss.item())


def run_replication(
    combo_dir: Path,
    base_config: object,
    sample_n: int,
    mask_prop: float,
    knn_k: int,
    local_pca_k: Optional[int],
    seed: int,
    combo_idx: int,
    rep_idx: int,
    do_rotation_eval: bool,
    reconstruct_Q_rot: bool,
    dataset_key: str,
    rotation_seed: int | None,
    debug: bool,
) -> Path:
    combo_dir.mkdir(parents=True, exist_ok=True)
    results_path = combo_dir / f"results_{rep_idx}.yaml"
    print(f"[INFO] Writing replication results to: {results_path}")
    if results_path.exists():
        raise Exception(f"Results already exist for combo {combo_idx} and replication {rep_idx}...exiting.")

    # ------------------------------------------------------------------
    # Prepare config
    # ------------------------------------------------------------------
    config = copy.deepcopy(base_config)

    # Override paths for this replication
    logs_dir = combo_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    config.train_logs_save_dir = str(logs_dir)

    # Ensure checkpoint directory exists even if best-model saving is enabled
    model_dir = combo_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    config.model_save_dir = str(model_dir)
    config.results_save_dir = str(combo_dir)
    config.dataset_config.dataset = dataset_key
    config.dataset_config.k_folds = 1
    vector_feat_key = config.dataset_config.vector_feat_key
    config.dataset_config.vector_feat_key = vector_feat_key
    config.dataset_config.vector_feat_dim = getattr(config.dataset_config, "vector_feat_dim", 3)
    geom_feat_key = getattr(config.dataset_config, "geom_feat_key", "pos")
    model_key = str(getattr(config.model_config, "model_key", "")).lower()
    if model_key in ("gcn", "gat", "gin", "legs"):
        if config.dataset_config.scalar_feat_key is None:
            config.dataset_config.scalar_feat_key = vector_feat_key
    config.model_config.k_neighbors = knn_k
    config.experiment_type = "tvt"

    # ------------------------------------------------------------------
    # Build dataset and dataloader
    # ------------------------------------------------------------------
    dataset_is_wind_rot = is_wind_rot_dataset(dataset_key)
    offline_comp_time = float("nan")
    test_data_raw = None
    test_data_proc = None
    base_data_for_rotation = None
    rotation_seed_val = rotation_seed if rotation_seed is not None else seed
    data_split_seed = None
    rotation_eval_seed = None

    if dataset_is_wind_rot:  # wind_rot dataset
        data_split_seed = int(rotation_seed_val + combo_idx * 1000 + rep_idx)
        rng = wind_rot_seed_rng(rotation_seed_val, combo_idx=combo_idx, rep_idx=rep_idx)
        train_data_raw, test_data_raw = create_wind_rot_datasets(
            root=config.dataset_config.data_dir,
            rng=rng,
            knn_k=knn_k,
            sample_n=sample_n,
            mask_prop=mask_prop,
            vector_feat_key=config.dataset_config.vector_feat_key,
        )
        base_data_for_rotation = None

        offline_t0 = time.time()
        train_data_proc = attach_operators(
            train_data_raw,
            vector_feat_key=config.dataset_config.vector_feat_key,
            geom_feat_key=geom_feat_key,
            ablate_scalar_track=getattr(config, "ablate_scalar_track"),
            local_pca_k=local_pca_k,
            debug=debug,
        )
        offline_comp_time = float(time.time() - offline_t0)

        print(train_data_proc)
        dataset = [train_data_proc]
        base_data_for_rotation = train_data_proc
    else:  # wind dataset
        data_split_seed = int(seed + combo_idx * 1000 + rep_idx)
        rng = seed_rng(seed, combo_idx=combo_idx, rep_idx=rep_idx)
        data_raw = create_dataset(
            root=config.dataset_config.data_dir,
            rng=rng,
            knn_k=knn_k,
            sample_n=sample_n,
            mask_prop=mask_prop,
            vector_feat_key=config.dataset_config.vector_feat_key,
        )
        base_data_for_rotation = None

        offline_t0 = time.time()
        data_proc = attach_operators(
            data_raw,
            vector_feat_key=config.dataset_config.vector_feat_key,
            geom_feat_key=geom_feat_key,
            ablate_scalar_track=getattr(config, "ablate_scalar_track"),
            local_pca_k=local_pca_k,
            debug=debug,
        )
        offline_comp_time = float(time.time() - offline_t0)
        print(data_proc)
        dataset = [data_proc]
        base_data_for_rotation = data_proc
    # Single DataLoader instance reused for train/valid; masks on the single graph control train/valid splits behavior in train.py
    single_loader = PyGDataLoader(dataset, batch_size=1, shuffle=False)
    dataloader_dict = {"train": single_loader} # , "valid": single_loader}

    # ------------------------------------------------------------------
    # Prepare accelerator, model, optimizer, and scheduler
    # ------------------------------------------------------------------
    acc = Accelerator(
        device_placement=False if config.using_pytorch_geo else config.device,
        mixed_precision="no" if config.mixed_precision == "none" else config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        split_batches=config.dataloader_split_batches,
    )
    if "vdw" in model_key:  # includes vdw_layer
        config, model, dataloader_dict = prepare_vdw_model(
            config,
            dataloader_dict,
            acc=acc,
        )
    elif model_key in ("gcn", "gat", "gin", "legs", "egnn", "tfn"):
        config, model, dataloader_dict = prepare_comparison_model(
            config,
            dataloader_dict,
            acc=acc,
        )
    else:
        raise ValueError(f"Unsupported model_key for wind runs: '{model_key}'")
    optimizer_class, optimizer_kwargs = prepare_optimizer(
        config.optimizer_config,
    )
    scheduler_class, scheduler_kwargs = prepare_scheduler(
        config.scheduler_config,
        config.validate_every,
    )

    # ------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------
    trained_model, records, epoch_ctr = train.train_model(
        config=config,
        dataloader=dataloader_dict,
        model=model,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        accelerator=acc,
    )

    # ------------------------------------------------------------------
    # Compute and save results
    # ------------------------------------------------------------------
    best_epoch = 0
    try:
        if epoch_ctr is not None:
            best_epoch = int(epoch_ctr.best.get(config.main_metric, {}).get("epoch", 0))
    except Exception:
        best_epoch = 0

    best_epoch_fallback = records[-1].get("epoch", 0) if records else 0
    best_epoch_value = best_epoch if (best_epoch > 0) else best_epoch_fallback

    mean_train_time = float("nan")
    mean_infer_time = float("nan")
    if records and best_epoch_value > 0:
        subset = [r for r in records if r.get("epoch", 0) <= best_epoch_value]
        if subset:
            mean_train_time = sum(float(r.get("train_time_sec", 0.0)) for r in subset) / len(subset)
            mean_infer_time = sum(float(r.get("valid_inference_time_sec", 0.0)) for r in subset) / len(subset)

    last_record = records[-1] if records else {}
    train_mse = last_record.get("mse_train", float("nan"))
    valid_mse = last_record.get("mse_valid", float("nan"))
    if records and best_epoch_value > 0:
        best_epoch_record = next(
            (r for r in records if int(r.get("epoch", 0)) == best_epoch_value),
            None,
        )
        if best_epoch_record is not None:
            valid_mse = best_epoch_record.get("mse_valid", valid_mse)
    param_count = count_parameters(trained_model)
    test_mse = float("nan")
    region_test_mse = float("nan")
    post_train_valid_mse = float("nan")
    if base_data_for_rotation is not None:
        try:
            post_train_valid_mse = compute_masked_mse(
                trained_model,
                base_data_for_rotation,
                target_key=config.dataset_config.target_key,
                mask_key="valid_mask",
            )
        except Exception:
            post_train_valid_mse = float("nan")
        try:
            test_mse = compute_masked_mse(
                trained_model,
                base_data_for_rotation,
                target_key=config.dataset_config.target_key,
                mask_key="test_mask",
            )
        except Exception:
            test_mse = float("nan")
    if base_data_for_rotation is not None and post_train_valid_mse == post_train_valid_mse:
        valid_mse = post_train_valid_mse
    if dataset_is_wind_rot and test_data_raw is not None:
        # Recompute Q on test geometry just before evaluation
        test_data_proc = attach_operators(
            test_data_raw,
            vector_feat_key=config.dataset_config.vector_feat_key,
            geom_feat_key=geom_feat_key,
            ablate_scalar_track=getattr(config, "ablate_scalar_track"),
            local_pca_k=local_pca_k,
            debug=debug,
        )
        region_test_mse = compute_masked_mse(
            trained_model,
            test_data_proc,
            target_key=config.dataset_config.target_key,
            mask_key="test_mask",
        )

    # --------------------------------------------------------------
    # Rotation equivariance evaluation (no retraining)
    # --------------------------------------------------------------
    rotation_mse = float("nan")
    rot_deg = float("nan")
    if do_rotation_eval and base_data_for_rotation is not None:
        adj_rot_seed = rotation_seed_val + combo_idx * 1000 + rep_idx
        rotation_eval_seed = int(adj_rot_seed)
        rot_rng = np.random.RandomState(adj_rot_seed)
        vector_dim = int(getattr(config.dataset_config, "vector_feat_dim", 2) or 2)
        if vector_dim == 3:
            rot_data, rot_deg, rot_mat = apply_random_3d_rotation(
                data=base_data_for_rotation,
                rng=rot_rng,
                vector_feat_key=config.dataset_config.vector_feat_key,
                target_key=config.dataset_config.target_key,
                return_matrix=True,
            )
        # else:
        #     rot_data, rot_deg, rot_mat = apply_random_2d_rotation(
        #         data=base_data_for_rotation,
        #         rng=rot_rng,
        #         vector_feat_key=config.dataset_config.vector_feat_key,
        #         target_key=config.dataset_config.target_key,
        #         return_matrix=True,
            # )

        # Rotate held-out nodes' (mean) vector features
        # if hasattr(rot_data, "valid_mask"):
        #     valid_mask = rot_data.valid_mask.bool()
        #     if hasattr(base_data_for_rotation, config.dataset_config.target_key):
        #         base_targets = getattr(base_data_for_rotation, config.dataset_config.target_key)
        #         mean_vec = base_targets[~valid_mask].mean(dim=0, keepdim=True)
        #         rotated_mean = mean_vec @ rot_mat.T
        #         if hasattr(rot_data, config.dataset_config.vector_feat_key):
        #             vec = getattr(rot_data, config.dataset_config.vector_feat_key)
        #             vec[valid_mask] = rotated_mean.repeat(valid_mask.sum(), 1)
        #             setattr(rot_data, config.dataset_config.vector_feat_key, vec)

        if reconstruct_Q_rot:
            # Legacy behavior: rebuild operators from scratch on rotated geometry
            rot_data = attach_operators(
                rot_data,
                vector_feat_key=config.dataset_config.vector_feat_key,
                geom_feat_key=geom_feat_key,
                ablate_scalar_track=getattr(config, "ablate_scalar_track"),
                local_pca_k=local_pca_k,
                debug=debug,
            )
        else:
            # Default: rotate the existing operator by conjugation
            if not hasattr(rot_data, "Q"):
                raise RuntimeError(
                    "Rotation eval expected precomputed Q on base_data_for_rotation, but 'Q' was missing."
                )
            Q_rot = rotate_sparse_block_operator(getattr(rot_data, "Q"), rot_mat)
            setattr(rot_data, "Q", Q_rot)
        rotation_mse = compute_masked_mse(
            trained_model,
            rot_data,
            target_key=config.dataset_config.target_key,
            mask_key="test_mask",
        )

        if debug and ("vdw" in model_key):
            report_path = combo_dir / f"debug_report_{rep_idx}.txt"
            report_lines = [
                f"combo_idx: {combo_idx}  rep_idx: {rep_idx}",
                f"dataset_seed_used: {data_split_seed}",
                f"rotation_seed_used: {rotation_eval_seed}",
                "rot_mat_reused: True",
                f"vector_feat_key: {config.dataset_config.vector_feat_key}",
                f"target_key: {config.dataset_config.target_key}",
                f"post_train_valid_mse: {post_train_valid_mse:.6e}",
                "valid_mse_definition: coordinate-wise",
            ]
            best_loaded = False
            if (epoch_ctr is not None) and hasattr(epoch_ctr, "best_model_wts"):
                best_state = getattr(epoch_ctr, "best_model_wts")
                if best_state is not None:
                    try:
                        real_model = trained_model.module if hasattr(trained_model, "module") else trained_model
                        real_model.load_state_dict(best_state)
                        report_lines.append("best_weights_loaded: True (memory)")
                        best_loaded = True
                    except Exception as exc:
                        report_lines.append(f"best_weights_loaded: False (memory: {exc})")
            if (not best_loaded):
                best_dir = Path(config.model_save_dir) / "best"
                if config.save_best_model_state and best_dir.exists():
                    try:
                        real_model = trained_model.module if hasattr(trained_model, "module") else trained_model
                        _load_pretrained_weights(real_model, str(best_dir), verbosity=config.verbosity)
                        report_lines.append("best_weights_loaded: True (disk)")
                    except Exception as exc:
                        report_lines.append(f"best_weights_loaded: False (disk: {exc})")
                else:
                    report_lines.append("best_weights_loaded: False (missing best snapshot)")
            model_device = next(trained_model.parameters()).device
            report = generate_equivariance_report(
                model=trained_model,
                data_a=base_data_for_rotation,
                data_b=rot_data,
                rot_mat=rot_mat,
                device=model_device,
                target_key=str(config.dataset_config.target_key),
                rotation_mse_check=rotation_mse,
                rot_deg=float(rot_deg),
                rotation_seed=rotation_eval_seed,
                extra_lines=report_lines,
            )
            report_path.write_text(report)

    results = {
        "sample_n": sample_n,
        "mask_prop": mask_prop,
        "replication": rep_idx,
        "data_split_seed": data_split_seed,
        "rotation_eval_seed": rotation_eval_seed,
        "parameter_count": param_count,
        "best_epoch": int(best_epoch_value),
        "mean_train_time": float(mean_train_time),
        "mean_infer_time": float(mean_infer_time),
        "offline_comp_time": float(offline_comp_time),
        "train_mse": float(train_mse) \
            if train_mse is not None else float("nan"),
        "valid_mse": float(valid_mse) \
            if valid_mse is not None else float("nan"),
        "test_mse": float(test_mse),
        "region_test_mse": float(region_test_mse),
        "rotation_mse": float(rotation_mse),
        "rotation_degree": float(rot_deg),
    }
    results_utils.write_yaml(results, results_path)
    return results_path


def main():
    args = parse_args()
    _, base_config = prep_config(args)
    if args.rotation_seed is not None:
        setattr(base_config, "rotation_seed", args.rotation_seed)

    dataset_key = getattr(base_config.dataset_config, "dataset", "wind")
    dataset_is_wind_rot = is_wind_rot_dataset(dataset_key)
    if dataset_is_wind_rot:
        if args.sample_n is None or args.mask_prop is None:
            sample_ns = [WIND_ROT_SAMPLE_N]
            mask_props = [WIND_ROT_MASK_PROP]
        else:
            sample_ns = parse_list(args.sample_n, int)
            mask_props = parse_list(args.mask_prop, float)
    else:
        if args.sample_n is None or args.mask_prop is None:
            raise ValueError("sample_n and mask_prop must be provided for wind experiments.")
        sample_ns = parse_list(args.sample_n, int)
        mask_props = parse_list(args.mask_prop, float)
    combos = product(sample_ns, mask_props)

    root_dir = Path(base_config.root_dir)
    experiment_root = dataset_key
    exp_name = args.exp_name or getattr(base_config.model_config, "model_key", "vdw")

    base_dir = root_dir / "experiments" / experiment_root / exp_name
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Base results directory: {base_dir}")

    # Loop experiment combos
    for combo_idx, (n, p) in enumerate(combos):
        combo_slug = format_combo(n, p)
        combo_dir = base_dir / combo_slug
        result_paths: List[Path] = []

        # Loop replications
        for rep_idx in range(args.replications):
            rp = run_replication(
                combo_dir=combo_dir,
                base_config=base_config,
                sample_n=n,
                mask_prop=p,
                knn_k=args.knn_k,
                local_pca_k=args.local_pca_k,
                seed=args.seed + combo_idx,
                combo_idx=combo_idx,
                rep_idx=rep_idx,
                do_rotation_eval=args.do_rotation_eval,
                reconstruct_Q_rot=args.reconstruct_Q_rot,
                dataset_key=dataset_key,
                rotation_seed=getattr(base_config, "rotation_seed", None),
                debug=args.debug,
            )
            result_paths.append(rp)

        # Aggregate results across replications and save
        aggregated = results_utils.aggregate_replication_results(
            result_paths,
            metrics=[
                "train_mse",
                "valid_mse",
                "test_mse",
                "region_test_mse",
                "rotation_mse",
                "best_epoch",
                "mean_train_time",
                "mean_infer_time",
                "parameter_count",
                "offline_comp_time",
            ],
        )
        aggregated["sample_n"] = n
        aggregated["mask_prop"] = p
        results_utils.write_yaml(
            aggregated,
            combo_dir / "reps_results.yaml",
        )


if __name__ == "__main__":
    main()
