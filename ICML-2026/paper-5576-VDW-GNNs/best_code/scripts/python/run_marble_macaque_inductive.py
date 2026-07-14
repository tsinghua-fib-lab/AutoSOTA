#!/usr/bin/env python3
"""
Inductive MARBLE training on the macaque reaching dataset with trial-wise k-fold CV.

Pipeline overview:
1. Build the spatial day graph solely from training-trial samples, adding edges via
   CkNN (with configurable k/delta) so hold-out trials never leak into the graph.
2. Train MARBLE contrastively on this train-only PyG graph (neighbor sampling,
   optimizer/scheduler overrides, patience-controlled early stopping).
3. Recompute MARBLE embeddings for all train nodes, flatten each trial's sequence
   of node embeddings into a feature vector, and fit an SVM probe on the labeled
   training trials. For validation/test trials that lack explicit nodes, estimate
   their embeddings as weighted averages of their cached k-nearest training
   neighbors before flattening. Apply the trained SVM to every held-out trial and
   report accuracy at the trial level (i.e., percent of trials whose condition
   labels are predicted correctly).

Additional details: k-fold splits are materialized via `MarbleFoldData`, experiment
artifacts (config, weights, embeddings, metrics, SVM) are stored per fold, and CLI
flags allow overriding optimizer, scheduler, batching, device, and kNN settings.

Parameter resolution summary (highest precedence first):
- CLI flags override everything.
- YAML config (e.g., config/yaml_files/macaque/marble.yaml): training.* (epochs, batch_size,
  patience) and model.* (k_neighbors, cknn_delta/dist_metric, plateau knobs) feed into the runner.
- Built-in fallbacks (DEFAULT_* constants) cover any unset values.
- MARBLE defaults in models/comparisons/MARBLE/default_params.yaml fill the rest; we override
  only batch_size, epochs, and lr (when provided via CLI).
"""

import argparse
# import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from torch import optim
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

# Add project root to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_CODE_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_CODE_DIR))

from data_processing.cknn import cknneighbors_graph  # noqa: E402
from data_processing.macaque_reaching import (  # noqa: E402
    MarbleFoldData,
    macaque_prepare_marble_fold_data,
)
from models.comparisons.MARBLE import dataloader, utils  # noqa: E402
from models.comparisons.MARBLE.main import net  # noqa: E402
from models.comparisons.MARBLE.preprocessing import _compute_geometric_objects  # noqa: E402
from os_utilities import create_experiment_dir, ensure_dir_exists, smart_pickle  # noqa: E402
from training.macaque_inductive_utils import (  # noqa: E402
    build_split_features,
    build_train_graph,
    infer_trial_embeddings,
    save_embeddings,
    train_and_evaluate_svm,
)
from models.nn_utilities import count_parameters  # noqa: E402
from training.kfold_results import compute_kfold_results


# Default MARBLE config (includes MARBLE-specific overrides like k-NN/CkNN and plateau knobs)
DEFAULT_CONFIG = _CODE_DIR / "config" / "yaml_files" / "macaque" / "marble.yaml"
# Base experiment defaults (provides dataset/root_dir/etc. to merge with MARBLE overrides)
DEFAULT_BASE_EXPERIMENT_CONFIG = _CODE_DIR / "config" / "yaml_files" / "macaque" / "experiment.yaml"
DEFAULT_PARAMS_YAML = _CODE_DIR / "models" / "comparisons" / "MARBLE" / "default_params.yaml"
DEFAULT_K_NEIGHBORS = 30
DEFAULT_CKNN_DELTA = 1.5
DEFAULT_CKNN_DIST_METRIC = "euclidean"
DEFAULT_CKNN_MIN_NBRS = None
DEFAULT_PLATEAU_PATIENCE = 10
DEFAULT_PLATEAU_MAX_RESTARTS = 2
DEFAULT_PLATEAU_FACTOR = 0.1


def parse_args() -> argparse.Namespace:
    """
    Build and parse CLI arguments for the inductive MARBLE runner.
    """
    parser = argparse.ArgumentParser(
        description="Train MARBLE inductively with k-fold CV on the macaque reaching dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to experiment YAML.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Override patience for early stopping.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override optimizer learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override MARBLE batch size.",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=None,
        help="Number of neighbors for both spatial graph construction and held-out embedding mapping.",
    )
    parser.add_argument(
        "--cknn_delta",
        type=float,
        default=None,
        help="Delta parameter for CkNN graph building.",
    )
    parser.add_argument(
        "--cknn_dist_metric",
        type=str,
        default=None,
        help="Distance metric for CkNN and held-out mapping (e.g., euclidean, cosine, l1).",
    )
    parser.add_argument(
        "--cknn_min_nbrs",
        type=int,
        default=None,
        help="Minimum neighbors to enforce per node for CkNN (add k-NN edges if deg < min).",
    )
    parser.add_argument(
        "--cknn_min_nbrs_match_pca",
        action="store_true",
        help="If set, enforce min neighbors equal to the PCA/manifold dim (overrides --cknn_min_nbrs).",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Maximum MARBLE epochs (default from config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Compute device.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Random seed override.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="marble",
        help="Experiment model name.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce MARBLE verbosity.",
    )
    parser.add_argument(
        "--svm_kernel",
        type=str,
        default="rbf",
        help="Kernel to use for the validation/test SVM probe.",
    )
    parser.add_argument(
        "--svm_c",
        type=float,
        default=1.0,
        help="C (regularization) parameter for the SVM probe. MARBLE setting (and default): 1.0",
    )
    parser.add_argument(
        "--svm_gamma",
        type=str,
        default="scale",
        help="Gamma parameter for the SVM probe (string or float).",
    )
    parser.add_argument(
        "--plateau_patience",
        type=int,
        default=None,
        help="Epochs without val improvement before lowering LR and resetting to best weights.",
    )
    parser.add_argument(
        "--plateau_max_restarts",
        type=int,
        default=None,
        help="Number of times to reduce LR and reset to best weights before final patience early stop.",
    )
    parser.add_argument(
        "--plateau_factor",
        type=float,
        default=None,
        help="Multiplicative LR decay when plateau patience is hit.",
    )
    parser.add_argument(
        "--original_scheduler",
        action="store_true",
        help=(
            "Use MARBLE's original 100-epoch training with ReduceLROnPlateau on "
            "training loss (disables custom LR reduction and early stopping)."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        help="Space-separated fold indices to run (defaults to all folds).",
    )
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        help="Existing experiment directory to reuse; skips creation of a new one.",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict:
    """
    Load a YAML configuration file into a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml_config(path: Path, config: Dict) -> None:
    """
    Persist a YAML configuration dictionary to disk, ensuring its directory exists.
    """
    ensure_dir_exists(path.parent, raise_exception=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge dictionaries (override wins).
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def set_global_seed(seed: int, set_torch_seed: bool = False) -> None:
    """
    Seed Python, NumPy, and (optionally) PyTorch RNGs for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    if set_torch_seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def instantiate_optimizer(
    optimizer_cfg: Dict,
    params,
    lr_override: float | None,
) -> optim.Optimizer:
    """
    Instantiate the optimizer defined in the YAML config (with optional CLI overrides).
    """
    key = optimizer_cfg.get("optimizer_key", "AdamW").lower()
    lr = float(lr_override) if lr_override is not None else float(optimizer_cfg.get("learn_rate", 1e-3))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
    eps = float(optimizer_cfg.get("eps", 1e-8))
    betas = optimizer_cfg.get("betas", [0.9, 0.999])
    if key == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, eps=eps, betas=tuple(betas))
    if key == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, eps=eps, betas=tuple(betas))
    if key == "sgd":
        momentum = float(optimizer_cfg.get("momentum", 0.9))
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    if key == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay, eps=eps)
    raise ValueError(f"Unsupported optimizer key '{optimizer_cfg.get('optimizer_key')}'.")


def instantiate_scheduler(
    scheduler_cfg: Dict,
    optimizer: optim.Optimizer,
) -> Tuple[torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None, str]:
    """
    Create the configured LR scheduler and return it with a tag indicating its stepping mode.
    """
    key = scheduler_cfg.get("scheduler_key")
    if not key:
        return None, ""
    key_lower = key.lower()
    if key_lower == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "min"),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 32)),
            threshold=float(scheduler_cfg.get("threshold", 1e-4)),
            threshold_mode=scheduler_cfg.get("threshold_mode", "rel"),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
        return scheduler, "plateau"
    if key_lower == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("T_max", 50)),
            eta_min=float(scheduler_cfg.get("min_lr", 0.0)),
        )
        return scheduler, "step"
    raise ValueError(f"Unsupported scheduler key '{key}'.")


def run_train_epoch(
    model: net,
    data,
    loader,
    optimizer: optim.Optimizer,
) -> float:
    """
    Execute one epoch of MARBLE training with the contrastive loss.
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        _, n_id, adjs = batch
        adjs = [adj.to(data.x.device) for adj in utils.to_list(adjs)]
        n_id = n_id.to(data.x.device)
        optimizer.zero_grad()
        emb, mask = model.forward(data, n_id, adjs)
        loss = model.loss(emb, mask)
        loss.backward()
        grad_mismatches = []
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.device != param.device:
                grad_mismatches.append((name, str(param.grad.device), str(param.device)))
                param.grad = param.grad.to(param.device, non_blocking=True)
        if grad_mismatches:
            print(
                "Corrected gradient device mismatches:",
                ", ".join(
                    f"{name} ({src} -> {dst})"
                    for name, src, dst in grad_mismatches[:5]
                ),
            )
        optimizer.step()
        total_loss += float(loss)
    return total_loss / max(len(loader), 1)


def compute_node_embeddings(model: net, data) -> np.ndarray:
    """
    Run a full-graph forward pass to capture per-node MARBLE embeddings.
    """
    model.eval()
    data_eval = data.clone()
    with torch.no_grad():
        transformed = model.transform(data_eval)
        embeddings = transformed.emb.detach().cpu().numpy()
    return embeddings


def run_training_epochs(
    *,
    fold_idx: int,
    marble_model: net,
    train_data,
    loader,
    fold_data: MarbleFoldData,
    max_epochs: int,
    patience: int,
    scheduler,
    scheduler_mode: str,
    optimizer: optim.Optimizer,
    log,
    svm_kernel: str,
    svm_C: float,
    svm_gamma,
    plateau_patience: int,
    plateau_max_restarts: int,
    plateau_factor: float,
    use_original_scheduler: bool,
    epoch_log,
) -> Tuple[Dict[str, torch.Tensor], int, float, List[Dict[str, float]]]:
    """
    Train MARBLE for a single fold with SVM-based validation monitoring/early stopping.
    """
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_acc = -np.inf
    epochs_without_improve = 0
    history: List[Dict[str, float]] = []
    restarts_remaining = int(plateau_max_restarts)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        train_start = time.time()
        loss_train = run_train_epoch(marble_model, train_data, loader, optimizer)
        train_time = time.time() - train_start
        infer_forward_start = time.time()
        train_embeddings = compute_node_embeddings(marble_model, train_data)
        marble_model, train_data, _ = utils.move_to_gpu(marble_model, train_data)
        train_feats, train_labels, _ = build_split_features(
            fold_data.train_trials,
            train_embeddings,
            fold_data.nodes_per_trial,
        )
        val_feats, val_labels, _ = build_split_features(
            fold_data.valid_trials,
            train_embeddings,
            fold_data.nodes_per_trial,
        )
        forward_time = time.time() - infer_forward_start

        svm_start = time.time()
        svm_stats = train_and_evaluate_svm(
            train_feats=train_feats,
            train_labels=train_labels,
            val_feats=val_feats,
            val_labels=val_labels,
            test_feats=np.empty((0, train_feats.shape[1]), dtype=np.float32),
            test_labels=np.empty((0,), dtype=np.int64),
            test_trial_ids=None,
            kernel=svm_kernel,
            C=svm_C,
            gamma=svm_gamma,
        )
        svm_time = time.time() - svm_start
        train_acc = float(svm_stats["train_accuracy"])
        val_acc = svm_stats["val_accuracy"]
        infer_time = forward_time
        val_correct = int(svm_stats["val_correct"])
        val_total = int(svm_stats["val_total"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss_train),
                "val_accuracy": float(val_acc),
                "epoch_time_sec": time.time() - epoch_start,
                "train_time_sec": float(train_time),
                "inference_time_sec": float(infer_time),
                "svm_time_sec": float(svm_time),
                "full_eval_time_sec": float(infer_time + svm_time),
                "val_correct": val_correct,
                "val_total": val_total,
            }
        )
        epoch_log(
            f"Epoch {epoch:03d} | train_loss={loss_train:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} ({val_correct}/{val_total}) | "
            f"train_time={train_time:.2f}s | infer_fwd_time={infer_time:.2f}s | svm_time={svm_time:.2f}s"
        )

        if len(val_labels) > 0 and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in marble_model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        # Custom plateau LR reduction/reset is disabled when using original scheduler regime
        if (
            not use_original_scheduler
            and best_state is not None
            and epochs_without_improve >= plateau_patience
            and restarts_remaining > 0
        ):
            restarts_remaining -= 1
            epochs_without_improve = 0
            for group in optimizer.param_groups:
                old_lr = float(group.get("lr", 0.0))
                group["lr"] = old_lr * plateau_factor
            marble_model.load_state_dict(best_state)
            log(
                f"[Fold {fold_idx}] Plateau hit: reduced LR by factor {plateau_factor}, "
                f"reloaded best weights (restart {plateau_max_restarts - restarts_remaining}/{plateau_max_restarts})."
            )
            continue

        if scheduler is not None:
            if scheduler_mode == "plateau":
                scheduler.step(loss_train)
            else:
                scheduler.step()

        # Final early stopping only after restarts are exhausted (disabled for original scheduler)
        if (
            not use_original_scheduler
            and restarts_remaining == 0
            and epochs_without_improve >= patience
        ):
            log(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    if best_state is None:
        raise RuntimeError("Training failed to record a best model state.")

    return best_state, best_epoch, best_val_acc, history


def evaluate_and_save_fold(
    *,
    fold_idx: int,
    marble_model: net,
    train_data,
    fold_data: MarbleFoldData,
    optimizer: optim.Optimizer,
    best_state: Dict[str, torch.Tensor],
    best_epoch: int,
    best_val_acc: float,
    history: List[Dict[str, float]],
    embeddings_dir: Path,
    metrics_dir: Path,
    models_dir: Path,
    patience: int,
    max_epochs: int,
    log,
    svm_kernel: str,
    svm_C: float,
    svm_gamma,
) -> Dict[str, float]:
    """
    Load the best MARBLE weights, compute final embeddings, fit the SVM once, and persist artifacts/results.
    """
    marble_model.load_state_dict(best_state)
    parameter_count = safe_count_parameters(marble_model, log)

    log("Recomputing embeddings with best MARBLE weights...")
    best_embeddings = compute_node_embeddings(marble_model, train_data)
    train_feats, train_labels, train_trial_ids = build_split_features(
        fold_data.train_trials,
        best_embeddings,
        fold_data.nodes_per_trial,
    )
    val_feats, val_labels, val_trial_ids = build_split_features(
        fold_data.valid_trials,
        best_embeddings,
        fold_data.nodes_per_trial,
    )
    test_feats, test_labels, test_trial_ids = build_split_features(
        fold_data.test_trials,
        best_embeddings,
        fold_data.nodes_per_trial,
    )

    svm_stats = train_and_evaluate_svm(
        train_feats=train_feats,
        train_labels=train_labels,
        val_feats=val_feats,
        val_labels=val_labels,
        test_feats=test_feats,
        test_labels=test_labels,
        test_trial_ids=test_trial_ids,
        kernel=svm_kernel,
        C=svm_C,
        gamma=svm_gamma,
    )
    svm = svm_stats["svm"]
    train_acc = float(svm_stats["train_accuracy"])
    val_acc = float(svm_stats["val_accuracy"])
    test_acc = float(svm_stats["test_accuracy"])

    log(
        f"[Fold {fold_idx}] best_epoch={best_epoch} | train_acc={train_acc:.4f} "
        f"| val_acc={val_acc:.4f} | test_acc={test_acc:.4f}"
    )

    ensure_dir_exists(models_dir, raise_exception=True)
    torch.save(
        {
            "epoch": best_epoch,
            "state_dict": best_state,
            "optimizer_state": optimizer.state_dict(),
            "val_accuracy": best_val_acc,
            "history": history,
            "params": marble_model.params,
        },
        models_dir / "best_model.pt",
    )
    with open(models_dir / "svm.pkl", "wb") as f:
        pickle.dump(svm, f)

    history_until_best = [h for h in history if h.get("epoch", 0) <= best_epoch]
    if history_until_best:
        mean_train_time = sum(h.get("train_time_sec", 0.0) for h in history_until_best) / len(history_until_best)
        mean_infer_time = sum(h.get("inference_time_sec", 0.0) for h in history_until_best) / len(history_until_best)
    else:
        mean_train_time = float("nan")
        mean_infer_time = float("nan")

    # Persist train history separately from the summary results
    with open(metrics_dir / "train_history.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(history, f)

    save_embeddings(embeddings_dir, "train", train_feats, train_labels, train_trial_ids)
    save_embeddings(embeddings_dir, "valid", val_feats, val_labels, val_trial_ids)
    save_embeddings(embeddings_dir, "test", test_feats, test_labels, test_trial_ids)

    prob_records = svm_stats["test_prob_records"]
    smart_pickle(str(metrics_dir / "test_probabilities.pkl"), prob_records, overwrite=True)

    results = {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "parameter_count": int(parameter_count) if parameter_count is not None else None,
        "mean_train_time": float(mean_train_time),
        "mean_infer_time": float(mean_infer_time),
    }
    with open(metrics_dir / "results.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f)
    return results


def safe_count_parameters(
    model: net,
    log,
) -> int | None:
    """
    Safely count model parameters, logging a warning on failure.
    """
    try:
        return count_parameters(model)
    except Exception as exc:
        log(f"Warning: failed to count parameters: {exc}")
        return None


def main() -> None:
    """
    Entry point for k-fold inductive MARBLE training with SVM supervision and evaluation.
    """
    args = parse_args()
    base_cfg = load_yaml_config(DEFAULT_BASE_EXPERIMENT_CONFIG)
    user_cfg = load_yaml_config(args.config)
    cfg = deep_merge(base_cfg, user_cfg)

    # Guarantee required sections/keys from the base experiment config
    cfg.setdefault("training", {})
    cfg.setdefault("dataset", {})
    # Backfill any missing dataset/train keys from base_cfg to avoid KeyErrors
    for key, val in base_cfg.get("training", {}).items():
        cfg["training"].setdefault(key, val)
    for key, val in base_cfg.get("dataaset", {}).items():
        cfg["dataset"].setdefault(key, val)

    training_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]
    optimizer_cfg = cfg.get("optimizer", {})
    scheduler_cfg = cfg.get("scheduler", {})
    model_cfg = cfg.get("model", {})

    seed = args.seed if args.seed is not None \
        else int(dataset_cfg.get("split_seed", 123456))
    set_global_seed(seed)

    patience = args.patience if args.patience is not None else int(
        scheduler_cfg.get(
            "patience",
            training_cfg.get("no_valid_metric_improve_patience", 32),
        )
    )
    if args.original_scheduler:
        max_epochs = 100
    else:
        max_epochs = args.max_epochs \
            if args.max_epochs is not None \
            else int(training_cfg.get("n_epochs", 100))
    plateau_patience = (
        args.plateau_patience
        if args.plateau_patience is not None
        else int(
            model_cfg.get(
                "plateau_patience",
                scheduler_cfg.get(
                    "patience",
                    training_cfg.get("no_valid_metric_improve_patience", DEFAULT_PLATEAU_PATIENCE),
                ),
            )
        )
        if model_cfg.get("plateau_patience") is not None
        or scheduler_cfg.get("patience") is not None
        or training_cfg.get("no_valid_metric_improve_patience") is not None
        else DEFAULT_PLATEAU_PATIENCE
    )
    plateau_max_restarts = (
        args.plateau_max_restarts
        if args.plateau_max_restarts is not None
        else int(model_cfg.get("plateau_max_restarts", DEFAULT_PLATEAU_MAX_RESTARTS))
    )
    plateau_factor = (
        args.plateau_factor
        if args.plateau_factor is not None
        else float(model_cfg.get("plateau_factor", DEFAULT_PLATEAU_FACTOR))
    )
    k_neighbors = (
        args.k_neighbors
        if args.k_neighbors is not None
        else int(model_cfg.get("k_neighbors", DEFAULT_K_NEIGHBORS))
    )
    cknn_delta = (
        args.cknn_delta
        if args.cknn_delta is not None
        else float(model_cfg.get("cknn_delta", DEFAULT_CKNN_DELTA))
    )
    cknn_dist_metric = (
        args.cknn_dist_metric
        if args.cknn_dist_metric is not None
        else str(model_cfg.get("cknn_dist_metric", DEFAULT_CKNN_DIST_METRIC))
    )
    cknn_min_nbrs_match_pca = bool(
        args.cknn_min_nbrs_match_pca
        or model_cfg.get("cknn_min_nbrs_match_pca", False)
    )
    cknn_min_nbrs = args.cknn_min_nbrs
    if cknn_min_nbrs is None:
        cfg_min = model_cfg.get("cknn_min_nbrs")
        if cfg_min is not None:
            cknn_min_nbrs = int(cfg_min)
    batch_size = args.batch_size \
        if args.batch_size is not None \
        else int(training_cfg.get("batch_size", 64))
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )

    results_root = Path(training_cfg["root_dir"]).expanduser() / training_cfg["save_dir"]
    if args.experiment_dir is not None:
        exp_dir_path = args.experiment_dir.expanduser()
        if not exp_dir_path.exists():
            raise FileNotFoundError(f"Experiment directory '{exp_dir_path}' does not exist.")
        ensure_dir_exists(exp_dir_path / "config", raise_exception=True)
        exp_dirs = {
            "exp_dir": str(exp_dir_path),
            "config_save_path": str(exp_dir_path / "config" / "config.yaml"),
        }
    else:
        exp_dirs = create_experiment_dir(
            root_dir=results_root,
            model_name=args.model_name,
            dataset_name=dataset_cfg.get("dataset", "macaque"),
            experiment_id=training_cfg.get("experiment_id"),
            config=cfg,
            verbosity=int(training_cfg.get("verbosity", 0)),
        )
    save_yaml_config(Path(exp_dirs["config_save_path"]), cfg)

    # Load default MARBLE parameters
    with open(DEFAULT_PARAMS_YAML, "r", encoding="utf-8") as f:
        marble_params = yaml.safe_load(f)
    # Override default MARBLE parameters with CLI overrides
    marble_params["batch_size"] = batch_size
    marble_params["epochs"] = max_epochs
    if args.lr is not None:
        marble_params["lr"] = args.lr

    root = Path(training_cfg["root_dir"])
    dataset_root = root / dataset_cfg["data_dir"]
    day_index = dataset_cfg.get("macaque_day_index")
    k_folds = int(training_cfg.get("k_folds", dataset_cfg.get("k_folds", 5)))
    if args.folds:
        fold_indices: List[int] = []
        seen = set()
        for idx in args.folds:
            if idx < 0 or idx >= k_folds:
                raise ValueError(f"Fold index {idx} is out of range for k_folds={k_folds}.")
            if idx not in seen:
                fold_indices.append(idx)
                seen.add(idx)
    else:
        fold_indices = list(range(k_folds))

    # Iterate over k-folds
    for fold_idx in fold_indices:
        fold_root = Path(exp_dirs["exp_dir"]) / f"fold_{fold_idx}"
        metrics_dir = fold_root / "metrics"
        models_dir = fold_root / "models"
        logs_dir = fold_root / "logs"
        log_path = logs_dir / "training.log"
        config_dir = fold_root / "config"
        embeddings_dir = fold_root / "embeddings"

        for dir_path in (
            metrics_dir, 
            models_dir, 
            logs_dir, 
            config_dir, 
            embeddings_dir,
        ):
            ensure_dir_exists(dir_path, raise_exception=True)
        save_yaml_config(config_dir / "config.yaml", cfg)

        def log(msg: str) -> None:
            timestamp = time.strftime("%H:%M:%S")
            line = f"[{timestamp}] {msg}"
            print(line)
            with open(log_path, "a", encoding="utf-8") as log_f:
                log_f.write(line + "\n")

        def log_fold_header(msg: str) -> None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] {msg}"
            print(line)
            with open(log_path, "a", encoding="utf-8") as log_f:
                log_f.write(line + "\n")

        # Prepare fold data
        log_fold_header(f"[Fold {fold_idx}] Preparing fold {fold_idx} data (orig. k-folds split seed={seed})...")
        fold_data = macaque_prepare_marble_fold_data(
            data_root=str(dataset_root),
            day_index=day_index,
            k_folds=k_folds,
            fold_i=fold_idx,
            seed=seed,
            include_lever_velocity=False,
            k_neighbors=k_neighbors,
            cknn_dist_metric=cknn_dist_metric,
        )

        # Build MARBLE training graph
        log(
            f"Building MARBLE training graph (cknn_metric={cknn_dist_metric})..."
        )
        # Resolve min_neighbors: CLI explicit int takes precedence, otherwise match PCA dim if requested
        min_neighbors: int | None
        if cknn_min_nbrs_match_pca:
            try:
                min_neighbors = int(fold_data.state_dim)
            except Exception:
                min_neighbors = int(fold_data.train_positions.shape[1])
        else:
            min_neighbors = cknn_min_nbrs

        train_data = build_train_graph(
            fold_data, 
            k_neighbors, 
            cknn_delta,
            dist_metric=cknn_dist_metric,
            min_neighbors=min_neighbors,
        )
        # Compute geometric objects needed by MARBLE (Laplace spectra, gauges, kernels)
        try:
            train_data = _compute_geometric_objects(
                train_data,
                n_geodesic_nb=int(marble_params.get("n_geodesic_nb", k_neighbors)),
                var_explained=float(marble_params.get("var_explained", 0.9)),
                local_gauges=bool(marble_params.get("local_gauges", False)),
                number_of_eigenvectors=marble_params.get("number_of_eigenvectors"),
            )
        except Exception as exc:
            log(f"Failed to compute geometric objects: {exc}")
            raise

        # Instantiate MARBLE model
        marble_model = net(
            train_data, 
            params=dict(marble_params), 
            verbose=not args.quiet
        )

        # Instantiate data loader
        loader = dataloader.NeighborSampler(
            train_data.edge_index,
            sizes=[marble_model.params["n_sampled_nb"]] \
                * marble_model.params["order"],
            batch_size=marble_model.params["batch_size"],
            shuffle=True,
            num_nodes=train_data.num_nodes,
            node_idx=train_data.train_mask,
        )

        # Move model/data to device
        log(f"Moving model/data to {device}...")
        utils.device = device
        marble_model, train_data, _ = utils.move_to_gpu(
            marble_model, 
            train_data
        )
        log(f"\tDone.")

        # Instantiate optimizer and scheduler (after moving model to target device)
        optimizer = instantiate_optimizer(
            optimizer_cfg, 
            marble_model.parameters(), 
            args.lr,
        )
        if args.original_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            scheduler_mode = "plateau"
        else:
            scheduler, scheduler_mode = instantiate_scheduler(
                scheduler_cfg, 
                optimizer,
            )

        # Train MARBLE for a single fold with SVM-based validation monitoring/early stopping
        best_state, best_epoch, best_val_acc, history = run_training_epochs(
            fold_idx=fold_idx,
            marble_model=marble_model,
            train_data=train_data,
            loader=loader,
            fold_data=fold_data,
            max_epochs=max_epochs,
            patience=patience,
            scheduler=scheduler, 
            scheduler_mode=scheduler_mode,
            optimizer=optimizer,
            log=log,
            epoch_log=log,
            svm_kernel=args.svm_kernel,
            svm_C=args.svm_c,
            svm_gamma=args.svm_gamma,
            plateau_patience=plateau_patience,
            plateau_max_restarts=plateau_max_restarts,
            plateau_factor=plateau_factor,
            use_original_scheduler=args.original_scheduler,
        )

        # After training, evaluate and save the fold results
        results = evaluate_and_save_fold(
            fold_idx=fold_idx,
            marble_model=marble_model,
            train_data=train_data,
            fold_data=fold_data,
            optimizer=optimizer,
            best_state=best_state,
            best_epoch=best_epoch,
            best_val_acc=best_val_acc,
            history=history,
            embeddings_dir=embeddings_dir,
            metrics_dir=metrics_dir,
            models_dir=models_dir,
            patience=patience,
            max_epochs=max_epochs,
            log=log,
            svm_kernel=args.svm_kernel,
            svm_C=args.svm_c,
            svm_gamma=args.svm_gamma,
        )

        log(f"Completed fold {fold_idx}. Results saved to {fold_root}.")

    if len(fold_indices) > 1:
        compute_kfold_results(
            parent_metrics_dir=Path(exp_dirs["exp_dir"]) / "metrics",
            timing_keys=("mean_train_time", "mean_infer_time"),
            weight_key="best_epoch",
        )


if __name__ == "__main__":
    main()


