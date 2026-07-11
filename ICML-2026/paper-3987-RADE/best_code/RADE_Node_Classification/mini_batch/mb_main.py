import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import csv
import math
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb

from ..full_batch.data_utils import class_rand_splits, eval_acc_nc, eval_f1_nc, make_random_split_idx
from ..full_batch.dataset import load_nc_dataset
from ..full_batch.logger import Logger, save_model as _save_model, save_result as _save_result
from ..full_batch.viz_pq import save_pq_plots

from .mb_augmentation import BatchBernoulliEdgeAugmentor
from .mb_eval import evaluate_minibatch
from .mb_loaders import NeighborLoaderConfig, build_neighbor_loaders
from .mb_parse import (
    apply_mini_batch_aug_defaults,
    apply_mini_batch_auto_defaults,
    get_explicit_arg_names,
    parse_method,
    parser_add_main_args,
)
from .mb_pq_gradnorm import PQGradNormTunerMB, PQTunerMBConfig


MB_DIR = os.path.dirname(os.path.abspath(__file__))


def _mb_path(*parts: str) -> str:
    return os.path.join(MB_DIR, *parts)


def _call_from_mb_dir(func, *args, **kwargs):
    cwd = os.getcwd()
    os.chdir(MB_DIR)
    try:
        return func(*args, **kwargs)
    finally:
        os.chdir(cwd)


def save_model(args, model, optimizer, run):
    return _call_from_mb_dir(_save_model, args, model, optimizer, run)


def save_result(args, results, runtime_summary=None):
    return _call_from_mb_dir(_save_result, args, results, runtime_summary=runtime_summary)


def fix_seed(seed: int = 42, device: Optional[torch.device] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")


def _enforce_aug_mode(p_val: float, q_val: float, aug_mode: str) -> Tuple[float, float]:
    aug_mode = str(aug_mode).lower().strip()
    if aug_mode == "drop":
        return float(p_val), 0.0
    if aug_mode == "add":
        return 0.0, float(q_val)
    if aug_mode == "none":
        return 0.0, 0.0
    return float(p_val), float(q_val)


def _apply_p_cap(args, p_val: float) -> float:
    if not bool(getattr(args, "pq_cap_p", False)):
        return float(p_val)
    return float(min(float(p_val), float(getattr(args, "pq_p_max", 0.9))))


def _require_no_rade_contrib(args, name: str) -> None:
    if getattr(args, "ep_correction", False):
        raise ValueError(f"--aug_tech={name} must be used with --ep_correction False.")
    if getattr(args, "pq_gradnorm", False):
        raise ValueError(f"--aug_tech={name} must be used with --pq_gradnorm False.")
    if str(getattr(args, "aug_mode", "none")).lower().strip() != "none":
        raise ValueError(f"--aug_tech={name} requires --aug_mode none.")
    if getattr(args, "linear", False):
        raise ValueError(f"--aug_tech={name} is incompatible with --linear.")


def _batch_seed(base_seed: int, batch_id: int, layer_id: int = 0) -> int:
    return int(base_seed) + 1_000_003 * int(batch_id) + 100_000_019 * int(layer_id)


def _resolve_pq_batch_update_interval(args, train_loader) -> Tuple[int, Optional[int]]:
    num_batches: Optional[int]
    try:
        num_batches = int(len(train_loader))
    except TypeError:
        num_batches = None

    update_unit = str(getattr(args, "pq_update_unit", "batch")).lower().strip()
    if update_unit == "epoch":
        if num_batches is None or num_batches <= 0:
            raise ValueError("--pq_update_unit=epoch requires a train loader with a known positive length.")
        return int(num_batches), num_batches
    if update_unit != "batch":
        raise ValueError("--pq_update_unit must be one of {batch, epoch}.")

    fraction = float(getattr(args, "pq_update_batch_fraction", 0.0))
    if fraction > 0.0:
        if num_batches is None or num_batches <= 0:
            raise ValueError("--pq_update_batch_fraction requires a train loader with a known length.")
        return max(1, int(math.ceil(float(num_batches) * fraction))), num_batches

    return max(1, int(getattr(args, "pq_update_every_batches", 1))), num_batches


def _sync_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _first_nonfinite_gradient(model: nn.Module) -> Optional[Tuple[str, torch.Size, int]]:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        finite = torch.isfinite(param.grad)
        if not bool(finite.all()):
            return name, param.grad.shape, int((~finite).sum().item())
    return None


def _format_tag_float(value: float) -> str:
    return str(f"{float(value):g}").replace(".", "p").replace("-", "m")


def _safe_std(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    return float(values.std().item())


def _format_runtime_list(values) -> str:
    return ";".join(f"{float(value):.6f}" for value in values)


def _write_single_dataset_csv(record: Dict[str, object], out_path: str = "single_dataset/results_mb.csv") -> str:
    abs_path = os.path.abspath(out_path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fieldnames = [
        "timestamp",
        "dataset",
        "final_test_acc_mean_pct",
        "final_test_acc_std_pct",
        "runtime_run_mean_sec",
        "runtime_run_std_sec",
        "runtime_per_run_mean_sec",
        "runtime_per_run_std_sec",
        "runtime_per_run_num_epochs",
        "runtime_pooled_mean_sec",
        "runtime_num_runs",
        "runtime_num_epochs",
        "model",
        "gnn_raw",
        "gnn",
        "hidden_channels",
        "local_layers",
        "gat_heads",
        "gat_concat",
        "gat_negative_slope",
        "gat_moment_chunk_size",
        "gat_moment_samples",
        "gat_nonedge_samples",
        "gat_moment_seed",
        "gat_ep_corr_clip",
        "lr",
        "weight_decay",
        "dropout",
        "dropedge_rate",
        "dropnode_rate",
        "dropmessage_rate",
        "patience",
        "runs",
        "epochs",
        "seed",
        "device",
        "batch_size",
        "mb_num_workers",
        "mb_pin_memory",
        "full_eval_cpu",
        "graph_density",
        "pq_expected_add_ratio_scale",
        "aug_tech",
        "aug_mode",
        "p",
        "q",
        "ep_correction",
        "mask_sharing",
        "rade_variant",
        "pq_gradnorm",
        "pq_optimize_rho",
        "pq_cap_p",
        "pq_p_max",
        "pq_cap_rho",
        "pq_rho_max",
        "pq_deletion_penalty_lambda",
        "pq_densification_penalty_lambda",
        "pq_adam_lr",
        "pq_subset_nodes",
        "pq_update_every",
        "pq_update_unit",
        "pq_update_every_batches",
        "pq_update_batch_fraction",
        "pq_warmup_epochs",
        "pq_eps",
        "pq_seed",
        "skip_nonfinite_grad_step",
        "train_prop",
        "valid_prop",
        "rand_split",
        "rand_split_class",
        "label_num_per_class",
        "valid_num",
        "test_num",
    ]

    write_header = not os.path.exists(abs_path) or os.path.getsize(abs_path) == 0
    with open(abs_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in fieldnames})

    return abs_path


parser = argparse.ArgumentParser(description="Mini-batch Training Pipeline for Node Classification")
parser_add_main_args(parser)
args = parser.parse_args()
explicit_arg_names = get_explicit_arg_names(sys.argv[1:])

if getattr(args, "linear", False):
    args.dropout = 0.0
    if hasattr(args, "bn"):
        args.bn = False
    if hasattr(args, "ln"):
        args.ln = False

wandb_run = None
if getattr(args, "wandb", False):
    tags = [tag.strip() for tag in str(getattr(args, "wandb_tags", "")).split(",") if tag.strip()]
    wandb_run = wandb.init(
        project=str(getattr(args, "wandb_project", "rade-nc-mb")),
        entity=getattr(args, "wandb_entity", None) or None,
        group=getattr(args, "wandb_group", None) or None,
        name=getattr(args, "wandb_name", None) or None,
        tags=tags,
        mode=str(getattr(args, "wandb_mode", "online")),
    )

    for key, value in dict(wandb.config).items():
        if hasattr(args, key):
            setattr(args, key, value)
    explicit_arg_names.update({key for key in dict(wandb.config).keys() if hasattr(args, key)})

    if getattr(args, "linear", False):
        args.dropout = 0.0
        if hasattr(args, "bn"):
            args.bn = False
        if hasattr(args, "ln"):
            args.ln = False

gnn_raw = str(getattr(args, "gnn", "gcn")).lower().strip()
args.gnn_raw = gnn_raw

preset_key, auto_applied, auto_skipped = apply_mini_batch_auto_defaults(args, explicit_arg_names)
if preset_key is not None:
    if auto_applied:
        print(
            f"[AUTO-CONFIG] mini-batch preset {preset_key[0]}/{preset_key[1]} applied: "
            + ", ".join(f"{name}={value}" for name, value in auto_applied.items())
        )
    if auto_skipped:
        print(
            f"[AUTO-CONFIG] kept explicit overrides for {preset_key[0]}/{preset_key[1]}: "
            + ", ".join(f"{name}={value}" for name, value in auto_skipped.items())
        )

if gnn_raw == "sgc":
    args.gnn = "gcn"
    args.linear = True
elif gnn_raw == "gin-linear":
    args.gnn = "gin"
    args.linear = True
elif gnn_raw == "gat" and bool(getattr(args, "linear", False)):
    raise ValueError(
        "--linear is unavailable for --gnn gat because GAT attention is feature-dependent "
        "and nonlinear. Use --linear False, or use --gnn sgc / --gnn gin-linear for "
        "strict linear baselines."
    )

if getattr(args, "linear", False) and gnn_raw in {"sgc", "gin-linear"}:
    args.aug_tech = "none"
    args.aug_mode = "none"
    args.ep_correction = False
    args.pq_gradnorm = False
    args.dropout = 0.0
    if hasattr(args, "bn"):
        args.bn = False
    if hasattr(args, "ln"):
        args.ln = False

aug_preset_key, aug_auto_applied, aug_auto_skipped = apply_mini_batch_aug_defaults(args, explicit_arg_names)
if aug_preset_key is not None:
    if aug_auto_applied:
        print(
            f"[AUTO-CONFIG] mini-batch augmentation preset {aug_preset_key[0]}/{aug_preset_key[1]} applied: "
            + ", ".join(f"{name}={value}" for name, value in aug_auto_applied.items())
        )
    if aug_auto_skipped:
        print(
            f"[AUTO-CONFIG] kept explicit augmentation overrides for {aug_preset_key[0]}/{aug_preset_key[1]}: "
            + ", ".join(f"{name}={value}" for name, value in aug_auto_skipped.items())
        )

aug_tech = str(getattr(args, "aug_tech", "none")).lower().strip()
if aug_tech not in {"rade", "dropout", "dropedge", "dropmessage", "dropnode", "none"}:
    raise ValueError(
        f"--aug_tech must be one of {{rade, dropout, dropedge, dropmessage, dropnode, none}}. Got {aug_tech}"
    )

if aug_tech == "dropmessage":
    _require_no_rade_contrib(args, "dropmessage")
if aug_tech == "dropnode":
    _require_no_rade_contrib(args, "dropnode")
if aug_tech == "dropout":
    _require_no_rade_contrib(args, "dropout")
if aug_tech == "dropedge":
    if getattr(args, "ep_correction", False):
        raise ValueError("--aug_tech=dropedge must be used with --ep_correction False.")
    if getattr(args, "pq_gradnorm", False):
        raise ValueError("--aug_tech=dropedge must be used with --pq_gradnorm False.")
    if str(getattr(args, "aug_mode", "drop")).lower().strip() != "drop":
        raise ValueError("--aug_tech=dropedge requires --aug_mode drop.")
    args.q = 0.0
if aug_tech == "none":
    args.aug_mode = "none"

pq_update_unit = str(getattr(args, "pq_update_unit", "batch")).lower().strip()
if pq_update_unit not in {"batch", "epoch"}:
    raise ValueError("--pq_update_unit must be one of {batch, epoch}.")
args.pq_update_unit = pq_update_unit
if int(getattr(args, "pq_update_every_batches", 1)) <= 0:
    raise ValueError("--pq_update_every_batches must be >= 1.")
if float(getattr(args, "pq_update_batch_fraction", 0.0)) < 0.0:
    raise ValueError("--pq_update_batch_fraction must be >= 0.")
if float(getattr(args, "pq_update_batch_fraction", 0.0)) > 1.0:
    raise ValueError("--pq_update_batch_fraction must be <= 1.")
if bool(getattr(args, "pq_cap_p", False)) and not (0.0 <= float(getattr(args, "pq_p_max", 0.9)) < 1.0):
    raise ValueError(f"--pq_p_max must be in [0, 1) when --pq_cap_p is enabled. Got {args.pq_p_max}")
if bool(getattr(args, "pq_cap_rho", False)) and float(getattr(args, "pq_rho_max", 0.2)) < 0.0:
    raise ValueError(f"--pq_rho_max must be >= 0 when --pq_cap_rho is enabled. Got {args.pq_rho_max}")
if float(getattr(args, "gat_ep_corr_clip", 2.0)) < 0.0:
    raise ValueError(f"--gat_ep_corr_clip must be >= 0. Got {args.gat_ep_corr_clip}")

print(args)
print(
    f"[CONFIG] dataset={args.dataset} | backbone_raw={getattr(args, 'gnn_raw', args.gnn)} -> backbone={args.gnn} | "
    f"linear={bool(getattr(args, 'linear', False))} | aug_tech={args.aug_tech} aug_mode={args.aug_mode} | "
    f"ep_correction={bool(getattr(args, 'ep_correction', False))} "
    f"(mode=analytic, variant={getattr(args, 'rade_variant', '-')}) | "
    f"pq_gradnorm={bool(getattr(args, 'pq_gradnorm', False))} "
    f"(search=adam, "
    f"update_unit={getattr(args, 'pq_update_unit', 'batch')}, "
    f"batch_update_every={int(getattr(args, 'pq_update_every_batches', 1))}, "
    f"batch_update_fraction={float(getattr(args, 'pq_update_batch_fraction', 0.0)):g}, "
    f"rho_opt={bool(getattr(args, 'pq_optimize_rho', False))}, "
    f"p_cap={'ON' if bool(getattr(args, 'pq_cap_p', False)) else 'OFF'}"
    f"{'@' + str(float(getattr(args, 'pq_p_max', 0.9))) if bool(getattr(args, 'pq_cap_p', False)) else ''}, "
    f"rho_cap={'ON' if bool(getattr(args, 'pq_cap_rho', False)) else 'OFF'}"
    f"{'@' + str(float(getattr(args, 'pq_rho_max', 0.2))) if bool(getattr(args, 'pq_cap_rho', False)) else ''}, "
    f"del_penalty={float(getattr(args, 'pq_deletion_penalty_lambda', 0.0)):g}, "
    f"dens_penalty=quadratic:{float(getattr(args, 'pq_densification_penalty_lambda', 0.0)):g}) | "
    f"p={float(getattr(args, 'p', 0.0)):.4f} q={float(getattr(args, 'q', 0.0)):.6f} | "
    f"mask_sharing={getattr(args, 'mask_sharing', '-')} | "
    f"gat_corr_clip={float(getattr(args, 'gat_ep_corr_clip', 2.0)):g}"
)

if wandb_run is not None:
    wandb.config.update(vars(args), allow_val_change=True)

device = torch.device("cpu") if args.cpu else (
    torch.device(f"cuda:{int(args.device)}") if torch.cuda.is_available() else torch.device("cpu")
)
fix_seed(int(args.seed), device=device)

data, base_split_idx = load_nc_dataset(
    root=args.data_dir,
    name=args.dataset,
    normalize_features=True,
    make_undirected_edges=True,
    seed=int(args.seed),
    train_prop=float(args.train_prop),
    valid_prop=float(args.valid_prop),
)
data.n_id = torch.arange(data.num_nodes, dtype=torch.long)

n = int(data.num_nodes)
d = int(data.x.size(1))
c = int(data.y.max().item()) + 1
e_undir = int(data.edge_index.size(1)) // 2 if data.edge_index is not None else 0
total_undir_pairs = max(float(n) * float(max(n - 1, 0)) / 2.0, 1.0)
graph_density = float(e_undir) / total_undir_pairs
rho_scale = 0.0 if graph_density <= 0.0 else 1.0 / graph_density

print(f"dataset {args.dataset} | num nodes {n} | num edges {e_undir} | num feats {d} | num classes {c}")
print(f"[PQ] graph_density={graph_density:.6e} | rho_scale=1/d={rho_scale:.6e}")
if wandb_run is not None:
    wandb.config.update(
        {
            "num_nodes": n,
            "num_edges": e_undir,
            "num_feats": d,
            "num_classes": c,
            "graph_density": graph_density,
            "pq_expected_add_ratio_scale": rho_scale,
        },
        allow_val_change=True,
    )

split_idx_lst: List[Dict[str, torch.Tensor]] = []
for run in range(int(args.runs)):
    if getattr(args, "rand_split", False):
        split_idx = make_random_split_idx(n, float(args.train_prop), float(args.valid_prop), seed=int(args.seed) + run)
    elif getattr(args, "rand_split_class", False):
        split_idx = class_rand_splits(
            data.y,
            int(getattr(args, "label_num_per_class", 20)),
            valid_num=int(getattr(args, "valid_num", 500)),
            test_num=int(getattr(args, "test_num", 1000)),
            seed=int(args.seed) + run,
        )
    else:
        split_idx = base_split_idx

    split_idx_lst.append({key: value.to(torch.long).cpu() for key, value in split_idx.items()})

model = parse_method(args, n, c, d, device)
criterion = nn.CrossEntropyLoss()
eval_func = eval_f1_nc if args.dataset.lower() == "flickr" else eval_acc_nc
logger = Logger(int(args.runs), args)
print("MODEL:", model)

num_workers = int(getattr(args, "mb_num_workers", 0))
batch_size_train = int(getattr(args, "batch_size", 2048))
default_num_neighbors = [15, 10, 5]
num_layers = int(getattr(args, "local_layers", 1))
if len(default_num_neighbors) < num_layers:
    default_num_neighbors += [default_num_neighbors[-1]] * (num_layers - len(default_num_neighbors))
default_num_neighbors = tuple(default_num_neighbors[:num_layers])

nl_cfg = NeighborLoaderConfig(
    num_neighbors=default_num_neighbors,
    batch_size_train=batch_size_train,
    batch_size_eval=max(1024, batch_size_train * 2),
    shuffle_train=bool(getattr(args, "mb_shuffle", False)),
    num_workers=num_workers,
    pin_memory=bool(getattr(args, "mb_pin_memory", False)),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2,
)

p_histories: List[List[float]] = []
q_histories: List[List[float]] = []
obj_histories: List[List[float]] = []
rho_histories: List[List[float]] = []

for run in range(int(args.runs)):
    split_idx = split_idx_lst[run]
    loaders = build_neighbor_loaders(data, split_idx, cfg=nl_cfg)
    pq_batch_update_interval, train_batches_per_epoch = _resolve_pq_batch_update_interval(args, loaders["train"])

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    p_eff, q_eff = _enforce_aug_mode(float(getattr(args, "p", 0.0)), float(getattr(args, "q", 0.0)), args.aug_mode)
    p_eff = _apply_p_cap(args, p_eff)

    pq_tuner: Optional[PQGradNormTunerMB] = None
    if bool(getattr(args, "pq_gradnorm", False)):
        if str(getattr(args, "gnn", "gcn")).lower().strip() == "gat":
            if int(getattr(args, "gat_heads", 1)) != 1:
                raise ValueError("GAT PQ-GradNorm currently supports only --gat_heads 1.")
        if aug_tech != "rade":
            raise ValueError("--pq_gradnorm is only supported when --aug_tech=rade in mini-batch.")
        if str(args.aug_mode).lower().strip() == "none":
            raise ValueError("--pq_gradnorm requires --aug_mode != none.")
        if not bool(getattr(args, "ep_correction", False)):
            raise ValueError("--pq_gradnorm currently requires --ep_correction True.")

        pq_tuner = PQGradNormTunerMB(
            gnn=str(getattr(args, "gnn", "gin")),
            rade_variant=str(getattr(args, "rade_variant", "rade-of")),
            cfg=PQTunerMBConfig(
                expected_add_ratio_scale=float(rho_scale),
                deletion_penalty_lambda=float(getattr(args, "pq_deletion_penalty_lambda", 0.0)),
                densification_penalty_lambda=float(getattr(args, "pq_densification_penalty_lambda", 0.0)),
                optimize_rho=bool(getattr(args, "pq_optimize_rho", False)),
                cap_p=bool(getattr(args, "pq_cap_p", False)),
                p_cap_max=float(getattr(args, "pq_p_max", 0.9)),
                cap_rho=bool(getattr(args, "pq_cap_rho", False)),
                rho_max=float(getattr(args, "pq_rho_max", 0.2)),
                eps=float(getattr(args, "pq_eps", 1e-12)),
                subset_nodes=int(getattr(args, "pq_subset_nodes", 1024)),
                seed=int(getattr(args, "pq_seed", 0)),
                adam_lr=float(getattr(args, "pq_adam_lr", 0.2)),
                adam_beta1=float(getattr(args, "pq_adam_beta1", 0.9)),
                adam_beta2=float(getattr(args, "pq_adam_beta2", 0.999)),
                adam_eps=float(getattr(args, "pq_adam_eps", 1e-8)),
            ),
        )
        q_eff = float(max(0.0, min(q_eff, float(pq_tuner.q_probability_upper))))

    best_val = float("-inf")
    best_test = float("-inf")
    patience = int(getattr(args, "patience", 100))
    epochs_no_improve = 0
    best_epoch = -1

    p_trace: List[float] = []
    q_trace: List[float] = []
    obj_trace: List[float] = []
    rho_trace: List[float] = []

    if bool(getattr(args, "save_model", False)):
        save_model(args, model, optimizer, run)

    for epoch in range(int(args.epochs)):
        _sync_for_timing(device)
        epoch_time = time.perf_counter()
        model.train()

        p_epoch_start = float(p_eff)
        q_epoch_start = float(q_eff)
        p_trace.append(p_epoch_start)
        q_trace.append(q_epoch_start)
        obj_trace.append(float("nan"))
        rho_trace.append(float(q_epoch_start) * float(rho_scale))

        epoch_loss_sum = 0.0
        epoch_seed_nodes = 0
        epoch_dropped_sum = 0.0
        epoch_added_sum = 0.0
        epoch_aug_batches = 0
        skipped_optimizer_steps = 0

        warmup = int(getattr(args, "pq_warmup_epochs", 0))
        every = int(getattr(args, "pq_update_every", 1))
        do_pq_update = (pq_tuner is not None) and ((epoch + 1) >= warmup) and (((epoch + 1) % every) == 0)
        do_pq_batchwise_adam = do_pq_update and (pq_tuner is not None)

        base_seed = int(args.seed) + 10_000 * int(run) + int(epoch)
        pq_epoch_seed = int(args.seed) + 100_000 * int(run) + int(epoch) + int(getattr(args, "pq_seed", 0))
        adam_batch_infos: List[Dict[str, Any]] = []

        for batch_id, batch in enumerate(loaders["train"]):
            batch = batch.to(device)
            x_b = batch.x
            edge_index_clean_b = batch.edge_index
            y_b = batch.y
            node_ids_b = getattr(batch, "n_id", None)
            p_batch = float(p_eff)
            q_batch = float(q_eff)

            seed_n = int(getattr(batch, "batch_size", x_b.size(0)))
            if seed_n <= 0:
                continue

            y_seed = y_b[:seed_n]
            if y_seed.dim() == 2 and y_seed.size(1) == 1:
                y_seed = y_seed.squeeze(1)
            y_seed = y_seed.view(-1).to(torch.long)
            train_mask_b = torch.zeros((x_b.size(0),), device=x_b.device, dtype=torch.bool)
            train_mask_b[:seed_n] = True

            edge_index_keep = None
            edge_index_add = None
            edge_index_train = edge_index_clean_b
            aug_stats = None

            dropmessage_rate = float(getattr(args, "dropmessage_rate", 0.0)) if aug_tech == "dropmessage" else 0.0
            dropnode_rate = float(getattr(args, "dropnode_rate", 0.0)) if aug_tech == "dropnode" else 0.0

            do_aug = (
                aug_tech in {"rade", "dropedge"}
                and str(getattr(args, "aug_mode", "none")).lower().strip() != "none"
                and (p_batch > 0.0 or q_batch > 0.0)
            )

            if do_aug:
                if aug_tech == "dropedge":
                    edge_index_train, _, _, aug_stats = BatchBernoulliEdgeAugmentor.augment_edge_indices(
                        edge_index_clean=edge_index_clean_b,
                        num_nodes=int(x_b.size(0)),
                        p=float(p_batch),
                        q=0.0,
                        mode="drop",
                        seed=_batch_seed(base_seed, batch_id, 0),
                        device=device,
                    )
                elif str(getattr(args, "mask_sharing", "shared")).lower().strip() == "shared":
                    _, edge_index_keep, edge_index_add, aug_stats = BatchBernoulliEdgeAugmentor.augment_edge_indices(
                        edge_index_clean=edge_index_clean_b,
                        num_nodes=int(x_b.size(0)),
                        p=float(p_batch),
                        q=float(q_batch),
                        mode=str(getattr(args, "aug_mode", "both")),
                        seed=_batch_seed(base_seed, batch_id, 0),
                        device=device,
                    )
                else:
                    edge_index_keep = []
                    edge_index_add = []
                    stats_list = []
                    for layer in range(int(getattr(args, "local_layers", 1))):
                        _, edge_index_keep_l, edge_index_add_l, stats_l = BatchBernoulliEdgeAugmentor.augment_edge_indices(
                            edge_index_clean=edge_index_clean_b,
                            num_nodes=int(x_b.size(0)),
                            p=float(p_batch),
                            q=float(q_batch),
                            mode=str(getattr(args, "aug_mode", "both")),
                            seed=_batch_seed(base_seed, batch_id, layer),
                            device=device,
                        )
                        edge_index_keep.append(edge_index_keep_l)
                        edge_index_add.append(edge_index_add_l)
                        stats_list.append(stats_l)

                    aug_stats = {
                        "dropped_undir_avg": float(sum(stats["dropped_undir"] for stats in stats_list)) / float(len(stats_list)),
                        "added_undir_avg": float(sum(stats["added_undir"] for stats in stats_list)) / float(len(stats_list)),
                    }

            if aug_stats is not None:
                if "dropped_undir" in aug_stats and "added_undir" in aug_stats:
                    epoch_dropped_sum += float(aug_stats["dropped_undir"])
                    epoch_added_sum += float(aug_stats["added_undir"])
                    epoch_aug_batches += 1
                elif "dropped_undir_avg" in aug_stats and "added_undir_avg" in aug_stats:
                    epoch_dropped_sum += float(aug_stats["dropped_undir_avg"])
                    epoch_added_sum += float(aug_stats["added_undir_avg"])
                    epoch_aug_batches += 1

            optimizer.zero_grad(set_to_none=True)

            if aug_tech == "dropmessage":
                logits_b = model(
                    x_b,
                    edge_index_clean_b,
                    dropmessage_rate=dropmessage_rate,
                    node_ids=node_ids_b,
                )
            elif aug_tech == "dropnode":
                logits_b = model(
                    x_b,
                    edge_index_clean_b,
                    dropnode_rate=dropnode_rate,
                    node_ids=node_ids_b,
                )
            elif aug_tech == "dropedge" and do_aug:
                logits_b = model(x_b, edge_index_train, node_ids=node_ids_b)
            elif aug_tech == "rade" and do_aug:
                logits_b = model(
                    x_b,
                    edge_index_clean_b,
                    edge_index_keep=edge_index_keep,
                    edge_index_add=edge_index_add,
                    p=float(p_batch),
                    q=float(q_batch),
                    node_ids=node_ids_b,
                )
            else:
                logits_b = model(x_b, edge_index_clean_b, node_ids=node_ids_b)

            loss_b = criterion(logits_b[:seed_n], y_seed)
            skipped_optimizer_step = False
            if bool(getattr(args, "skip_nonfinite_grad_step", True)):
                if not bool(torch.isfinite(loss_b).all()):
                    print(
                        f"[NonfiniteGuard-MB] run={run + 1:02d} epoch={epoch + 1:03d} "
                        f"batch={batch_id:04d} loss={float(loss_b.detach().cpu())}; skipped optimizer step."
                    )
                    skipped_optimizer_step = True
                    optimizer.zero_grad(set_to_none=True)
                else:
                    loss_b.backward()
                    bad_grad = _first_nonfinite_gradient(model)
                    if bad_grad is not None:
                        name, shape, count = bad_grad
                        print(
                            f"[NonfiniteGuard-MB] run={run + 1:02d} epoch={epoch + 1:03d} "
                            f"batch={batch_id:04d} nonfinite gradient in {name} "
                            f"shape={tuple(shape)} count={count}; skipped optimizer step."
                        )
                        skipped_optimizer_step = True
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        optimizer.step()
            else:
                loss_b.backward()
                optimizer.step()
            if skipped_optimizer_step:
                skipped_optimizer_steps += 1

            epoch_loss_sum += float(loss_b.item()) * seed_n
            epoch_seed_nodes += seed_n

            if (
                do_pq_batchwise_adam
                and not skipped_optimizer_step
                and (((int(batch_id) + 1) % int(pq_batch_update_interval)) == 0)
            ):
                p_next, q_next, info_b = pq_tuner.suggest_pq_for_batch(
                    model=model,
                    x_b=x_b,
                    edge_index_b_clean=edge_index_clean_b,
                    y_b=y_b,
                    train_mask_b=train_mask_b,
                    criterion=criterion,
                    current_p=float(p_batch),
                    current_q=float(q_batch),
                    aug_mode=str(getattr(args, "aug_mode", "both")).lower().strip(),
                    epoch_seed=int(pq_epoch_seed),
                    batch_id=int(batch_id),
                    node_ids_b=node_ids_b,
                    mask_sharing=str(getattr(args, "mask_sharing", "shared")).lower().strip(),
                    num_layers=int(getattr(args, "local_layers", 1)),
                )
                if not bool(info_b.get("skip", 0.0)):
                    p_eff, q_eff = _enforce_aug_mode(p_next, q_next, args.aug_mode)
                    p_eff = _apply_p_cap(args, p_eff)
                    q_eff = float(max(0.0, min(q_eff, float(pq_tuner.q_probability_upper))))
                    adam_batch_infos.append(info_b)

        avg_train_loss = epoch_loss_sum / max(epoch_seed_nodes, 1)
        p_eval_epoch = float(p_eff)
        q_eval_epoch = float(q_eff)

        model.eval()
        forward_eval: Dict[str, Any] = {}
        if aug_tech == "dropmessage":
            forward_eval = {"dropmessage_rate": 0.0}
        elif aug_tech == "dropnode":
            forward_eval = {"dropnode_rate": 0.0}
        elif (
            aug_tech == "rade"
            and bool(getattr(args, "ep_correction", False))
            and str(getattr(args, "rade_variant", "rade-of")).lower().strip() == "rade-ofs"
            and float(q_eval_epoch) > 0.0
        ):
            forward_eval = {"p": float(p_eval_epoch), "q": float(q_eval_epoch)}

        if bool(getattr(args, "full_eval_cpu", False)) and device.type == "cuda":
            model.to(torch.device("cpu"))
            tr, va, te, va_loss, _ = evaluate_minibatch(
                model=model,
                data=data,
                loaders=loaders,
                split_idx=split_idx,
                eval_func=eval_func,
                criterion=criterion,
                device=torch.device("cpu"),
                forward_kwargs=forward_eval,
            )
            model.to(device)
            torch.cuda.empty_cache()
        else:
            tr, va, te, va_loss, _ = evaluate_minibatch(
                model=model,
                data=data,
                loaders=loaders,
                split_idx=split_idx,
                eval_func=eval_func,
                criterion=criterion,
                device=device,
                forward_kwargs=forward_eval,
            )

        result = (float(tr), float(va), float(te), float(va_loss), None)
        logger.add_result(run, result[:4])

        improved = result[1] > best_val
        if improved:
            best_val = float(result[1])
            best_test = float(result[2])
            best_epoch = int(epoch)
            epochs_no_improve = 0
            if bool(getattr(args, "save_model", False)):
                save_model(args, model, optimizer, run)
        else:
            epochs_no_improve += 1

        if do_pq_batchwise_adam and adam_batch_infos:
            def _mean_info(key: str) -> float:
                values = [float(item[key]) for item in adam_batch_infos if key in item]
                return float(sum(values) / max(len(values), 1))

            p_trace[-1] = float(p_eff)
            q_trace[-1] = float(q_eff)
            rho_trace[-1] = float(q_eff) * float(rho_scale)
            obj_trace[-1] = _mean_info("obj_new")

            print(
                f"[PQ-GradNorm-MB-Adam] run={run + 1:02d} epoch={epoch + 1:03d} "
                f"updates={len(adam_batch_infos):03d} "
                f"interval={int(pq_batch_update_interval)} "
                f"p(last)={float(p_eff):.4f} q(last)={float(q_eff):.6f} "
                f"rho(last)={float(q_eff) * float(rho_scale):.3e} "
                f"G_data_mean={_mean_info('G_data'):.3e} "
                f"G_reg_mean={_mean_info('G_reg_best'):.3e} "
                f"obj_mean={_mean_info('obj'):.2e}"
            )

            if wandb_run is not None:
                wandb.log(
                    {
                        "pq/G_data_mean": _mean_info("G_data"),
                        "pq/G_reg_best_mean": _mean_info("G_reg_best"),
                        "pq/G_reg_new_mean": _mean_info("G_reg_new"),
                        "pq/obj_mean": _mean_info("obj"),
                        "pq/obj_new_mean": _mean_info("obj_new"),
                        "pq/p_epoch_best": _mean_info("p_best"),
                        "pq/q_epoch_best": _mean_info("q_best"),
                        "pq/p_new": float(p_eff),
                        "pq/q_new": float(q_eff),
                        "pq/rho_new": float(q_eff) * float(rho_scale),
                        "pq/obj_selected": _mean_info("obj_new"),
                        "pq/used_batches": float(len(adam_batch_infos)),
                        "pq/skipped_batches": float(max(0, (train_batches_per_epoch or 0) - len(adam_batch_infos))),
                        "pq/update_unit": str(getattr(args, "pq_update_unit", "batch")),
                        "pq/update_every_batches": float(pq_batch_update_interval),
                        "pq/update_batch_fraction": float(getattr(args, "pq_update_batch_fraction", 0.0)),
                        "pq/adam_lr": float(getattr(args, "pq_adam_lr", 0.2)),
                        "pq/optimize_rho": float(bool(getattr(args, "pq_optimize_rho", False))),
                        "pq/p_cap_enabled": float(bool(getattr(args, "pq_cap_p", False))),
                        "pq/p_probability_upper": float(getattr(pq_tuner, "p_probability_upper", float("nan"))),
                        "pq/rho_cap_enabled": float(bool(getattr(args, "pq_cap_rho", False))),
                        "pq/rho_probability_upper": float(getattr(pq_tuner, "rho_upper", float("nan"))),
                        "pq/q_probability_upper": float(getattr(pq_tuner, "q_probability_upper", float("nan"))),
                        "pq/deletion_penalty_best_mean": _mean_info("deletion_penalty_best"),
                        "pq/deletion_penalty_new_mean": _mean_info("deletion_penalty_new"),
                        "pq/densification_penalty_best_mean": _mean_info("densification_penalty_best"),
                        "pq/densification_penalty_new_mean": _mean_info("densification_penalty_new"),
                    },
                    step=(epoch + run * int(args.epochs)),
                )

        _sync_for_timing(device)
        epoch_time = time.perf_counter() - epoch_time
        logger.add_runtime(run, epoch_time)

        if wandb_run is not None and (epoch % int(getattr(args, "wandb_log_every", 1)) == 0):
            step = epoch + run * int(args.epochs)
            log_dict = {
                "aug/tech": str(aug_tech),
                "run": int(run + 1),
                "epoch": int(epoch + 1),
                "train/loss_mb": float(avg_train_loss),
                "train/acc": float(result[0]),
                "valid/acc": float(result[1]),
                "test/acc": float(result[2]),
                "valid/loss": float(result[3]),
                "best/valid_acc": float(best_val),
                "best/test_acc_at_best_valid": float(best_test),
                "runtime/epoch_sec": float(epoch_time),
                "train/skipped_optimizer_steps": float(skipped_optimizer_steps),
            }

            if aug_tech == "rade":
                log_dict["aug/p"] = float(p_eval_epoch)
                log_dict["aug/q"] = float(q_eval_epoch)
                log_dict["aug/rho"] = float(q_eval_epoch) * float(rho_scale)
                log_dict["aug/rho_next"] = float(q_eff) * float(rho_scale)
                log_dict["rade/ep_correction"] = bool(getattr(args, "ep_correction", False))
                log_dict["rade/variant"] = str(getattr(args, "rade_variant", "rade-of"))
                if epoch_aug_batches > 0:
                    log_dict["aug/dropped_undir_mean_per_batch"] = float(epoch_dropped_sum) / float(epoch_aug_batches)
                    log_dict["aug/added_undir_mean_per_batch"] = float(epoch_added_sum) / float(epoch_aug_batches)
            elif aug_tech == "dropmessage":
                log_dict["aug/dropmessage_rate"] = float(getattr(args, "dropmessage_rate", 0.0))
            elif aug_tech == "dropnode":
                log_dict["aug/dropnode_rate"] = float(getattr(args, "dropnode_rate", 0.0))
            elif aug_tech == "dropedge":
                log_dict["aug/dropedge_rate"] = float(p_eval_epoch)
                if epoch_aug_batches > 0:
                    log_dict["aug/dropped_undir_mean_per_batch"] = float(epoch_dropped_sum) / float(epoch_aug_batches)
                    log_dict["aug/added_undir_mean_per_batch"] = float(epoch_added_sum) / float(epoch_aug_batches)
            elif aug_tech == "dropout":
                log_dict["regularization/dropout"] = float(getattr(args, "dropout", 0.0))

            wandb.log(log_dict, step=step)

        if epoch % int(getattr(args, "display_step", 1)) == 0:
            aug_str = ""
            if aug_tech == "rade":
                if epoch_aug_batches > 0:
                    d_mean = float(epoch_dropped_sum) / float(epoch_aug_batches)
                    a_mean = float(epoch_added_sum) / float(epoch_aug_batches)
                    aug_str = f", Drop/Add(mean per batch): -{d_mean:.2f} +{a_mean:.2f} undir"
                aug_str += (
                    f" (p={p_eval_epoch:.4f}, q={q_eval_epoch:.6f}, "
                    f"rho={float(q_eval_epoch) * float(rho_scale):.3e}, "
                    f"ep_correction={bool(getattr(args, 'ep_correction', False))}, "
                    f"variant={getattr(args, 'rade_variant', 'rade-of')})"
                )
            elif aug_tech == "dropmessage":
                aug_str = f" (dropmessage={float(getattr(args, 'dropmessage_rate', 0.0)):.3f})"
            elif aug_tech == "dropnode":
                aug_str = f" (dropnode={float(getattr(args, 'dropnode_rate', 0.0)):.3f})"
            elif aug_tech == "dropedge":
                aug_str = f" (dropedge={p_eval_epoch:.3f})"
                if epoch_aug_batches > 0:
                    d_mean = float(epoch_dropped_sum) / float(epoch_aug_batches)
                    aug_str += f", DropEdge(mean per batch): -{d_mean:.2f} undir"
            elif aug_tech == "dropout" and float(getattr(args, "dropout", 0.0)) > 0.0:
                aug_str = f" (dropout={float(getattr(args, 'dropout', 0.0)):.3f})"

            if skipped_optimizer_steps > 0:
                aug_str += f", skipped_optimizer_steps={skipped_optimizer_steps}"
            aug_str += f", Epoch Time: {epoch_time:.4f}s"
            print(
                f"Run: {run + 1:02d}, Epoch: {epoch + 1:03d}, TrainLoss: {avg_train_loss:.4f}, "
                f"Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}%, "
                f"Best Valid: {100 * best_val:.2f}%, Best Test: {100 * best_test:.2f}%{aug_str}"
            )

        if patience > 0 and epochs_no_improve >= patience:
            stop_epoch = epoch + 1
            best_epoch_disp = best_epoch + 1 if best_epoch >= 0 else -1
            print(
                f"[EarlyStop] run={run + 1:02d} stopping at epoch {stop_epoch:03d} "
                f"(no val improvement for {patience} epochs). "
                f"Best val={best_val:.4f} at epoch {best_epoch_disp:03d}, test@best={best_test:.4f}."
            )
            if wandb_run is not None:
                step = epoch + run * int(args.epochs)
                wandb.log(
                    {
                        "early_stop/patience": int(patience),
                        "early_stop/stop_epoch": int(stop_epoch),
                        "early_stop/best_epoch": int(best_epoch_disp),
                        "early_stop/best_val": float(best_val),
                        "early_stop/best_test_at_best_val": float(best_test),
                    },
                    step=step,
                )
            break

    logger.print_statistics(run)
    p_histories.append(p_trace)
    q_histories.append(q_trace)
    obj_histories.append(obj_trace)
    rho_histories.append(rho_trace)


results = logger.print_statistics()
runtime_summary = logger.get_runtime_summary()

if getattr(args, "pq_gradnorm", False) and str(getattr(args, "aug_tech", "rade")).lower().strip() == "rade":
    visualization_dir = _mb_path("visualization")
    dataset_tag = str(args.dataset).lower().strip()
    gnn_tag = str(getattr(args, "gnn_raw", getattr(args, "gnn", "gcn"))).lower().strip()
    variant_tag = str(getattr(args, "rade_variant", "rade-of")).lower().strip().replace("-", "")
    adam_lr_tag = _format_tag_float(float(getattr(args, "pq_adam_lr", 0.0)))
    search_tag = f"adam_lr{adam_lr_tag}"
    if bool(getattr(args, "pq_optimize_rho", False)):
        search_tag = f"{search_tag}_rhoopt"
    lam_p_tag = f"plam{_format_tag_float(float(getattr(args, 'pq_deletion_penalty_lambda', 0.0)))}"
    lam_tag = f"lam{_format_tag_float(float(getattr(args, 'pq_densification_penalty_lambda', 0.0)))}"
    penalty_tag = "quadratic"
    tag = f"{dataset_tag}_{gnn_tag}_{variant_tag}_{search_tag}_{lam_p_tag}_{lam_tag}_{penalty_tag}_mb"

    save_pq_plots(
        p_histories,
        q_histories,
        obj_histories,
        rho_histories,
        tag=tag,
        out_dir=visualization_dir,
        show_runs=True,
    )

save_result(args, results, runtime_summary=runtime_summary)

single_dataset_record = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": str(args.dataset),
    "final_test_acc_mean_pct": float(results.mean().item()),
    "final_test_acc_std_pct": _safe_std(results),
    "runtime_run_mean_sec": "" if runtime_summary is None else float(runtime_summary["run_mean"]),
    "runtime_run_std_sec": "" if runtime_summary is None else float(runtime_summary["run_std"]),
    "runtime_per_run_mean_sec": "" if runtime_summary is None else _format_runtime_list(runtime_summary["run_means"]),
    "runtime_per_run_std_sec": "" if runtime_summary is None else _format_runtime_list(runtime_summary["run_stds"]),
    "runtime_per_run_num_epochs": "" if runtime_summary is None else ";".join(
        str(int(value)) for value in runtime_summary["run_epochs"]
    ),
    "runtime_pooled_mean_sec": "" if runtime_summary is None else float(runtime_summary["pooled_mean"]),
    "runtime_num_runs": "" if runtime_summary is None else int(runtime_summary["num_runs"]),
    "runtime_num_epochs": "" if runtime_summary is None else int(runtime_summary["num_epochs"]),
    "model": str(args.model),
    "gnn_raw": str(getattr(args, "gnn_raw", args.gnn)),
    "gnn": str(args.gnn),
    "hidden_channels": int(args.hidden_channels),
    "local_layers": int(args.local_layers),
    "gat_heads": int(getattr(args, "gat_heads", 1)),
    "gat_concat": bool(getattr(args, "gat_concat", True)),
    "gat_negative_slope": float(getattr(args, "gat_negative_slope", 0.2)),
    "gat_moment_chunk_size": int(getattr(args, "gat_moment_chunk_size", 1024)),
    "gat_moment_samples": int(getattr(args, "gat_moment_samples", 256)),
    "gat_nonedge_samples": int(getattr(args, "gat_nonedge_samples", 256)),
    "gat_moment_seed": int(getattr(args, "gat_moment_seed", 0)),
    "gat_ep_corr_clip": float(getattr(args, "gat_ep_corr_clip", 2.0)),
    "lr": float(args.lr),
    "weight_decay": float(args.weight_decay),
    "dropout": float(args.dropout),
    "dropedge_rate": float(getattr(args, "dropedge_rate", 0.0)),
    "dropnode_rate": float(getattr(args, "dropnode_rate", 0.0)),
    "dropmessage_rate": float(getattr(args, "dropmessage_rate", 0.0)),
    "patience": int(args.patience),
    "runs": int(args.runs),
    "epochs": int(args.epochs),
    "seed": int(args.seed),
    "device": str(device),
    "batch_size": int(args.batch_size),
    "mb_num_workers": int(args.mb_num_workers),
    "mb_pin_memory": bool(args.mb_pin_memory),
    "full_eval_cpu": bool(args.full_eval_cpu),
    "graph_density": float(graph_density),
    "pq_expected_add_ratio_scale": float(rho_scale),
    "aug_tech": str(args.aug_tech),
    "aug_mode": str(args.aug_mode),
    "p": float(args.p),
    "q": float(args.q),
    "ep_correction": bool(args.ep_correction),
    "mask_sharing": str(args.mask_sharing),
    "rade_variant": str(args.rade_variant),
    "pq_gradnorm": bool(args.pq_gradnorm),
    "pq_optimize_rho": bool(args.pq_optimize_rho),
    "pq_cap_p": bool(args.pq_cap_p),
    "pq_p_max": float(args.pq_p_max),
    "pq_cap_rho": bool(args.pq_cap_rho),
    "pq_rho_max": float(args.pq_rho_max),
    "pq_deletion_penalty_lambda": float(args.pq_deletion_penalty_lambda),
    "pq_densification_penalty_lambda": float(args.pq_densification_penalty_lambda),
    "pq_adam_lr": float(args.pq_adam_lr),
    "pq_subset_nodes": int(args.pq_subset_nodes),
    "pq_update_every": int(args.pq_update_every),
    "pq_update_unit": str(args.pq_update_unit),
    "pq_update_every_batches": int(args.pq_update_every_batches),
    "pq_update_batch_fraction": float(args.pq_update_batch_fraction),
    "pq_warmup_epochs": int(args.pq_warmup_epochs),
    "pq_eps": float(args.pq_eps),
    "pq_seed": int(args.pq_seed),
    "skip_nonfinite_grad_step": bool(args.skip_nonfinite_grad_step),
    "train_prop": float(args.train_prop),
    "valid_prop": float(args.valid_prop),
    "rand_split": bool(args.rand_split),
    "rand_split_class": bool(args.rand_split_class),
    "label_num_per_class": int(args.label_num_per_class),
    "valid_num": int(args.valid_num),
    "test_num": int(args.test_num),
}
single_dataset_dataset_tag = str(args.dataset).lower().strip().replace(os.sep, "_").replace("/", "_")
single_dataset_gnn_tag = (
    str(getattr(args, "gnn_raw", getattr(args, "gnn", "gcn")))
    .lower()
    .strip()
    .replace(os.sep, "_")
    .replace("/", "_")
)
single_dataset_csv = _write_single_dataset_csv(
    single_dataset_record,
    out_path=_mb_path("single_dataset", f"results_mb_{single_dataset_dataset_tag}_{single_dataset_gnn_tag}.csv"),
)
print(f"[SINGLE-DATASET-MB] Saved summary CSV row to {single_dataset_csv}")

if wandb_run is not None:
    try:
        wandb_run.summary["final_test_mean"] = float(results.mean().item())
        wandb_run.summary["final_test_std"] = float(results.std().item())
        if runtime_summary is not None:
            wandb_run.summary["runtime/profile"] = "all_mini_batch_runs"
            wandb_run.summary["runtime/epoch_sec_run_mean"] = float(runtime_summary["run_mean"])
            wandb_run.summary["runtime/epoch_sec_run_std"] = float(runtime_summary["run_std"])
            wandb_run.summary["runtime/epoch_sec_pooled_mean"] = float(runtime_summary["pooled_mean"])
            for idx, (run_mean, run_std, run_epochs) in enumerate(
                zip(
                    runtime_summary["run_means"],
                    runtime_summary["run_stds"],
                    runtime_summary["run_epochs"],
                ),
                start=1,
            ):
                wandb_run.summary[f"runtime/run_{idx:02d}_epoch_sec_mean"] = float(run_mean)
                wandb_run.summary[f"runtime/run_{idx:02d}_epoch_sec_std"] = float(run_std)
                wandb_run.summary[f"runtime/run_{idx:02d}_num_epochs"] = int(run_epochs)
    except Exception:
        pass
    wandb.finish()
