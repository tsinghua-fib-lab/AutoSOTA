import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import csv
import random
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
except Exception:
    wandb = None

from logger import Logger, save_model, save_result
from dataset import load_nc_dataset
from data_utils import eval_acc_nc, class_rand_splits, make_random_split_idx, eval_f1_nc
from eval import evaluate
from parse import (
    apply_full_batch_aug_defaults,
    apply_full_batch_auto_defaults,
    get_explicit_arg_names,
    parse_method,
    parser_add_main_args,
)
from augmentation import BernoulliEdgeAugmentor
from viz_pq import save_pq_plots
from pq_gradnorm import PQGradNormTuner, PQTunerConfig


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


def _first_nonfinite_gradient(model: nn.Module) -> Optional[Tuple[str, torch.Size, int]]:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        finite = torch.isfinite(param.grad)
        if not bool(finite.all()):
            return name, param.grad.shape, int((~finite).sum().item())
    return None


# -----------------------
# Parse args
# -----------------------
parser = argparse.ArgumentParser(description="Training Pipeline for Node Classification")
parser_add_main_args(parser)
args = parser.parse_args()
explicit_arg_names = get_explicit_arg_names(sys.argv[1:])

# Enforce strict linearity at CLI level (before wandb overrides)
if getattr(args, "linear", False):
    args.dropout = 0.0
    if hasattr(args, "bn"):
        args.bn = False
    if hasattr(args, "ln"):
        args.ln = False


# -----------------------
# wandb init (optional)
# -----------------------
wandb_run = None
if getattr(args, "wandb", False):
    if wandb is None:
        raise ImportError("wandb is not installed. Run: pip install wandb")

    tags = [t.strip() for t in str(getattr(args, "wandb_tags", "")).split(",") if t.strip()]

    wandb_run = wandb.init(
        project=str(getattr(args, "wandb_project", "rade-nc")),
        entity=getattr(args, "wandb_entity", None) or None,
        group=getattr(args, "wandb_group", None) or None,
        name=getattr(args, "wandb_name", None) or None,
        tags=tags,
        mode=str(getattr(args, "wandb_mode", "online")),
    )

    for k, v in dict(wandb.config).items():
        if hasattr(args, k):
            setattr(args, k, v)
    explicit_arg_names.update({k for k in dict(wandb.config).keys() if hasattr(args, k)})

    if getattr(args, "linear", False):
        args.dropout = 0.0
        if hasattr(args, "bn"):
            args.bn = False
        if hasattr(args, "ln"):
            args.ln = False


preset_key, auto_applied, auto_skipped = apply_full_batch_auto_defaults(args, explicit_arg_names)
if preset_key is not None:
    auto_parts = [f"{name}={value}" for name, value in auto_applied.items()]
    skipped_parts = [f"{name}={value}" for name, value in auto_skipped.items()]

    if auto_parts:
        print(f"[AUTO-CONFIG] full-batch preset {preset_key[0]}/{preset_key[1]} applied: {', '.join(auto_parts)}")
    if skipped_parts:
        print(
            f"[AUTO-CONFIG] kept explicit overrides for {preset_key[0]}/{preset_key[1]}: "
            f"{', '.join(skipped_parts)}"
        )

aug_preset_key, aug_auto_applied, aug_auto_skipped = apply_full_batch_aug_defaults(args, explicit_arg_names)
if aug_preset_key is not None:
    auto_parts = [f"{name}={value}" for name, value in aug_auto_applied.items()]
    skipped_parts = [f"{name}={value}" for name, value in aug_auto_skipped.items()]

    if auto_parts:
        print(
            f"[AUTO-CONFIG] full-batch augmentation preset {aug_preset_key[0]}/{aug_preset_key[1]} applied: "
            f"{', '.join(auto_parts)}"
        )
    if skipped_parts:
        print(
            f"[AUTO-CONFIG] kept explicit augmentation overrides for {aug_preset_key[0]}/{aug_preset_key[1]}: "
            f"{', '.join(skipped_parts)}"
        )


# -----------------------
# Linear-backbone routing
# -----------------------
gnn_raw = str(getattr(args, "gnn", "gcn")).lower().strip()
args.gnn_raw = gnn_raw  # keep for tagging/logging

if gnn_raw == "sgc":
    # SGC = linearized version of GCN stack (no nonlinearity, no augmentation)
    args.gnn = "gcn"
    args.linear = True
elif gnn_raw == "gin-linear":
    # GIN-Linear = linearized version of GIN stack (no nonlinearity, no augmentation)
    args.gnn = "gin"
    args.linear = True
elif gnn_raw == "gat" and bool(getattr(args, "linear", False)):
    raise ValueError(
        "--linear is unavailable for --gnn gat because GAT attention is feature-dependent "
        "and nonlinear. Use --linear False, or use --gnn sgc / --gnn gin-linear for "
        "strict linear baselines."
    )

# If routed to a strict linear backbone, disable stochastic augmentation / tuning
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

print(args)
if wandb_run is not None:
    wandb.config.update(vars(args), allow_val_change=True)


# -----------------------
# Augmentation technique routing
# -----------------------
aug_tech = str(getattr(args, "aug_tech", "rade")).lower().strip()
if aug_tech not in {"rade", "dropout", "dropedge", "dropmessage", "dropnode", "none"}:
    raise ValueError(
        f"--aug_tech must be one of {{rade, dropout, dropedge, dropmessage, dropnode, none}}. Got {aug_tech}"
    )


def _require_no_rade_contrib(name: str) -> None:
    if getattr(args, "ep_correction", False):
        raise ValueError(f"--aug_tech={name} must be used with --ep_correction False.")
    if getattr(args, "pq_gradnorm", False):
        raise ValueError(f"--aug_tech={name} must be used with --pq_gradnorm False.")
    if str(getattr(args, "aug_mode", "none")).lower().strip() != "none":
        raise ValueError(f"--aug_tech={name} requires --aug_mode none (edge augmentation disabled).")
    if getattr(args, "linear", False):
        raise ValueError(f"--aug_tech={name} is incompatible with --linear (strict linearity).")


# DropMessage must NOT use RADE-specific contributions
if aug_tech == "dropmessage":
    _require_no_rade_contrib("dropmessage")

# DropNode must NOT use RADE-specific contributions
if aug_tech == "dropnode":
    _require_no_rade_contrib("dropnode")

if aug_tech == "dropout":
    _require_no_rade_contrib("dropout")

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
    args.ep_correction = False
    args.pq_gradnorm = False

if (
    str(getattr(args, "gnn", "")).lower().strip() == "gat"
    and bool(getattr(args, "pq_gradnorm", False))
    and int(getattr(args, "gat_heads", 1)) != 1
):
    raise ValueError("GAT PQ-GradNorm variance currently supports only --gat_heads 1.")

if bool(getattr(args, "pq_cap_p", False)) and not (0.0 <= float(getattr(args, "pq_p_max", 0.9)) < 1.0):
    raise ValueError(f"--pq_p_max must be in [0, 1) when --pq_cap_p is enabled. Got {args.pq_p_max}")
if bool(getattr(args, "pq_cap_rho", False)) and float(getattr(args, "pq_rho_max", 0.2)) < 0.0:
    raise ValueError(f"--pq_rho_max must be >= 0 when --pq_cap_rho is enabled. Got {args.pq_rho_max}")
if float(getattr(args, "gat_ep_corr_clip", 2.0)) < 0.0:
    raise ValueError(f"--gat_ep_corr_clip must be >= 0. Got {args.gat_ep_corr_clip}")

print(
    f"[CONFIG] dataset={args.dataset} | backbone_raw={getattr(args, 'gnn_raw', args.gnn)} -> backbone={args.gnn} | "
    f"linear={bool(getattr(args, 'linear', False))} | "
    f"aug_tech={args.aug_tech} aug_mode={args.aug_mode} | "
    f"ep_correction={bool(getattr(args, 'ep_correction', False))} "
    f"(mode=analytic, variant={getattr(args, 'rade_variant', '-')}) | "
    f"pq_gradnorm={bool(getattr(args, 'pq_gradnorm', False))} "
    f"(search=adam, "
    f"p_cap={'ON' if bool(getattr(args, 'pq_cap_p', False)) else 'OFF'}"
    f"{'@' + str(float(getattr(args, 'pq_p_max', 0.9))) if bool(getattr(args, 'pq_cap_p', False)) else ''}, "
    f"rho_cap={'ON' if bool(getattr(args, 'pq_cap_rho', False)) else 'OFF'}"
    f"{'@' + str(float(getattr(args, 'pq_rho_max', 0.2))) if bool(getattr(args, 'pq_cap_rho', False)) else ''}) | "
    f"p={float(getattr(args, 'p', 0.0)):.4f} q={float(getattr(args, 'q', 0.0)):.6f} | "
    f"mask_sharing={getattr(args, 'mask_sharing', '-')} | "
    f"gat_moment=sampled "
    f"(corr_clip={float(getattr(args, 'gat_ep_corr_clip', 2.0)):g})"
)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

fix_seed(args.seed, device=device)


# -----------------------
# Load data (contract)
# -----------------------
data, base_split_idx = load_nc_dataset(
    root=args.data_dir,
    name=args.dataset,
    normalize_features=True,
    make_undirected_edges=True,
    remove_citeseer_isolated_nodes=bool(getattr(args, "remove_citeseer_isolated_nodes", False)),
    seed=args.seed,
    train_prop=args.train_prop,
    valid_prop=args.valid_prop,
)

# Basic dataset stats
n = int(data.num_nodes)
e = int(data.edge_index.size(1)) // 2
d = int(data.x.size(1))
c = int(data.y.max().item()) + 1
total_undir_pairs = max(float(n) * float(max(n - 1, 0)) / 2.0, 1.0)
graph_density = float(e) / total_undir_pairs
rho_scale_once = 0.0 if graph_density <= 0.0 else 1.0 / graph_density

print(f"dataset {args.dataset} | num nodes {n} | num edges {e} | num feats {d} | num classes {c}")
print(f"[DATASET] graph_density={graph_density:.6e} | rho_scale=1/d={rho_scale_once:.6e}")
if wandb_run is not None:
    wandb.config.update(
        {
            "num_nodes": n,
            "num_edges": e,
            "num_feats": d,
            "num_classes": c,
            "graph_density": graph_density,
            "pq_expected_add_ratio_scale": rho_scale_once,
        },
        allow_val_change=True,
    )


# Cache clean edges on CPU for fast per-epoch augmentation
augmentor = None
if aug_tech in {"rade", "dropedge"}:
    augmentor = BernoulliEdgeAugmentor.from_edge_index(data.edge_index, num_nodes=int(data.num_nodes))


# -----------------------
# Build split list for runs
# -----------------------
split_idx_lst = []
p_histories = []
q_histories = []
obj_histories = []
rho_histories = []
for run in range(args.runs):
    if args.rand_split:
        split_idx = make_random_split_idx(n, args.train_prop, args.valid_prop, seed=args.seed + run)
    elif args.rand_split_class:
        split_idx = class_rand_splits(
            data.y,
            args.label_num_per_class,
            valid_num=args.valid_num,
            test_num=args.test_num,
            seed=args.seed + run,
        )
    else:
        split_idx = base_split_idx

    split_idx = {k: v.to(torch.long) for k, v in split_idx.items()}
    split_idx_lst.append(split_idx)

# Move data once
data = data.to(device)


def _enforce_aug_mode(p_val: float, q_val: float, aug_mode: str) -> Tuple[float, float]:
    aug_mode = str(aug_mode).lower().strip()
    if aug_mode == "drop":
        return float(p_val), 0.0
    if aug_mode == "add":
        return 0.0, float(q_val)
    if aug_mode == "none":
        return 0.0, 0.0
    return float(p_val), float(q_val)


def _apply_p_cap(p_val: float) -> float:
    if not bool(getattr(args, "pq_cap_p", False)):
        return float(p_val)
    return float(min(float(p_val), float(getattr(args, "pq_p_max", 0.9))))


def _sync_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _format_tag_float(value: float) -> str:
    return str(f"{float(value):g}").replace(".", "p").replace("-", "m")


def _safe_std(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    return float(values.std().item())


def _format_runtime_list(values) -> str:
    return ";".join(f"{float(value):.6f}" for value in values)


def _write_single_dataset_csv(record: Dict[str, object], out_path: str = "single_dataset/results.csv") -> str:
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
        "gat_moment_samples",
        "gat_nonedge_samples",
        "gat_moment_seed",
        "gat_ep_corr_clip",
        "lr",
        "weight_decay",
        "dropout",
        "patience",
        "runs",
        "epochs",
        "seed",
        "device",
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
        "pq_warmup_epochs",
        "pq_eps",
        "pq_seed",
        "train_prop",
        "valid_prop",
        "rand_split",
        "rand_split_class",
        "label_num_per_class",
        "valid_num",
        "test_num",
        "remove_citeseer_isolated_nodes",
    ]

    write_header = not os.path.exists(abs_path) or os.path.getsize(abs_path) == 0
    with open(abs_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in fieldnames})

    return abs_path


def _build_gradnorm_plot_tag(args) -> str:
    dataset_tag = str(args.dataset).lower().strip()
    gnn_tag = str(getattr(args, "gnn_raw", getattr(args, "gnn", "gcn"))).lower().strip()
    variant_tag = str(getattr(args, "rade_variant", "rade-of")).lower().strip().replace("-", "")
    search_tag = f"adam_lr{_format_tag_float(float(getattr(args, 'pq_adam_lr', 0.0)))}"
    if bool(getattr(args, "pq_optimize_rho", False)):
        search_tag = f"{search_tag}_rhoopt"
    if bool(getattr(args, "pq_cap_p", False)):
        search_tag = f"{search_tag}_pcap{_format_tag_float(float(getattr(args, 'pq_p_max', 0.9)))}"
    if bool(getattr(args, "pq_cap_rho", False)):
        search_tag = f"{search_tag}_rhocap{_format_tag_float(float(getattr(args, 'pq_rho_max', 0.2)))}"

    lam_p_tag = f"plam{_format_tag_float(float(getattr(args, 'pq_deletion_penalty_lambda', 0.0)))}"
    lam_tag = f"lam{_format_tag_float(float(getattr(args, 'pq_densification_penalty_lambda', 0.0)))}"
    penalty_tag = "quadratic"
    return f"{dataset_tag}_{search_tag}_{lam_p_tag}_{lam_tag}_{penalty_tag}_{gnn_tag}_{variant_tag}"


# -----------------------
# Model / loss / metric
# -----------------------
model = parse_method(args, n, c, d, device)

# IMPORTANT for RADE-GCN / RADE-GIN:
if aug_tech == "rade":
    if not hasattr(model, "set_graph"):
        raise RuntimeError("aug_tech=rade but model has no set_graph(). Add it to your model class.")
    model.set_graph(data.edge_index, int(data.num_nodes))

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
eval_func = eval_f1_nc if args.dataset.lower() == "flickr" else eval_acc_nc
logger = Logger(args.runs, args)

print("MODEL:", model)
print("[Timing] enabled for all full-batch runs")


# -----------------------
# Training loop
# -----------------------
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx["train"].to(device)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # (p, q) used for this run
    p_eff, q_eff = _enforce_aug_mode(float(args.p), float(args.q), args.aug_mode)
    p_eff = _apply_p_cap(p_eff)

    # Record p, q used at each epoch
    p_trace = []
    q_trace = []
    obj_trace = []
    rho_trace = []

    # Fresh tuner per run
    pq_tuner = None
    if getattr(args, "pq_gradnorm", False):
        if str(args.aug_mode).lower().strip() == "none":
            raise ValueError("--pq_gradnorm requires --aug_mode != none.")
        if not getattr(args, "ep_correction", False):
            raise ValueError("--pq_gradnorm currently requires --ep_correction True.")

        cfg = PQTunerConfig(
            eps=args.pq_eps,
            subset_nodes=args.pq_subset_nodes,
            seed=args.seed,
            deletion_penalty_lambda=args.pq_deletion_penalty_lambda,
            densification_penalty_lambda=args.pq_densification_penalty_lambda,
            optimize_rho=bool(getattr(args, "pq_optimize_rho", False)),
            cap_p=bool(getattr(args, "pq_cap_p", False)),
            p_max=float(getattr(args, "pq_p_max", 0.9)),
            cap_rho=bool(getattr(args, "pq_cap_rho", False)),
            rho_max=float(getattr(args, "pq_rho_max", 0.2)),
            adam_lr=args.pq_adam_lr,
            adam_beta1=args.pq_adam_beta1,
            adam_beta2=args.pq_adam_beta2,
            adam_eps=args.pq_adam_eps,
        )
        pq_tuner = PQGradNormTuner(
            edge_index_clean=data.edge_index,
            num_nodes=int(data.num_nodes),
            gnn=str(args.gnn),
            rade_variant=str(args.rade_variant),
            cfg=cfg,
        )
    rho_scale = 0.0 if pq_tuner is None else float(pq_tuner.expected_add_ratio_scale)
    if pq_tuner is not None:
        q_eff = float(max(0.0, min(q_eff, float(pq_tuner.q_probability_upper))))

    best_val = float("-inf")
    best_test = float("-inf")
    patience = int(getattr(args, "patience", 100))
    epochs_no_improve = 0
    best_epoch = -1

    if args.save_model:
        save_model(args, model, optimizer, run)

    for epoch in range(args.epochs):
        _sync_for_timing(device)
        epoch_time = time.perf_counter()
        epoch_display = int(epoch + 1)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        p_epoch = float(p_eff)
        q_epoch = float(q_eff)

        p_trace.append(p_epoch)
        q_trace.append(q_epoch)
        obj_trace.append(float("nan"))
        rho_trace.append(float(q_epoch) * float(rho_scale))

        # -----------------------
        # Augment per epoch
        # -----------------------
        aug_stats = None
        edge_index_train = data.edge_index
        edge_index_keep = None
        edge_index_add = None

        dropmessage_rate = float(getattr(args, "dropmessage_rate", 0.0)) if aug_tech == "dropmessage" else 0.0
        dropnode_rate = float(getattr(args, "dropnode_rate", 0.0)) if aug_tech == "dropnode" else 0.0

        do_aug = (
            aug_tech in {"rade", "dropedge"}
            and (str(args.aug_mode).lower().strip() != "none")
            and (p_epoch > 0.0 or q_epoch > 0.0)
        )

        if do_aug:
            base_seed = args.seed + 10_000 * run + epoch

            if aug_tech == "dropedge":
                edge_index_train, aug_stats = augmentor.augment_edge_index(
                    p=p_epoch,
                    q=0.0,
                    mode="drop",
                    seed=base_seed,
                    device=device,
                    return_stats=True,
                )
            elif str(getattr(args, "mask_sharing", "shared")).lower().strip() == "shared":
                _, edge_index_keep, edge_index_add, aug_stats = augmentor.augment_edge_indices(
                    p=p_epoch,
                    q=q_epoch,
                    mode=args.aug_mode,
                    seed=base_seed,
                    device=device,
                )
            else:
                edge_index_keep = []
                edge_index_add = []
                stats_list = []

                for layer in range(int(args.local_layers)):
                    _, ei_keep_l, ei_add_l, st = augmentor.augment_edge_indices(
                        p=p_epoch,
                        q=q_epoch,
                        mode=args.aug_mode,
                        seed=base_seed + 1_000_000 * layer,
                        device=device,
                    )
                    edge_index_keep.append(ei_keep_l)
                    edge_index_add.append(ei_add_l)
                    stats_list.append(st)

                aug_stats = {
                    "dropped_undir_avg": float(sum(s["dropped_undir"] for s in stats_list)) / float(len(stats_list)),
                    "added_undir_avg": float(sum(s["added_undir"] for s in stats_list)) / float(len(stats_list)),
                }

        # -----------------------
        # Forward
        # -----------------------
        if aug_tech == "dropmessage":
            logits = model(data.x, data.edge_index, dropmessage_rate=dropmessage_rate)
        elif aug_tech == "dropnode":
            logits = model(data.x, data.edge_index, dropnode_rate=dropnode_rate)
        elif aug_tech == "dropedge" and do_aug:
            logits = model(data.x, edge_index_train)
        elif aug_tech == "rade" and do_aug:
            logits = model(
                data.x,
                data.edge_index,
                edge_index_keep=edge_index_keep,
                edge_index_add=edge_index_add,
                p=p_epoch,
                q=q_epoch,
            )
        else:
            logits = model(data.x, data.edge_index)

        loss = criterion(logits[train_idx], data.y[train_idx])
        skipped_optimizer_step = False
        if bool(getattr(args, "skip_nonfinite_grad_step", True)):
            if not bool(torch.isfinite(loss).all()):
                print(
                    f"[NonfiniteGuard] run={run + 1:02d} epoch={epoch + 1:03d} "
                    f"loss={float(loss.detach().cpu())}; skipped optimizer step."
                )
                skipped_optimizer_step = True
                optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                bad_grad = _first_nonfinite_gradient(model)
                if bad_grad is not None:
                    name, shape, count = bad_grad
                    print(
                        f"[NonfiniteGuard] run={run + 1:02d} epoch={epoch + 1:03d} "
                        f"nonfinite gradient in {name} shape={tuple(shape)} count={count}; "
                        "skipped optimizer step."
                    )
                    skipped_optimizer_step = True
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()
        else:
            loss.backward()
            optimizer.step()

        # -----------------------
        # Evaluation
        # -----------------------
        model.eval()
        with torch.no_grad():
            if aug_tech == "dropmessage":
                logits_eval = model(data.x, data.edge_index, dropmessage_rate=0.0)
            elif aug_tech == "dropnode":
                logits_eval = model(data.x, data.edge_index, dropnode_rate=0.0)
            elif aug_tech == "dropedge":
                logits_eval = model(data.x, data.edge_index)
            elif getattr(args, "ep_correction", False) and str(getattr(args, "rade_variant", "rade-of")) == "rade-ofs":
                logits_eval = model(data.x, data.edge_index, p=p_epoch, q=q_epoch)
            else:
                logits_eval = model(data.x, data.edge_index)

        result = evaluate(
            model,
            data,
            split_idx,
            eval_func,
            criterion,
            args=args,
            result=logits_eval,
            device=device,
        )

        logger.add_result(run, result[:4])

        improved = (result[1] > best_val)
        if improved:
            best_val = float(result[1])
            best_test = float(result[2])
            best_epoch = int(epoch)
            epochs_no_improve = 0
            if args.save_model:
                save_model(args, model, optimizer, run)
        else:
            epochs_no_improve += 1

        stop_now = patience > 0 and epochs_no_improve >= patience

        # -----------------------
        # PQ-GradNorm update (RADE only)
        # -----------------------
        if (not stop_now) and pq_tuner is not None:
            warmup = int(getattr(args, "pq_warmup_epochs", 0))
            every = int(getattr(args, "pq_update_every", 1))
            if (epoch + 1) >= warmup and ((epoch + 1) % every == 0):
                epoch_seed = int(args.seed) + 100000 * int(run) + int(epoch) + int(getattr(args, "pq_seed", 0))

                prev_mode_training = model.training
                model.train()

                p_new, q_new, info = pq_tuner.suggest_pq(
                    model=model,
                    data=data,
                    train_idx=train_idx,
                    criterion=criterion,
                    current_p=p_epoch,
                    current_q=q_epoch,
                    aug_mode=str(args.aug_mode).lower().strip(),
                    epoch_seed=epoch_seed,
                    mask_sharing=str(getattr(args, "mask_sharing", "shared")).lower().strip(),
                    num_layers=int(getattr(args, "local_layers", 1)),
                )

                if not prev_mode_training:
                    model.eval()

                p_new, q_new = _enforce_aug_mode(p_new, q_new, args.aug_mode)
                p_new = _apply_p_cap(p_new)
                p_eff, q_eff = p_new, q_new
                p_trace[-1] = float(p_eff)
                q_trace[-1] = float(q_eff)
                rho_trace[-1] = float(q_eff) * float(rho_scale)
                obj_trace[-1] = float(info.get("obj_new", info["obj"]))

                g_reg_floor = " floor" if bool(info.get("G_reg_best_at_eps_floor", False)) else ""
                p_cap = "ON" if bool(info.get("p_cap_enabled", False)) else "OFF"
                print(
                    f"[PQ-GradNorm-Adam] run={run + 1:02d} epoch={epoch + 1:03d} "
                    f"G_data={info['G_data']:.3e} G_reg={info['G_reg_best']:.3e}{g_reg_floor} "
                    f"p={info['p_best']:.6f}->{info['p_new']:.6f} "
                    f"q={info['q_best']:.6f}->{info['q_new']:.6f} "
                    f"rho={info.get('rho_best', float('nan')):.3e}->{info.get('rho_new', float('nan')):.3e} "
                    f"obj={info.get('obj_new', info['obj']):.2e} "
                    f"lr={info.get('adam_lr', float('nan')):.2e} "
                    f"p_cap={p_cap}@{info.get('p_probability_upper', float('nan')):.4f} "
                    f"eps={info.get('pq_eps', float(getattr(args, 'pq_eps', float('nan')))):.1e} "
                    f"opt={info.get('optimization_variable', 'q')}"
                )

                if wandb_run is not None:
                    pq_log = {
                        "pq/epoch": epoch_display,
                        "pq/G_data": float(info["G_data"]),
                        "pq/G_reg_best": float(info["G_reg_best"]),
                        "pq/obj": float(info["obj"]),
                        "pq/obj_selected": float(info.get("obj_new", info["obj"])),
                        "pq/p_used": p_epoch,
                        "pq/q_used": q_epoch,
                        "pq/rho_used": float(q_epoch) * float(rho_scale),
                        "pq/p_eval": float(info["p_best"]),
                        "pq/q_eval": float(info["q_best"]),
                        "pq/p_next": float(info["p_new"]),
                        "pq/q_next": float(info["q_new"]),
                        "pq/rho_next": float(info.get("rho_new", float(q_eff) * float(rho_scale))),
                        "pq/p_best": float(info["p_best"]),
                        "pq/q_best": float(info["q_best"]),
                        "pq/p_new": float(info["p_new"]),
                        "pq/q_new": float(info["q_new"]),
                        "pq/rho_best": float(info.get("rho_best", float("nan"))),
                        "pq/rho_new": float(info.get("rho_new", float("nan"))),
                        "pq/optimize_rho": float(bool(info.get("optimize_rho", False))),
                        "pq/p_cap_enabled": float(bool(info.get("p_cap_enabled", False))),
                        "pq/p_probability_upper": float(info.get("p_probability_upper", float("nan"))),
                        "pq/rho_cap_enabled": float(bool(info.get("rho_cap_enabled", False))),
                        "pq/rho_probability_upper": float(info.get("rho_probability_upper", float("nan"))),
                        "pq/q_probability_upper": float(info.get("q_probability_upper", float("nan"))),
                    }
                    if pq_method == "adam":
                        pq_log["pq/G_reg_curr"] = float(info["G_reg_best"])
                        pq_log["pq/G_reg_next"] = float(info.get("G_reg_new", float("nan")))
                        pq_log["pq/obj_curr"] = float(info["obj"])
                        pq_log["pq/obj_next"] = float(info.get("obj_new", float("nan")))
                        pq_log["pq/eps"] = float(info.get("pq_eps", getattr(args, "pq_eps", float("nan"))))
                        pq_log["pq/G_reg_best_at_eps_floor"] = float(bool(info.get("G_reg_best_at_eps_floor", False)))
                        pq_log["pq/G_reg_new_at_eps_floor"] = float(bool(info.get("G_reg_new_at_eps_floor", False)))
                    if "deletion_penalty_best" in info:
                        pq_log["pq/deletion_penalty_best"] = float(info["deletion_penalty_best"])
                    if "deletion_penalty_new" in info:
                        pq_log["pq/deletion_penalty_new"] = float(info["deletion_penalty_new"])
                    if "densification_penalty_best" in info:
                        pq_log["pq/densification_penalty_best"] = float(info["densification_penalty_best"])
                    if "rho_penalty_best" in info:
                        pq_log["pq/rho_penalty_best"] = float(info["rho_penalty_best"])
                    if "densification_penalty_new" in info:
                        pq_log["pq/densification_penalty_new"] = float(info["densification_penalty_new"])
                    if "rho_penalty_new" in info:
                        pq_log["pq/rho_penalty_new"] = float(info["rho_penalty_new"])
                    if "grad_rho" in info:
                        pq_log["pq/grad_rho"] = float(info["grad_rho"])
                        pq_log["pq/grad_rho_abs"] = float(
                            info.get("grad_rho_abs", abs(float(info["grad_rho"])))
                        )
                    if "delta_rho" in info:
                        pq_log["pq/delta_rho"] = float(info["delta_rho"])
                    if "grad_p" in info:
                        pq_log["pq/grad_p"] = float(info["grad_p"])
                        pq_log["pq/grad_p_abs"] = float(info.get("grad_p_abs", abs(float(info["grad_p"]))))
                    if "delta_p" in info:
                        pq_log["pq/delta_p"] = float(info["delta_p"])
                    wandb.log(
                        pq_log,
                        step=(epoch + run * int(args.epochs)),
                    )

        _sync_for_timing(device)
        epoch_time = time.perf_counter() - epoch_time
        logger.add_runtime(run, epoch_time)

        # -----------------------
        # wandb logging
        # -----------------------
        if wandb_run is not None and (epoch % int(getattr(args, "wandb_log_every", 1)) == 0):
            step = epoch + run * int(args.epochs)

            log_dict = {
                "aug/tech": str(aug_tech),
                "run": int(run + 1),
                "epoch": epoch_display,
                "train/loss": float(loss.item()),
                "train/acc": float(result[0]),
                "valid/acc": float(result[1]),
                "test/acc": float(result[2]),
                "valid/loss": float(result[3].item()) if hasattr(result[3], "item") else float(result[3]),
                "best/valid_acc": float(best_val),
                "best/test_acc_at_best_valid": float(best_test),
                "runtime/epoch_sec": float(epoch_time),
                "train/skipped_optimizer_step": float(skipped_optimizer_step),
            }

            if aug_tech == "rade":
                log_dict["aug/p"] = p_epoch
                log_dict["aug/q"] = q_epoch
                log_dict["aug/p_used"] = p_epoch
                log_dict["aug/q_used"] = q_epoch
                log_dict["aug/rho_used"] = float(q_epoch) * float(rho_scale)
                log_dict["aug/p_next"] = float(p_eff)
                log_dict["aug/q_next"] = float(q_eff)
                log_dict["aug/rho_next"] = float(q_eff) * float(rho_scale)
                log_dict["rade/ep_correction"] = bool(getattr(args, "ep_correction", False))
                log_dict["rade/variant"] = str(getattr(args, "rade_variant", "rade-of"))

                if aug_stats is not None:
                    if "dropped_undir" in aug_stats:
                        log_dict["aug/dropped_undir"] = float(aug_stats["dropped_undir"])
                    if "added_undir" in aug_stats:
                        log_dict["aug/added_undir"] = float(aug_stats["added_undir"])
                    if "dropped_undir_avg" in aug_stats:
                        log_dict["aug/dropped_undir_avg"] = float(aug_stats["dropped_undir_avg"])
                    if "added_undir_avg" in aug_stats:
                        log_dict["aug/added_undir_avg"] = float(aug_stats["added_undir_avg"])

            elif aug_tech == "dropmessage":
                log_dict["aug/dropmessage_rate"] = float(dropmessage_rate)

            elif aug_tech == "dropnode":
                log_dict["aug/dropnode_rate"] = float(dropnode_rate)

            elif aug_tech == "dropedge":
                log_dict["aug/dropedge_rate"] = float(p_epoch)
                if aug_stats is not None:
                    if "dropped_undir" in aug_stats:
                        log_dict["aug/dropped_undir"] = float(aug_stats["dropped_undir"])
                    if "kept_undir" in aug_stats:
                        log_dict["aug/kept_undir"] = float(aug_stats["kept_undir"])

            elif aug_tech == "dropout":
                log_dict["regularization/dropout"] = float(getattr(args, "dropout", 0.0))

            wandb.log(log_dict, step=step)

        # -----------------------
        # Console print
        # -----------------------
        if epoch % int(args.display_step) == 0:
            aug_str = ""

            if aug_tech == "rade":
                if aug_stats is not None:
                    if "dropped_undir" in aug_stats and "added_undir" in aug_stats:
                        dU = aug_stats["dropped_undir"]
                        aU = aug_stats["added_undir"]
                        aug_str = f", Drop/Add: -{dU} +{aU} edges"
                    elif "dropped_undir_avg" in aug_stats and "added_undir_avg" in aug_stats:
                        dU = aug_stats["dropped_undir_avg"]
                        aU = aug_stats["added_undir_avg"]
                        aug_str = f", Drop/Add(avg per layer): -{dU:.1f} +{aU:.1f} edges"

                aug_str += (
                    f" (p={p_epoch:.4f}, q={q_epoch:.6f}, "
                    f"rho={float(q_epoch) * float(rho_scale):.3e}, "
                    f"ep_correction={bool(getattr(args, 'ep_correction', False))}, "
                    f"ep_mode=analytic, "
                    f"variant={getattr(args, 'rade_variant', 'rade-of')})"
                )

            elif aug_tech == "dropmessage":
                aug_str = f" (dropmessage={dropmessage_rate:.3f})"

            elif aug_tech == "dropnode":
                aug_str = f" (dropnode={dropnode_rate:.3f})"

            elif aug_tech == "dropedge":
                aug_str = f" (dropedge={p_epoch:.3f})"
                if aug_stats is not None and "dropped_undir" in aug_stats:
                    aug_str += f", DropEdge: -{aug_stats['dropped_undir']} edges"

            elif aug_tech == "dropout" and float(getattr(args, "dropout", 0.0)) > 0.0:
                aug_str = f" (dropout={float(getattr(args, 'dropout', 0.0)):.3f})"

            if skipped_optimizer_step:
                aug_str += ", skipped_optimizer_step=True"
            aug_str += f", Epoch Time: {epoch_time:.4f}s"

            print(
                f"Run: {run + 1:02d}, "
                f"Epoch: {epoch_display:03d}, "
                f"Loss: {loss:.4f}, "
                f"Train: {100 * result[0]:.2f}%, "
                f"Valid: {100 * result[1]:.2f}%, "
                f"Test: {100 * result[2]:.2f}%, "
                f"Best Valid: {100 * best_val:.2f}%, "
                f"Best Test: {100 * best_test:.2f}%"
                f"{aug_str}"
            )

        # -----------------------
        # Early stopping trigger
        # -----------------------
        if stop_now:
            stop_epoch = epoch + 1
            best_epoch_disp = best_epoch + 1 if best_epoch >= 0 else -1
            msg = (
                f"[EarlyStop] run={run + 1:02d} stopping at epoch {stop_epoch:03d} "
                f"(no val improvement for {patience} epochs). "
                f"Best val={best_val:.4f} at epoch {best_epoch_disp:03d}, "
                f"test@best={best_test:.4f}."
            )
            print(msg)

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

# Save GradNorm p/q schedules (mean±std over runs) when pq_gradnorm is enabled
if getattr(args, "pq_gradnorm", False) and str(getattr(args, "aug_tech", "rade")).lower().strip() == "rade":
    tag = _build_gradnorm_plot_tag(args)

    save_pq_plots(
        p_histories,
        q_histories,
        obj_histories,
        rho_histories,
        tag=tag,
        out_dir="visualization",
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
    "gat_moment_samples": int(getattr(args, "gat_moment_samples", 256)),
    "gat_nonedge_samples": int(getattr(args, "gat_nonedge_samples", 256)),
    "gat_moment_seed": int(getattr(args, "gat_moment_seed", 0)),
    "gat_ep_corr_clip": float(getattr(args, "gat_ep_corr_clip", 2.0)),
    "lr": float(args.lr),
    "weight_decay": float(args.weight_decay),
    "dropout": float(args.dropout),
    "patience": int(args.patience),
    "runs": int(args.runs),
    "epochs": int(args.epochs),
    "seed": int(args.seed),
    "device": str(device),
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
    "pq_warmup_epochs": int(args.pq_warmup_epochs),
    "pq_eps": float(args.pq_eps),
    "pq_seed": int(args.pq_seed),
    "train_prop": float(args.train_prop),
    "valid_prop": float(args.valid_prop),
    "rand_split": bool(args.rand_split),
    "rand_split_class": bool(args.rand_split_class),
    "label_num_per_class": int(args.label_num_per_class),
    "valid_num": int(args.valid_num),
    "test_num": int(args.test_num),
    "remove_citeseer_isolated_nodes": bool(args.remove_citeseer_isolated_nodes),
}
single_dataset_csv = _write_single_dataset_csv(single_dataset_record)
print(f"[SINGLE-DATASET] Saved summary CSV row to {single_dataset_csv}")

if wandb_run is not None:
    try:
        wandb_run.summary["final_test_mean"] = float(results.mean().item())
        wandb_run.summary["final_test_std"] = float(results.std().item())
        if runtime_summary is not None:
            wandb_run.summary["runtime/profile"] = "all_full_batch_runs"
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
