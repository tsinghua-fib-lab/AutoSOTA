from __future__ import annotations

import argparse

from .mb_models import MPNNsMB


def str2bool(value):
    if isinstance(value, bool):
        return value

    value_norm = str(value).strip().lower()
    if value_norm in {"1", "true", "t", "yes", "y"}:
        return True
    if value_norm in {"0", "false", "f", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


MINI_BATCH_AUTO_DEFAULTS = {
    ("flickr", "gcn"): {
        "hidden_channels": 256,
        "epochs": 100,
        "p": 0.5,
        "q": 0.0000564785,
    },
    ("flickr", "gin"): {
        "hidden_channels": 256,
        "epochs": 1000,
        "p": 0.5,
        "q": 0.0000564785,
    },
    ("flickr", "gat"): {
        "hidden_channels": 256,
        "epochs": 1000,
        "p": 0.5,
        "q": 0.0000564785,
    },
    ("ogbn-arxiv", "gcn"): {
        "hidden_channels": 256,
        "epochs": 100,
        "p": 0.5,
        "q": 0.00004037395,
    },
    ("ogbn-arxiv", "gin"): {
        "hidden_channels": 256,
        "epochs": 2000,
        "p": 0.5,
        "q": 0.00004037395,
    },
    ("ogbn-arxiv", "gat"): {
        "hidden_channels": 256,
        "epochs": 2000,
        "p": 0.5,
        "q": 0.00004037395,
    },
}


MINI_BATCH_AUG_DEFAULTS = {
    "flickr": {
        "dropout": 0.50,
        "dropedge_rate": 0.20,
        "dropnode_rate": 0.20,
        "dropmessage_rate": 0.50,
    },
    "ogbn-arxiv": {
        "dropout": 0.20,
        "dropedge_rate": 0.20,
        "dropnode_rate": 0.50,
        "dropmessage_rate": 0.50,
    },
}


def get_explicit_arg_names(argv):
    explicit = set()

    for token in argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue

        option = token[2:].split("=", 1)[0].strip()
        if option:
            explicit.add(option.replace("-", "_"))

    return explicit


def apply_mini_batch_auto_defaults(args, explicit_arg_names=None):
    explicit_arg_names = explicit_arg_names or set()

    dataset = str(getattr(args, "dataset", "")).lower().strip()
    gnn = str(getattr(args, "gnn", "")).lower().strip()
    preset_key = (dataset, gnn)
    preset = MINI_BATCH_AUTO_DEFAULTS.get(preset_key)

    if preset is None:
        return None, {}, {}

    applied = {}
    skipped = {}
    for field_name, field_value in preset.items():
        if field_name in explicit_arg_names:
            skipped[field_name] = getattr(args, field_name)
            continue
        setattr(args, field_name, field_value)
        applied[field_name] = field_value

    return preset_key, applied, skipped


def apply_mini_batch_aug_defaults(args, explicit_arg_names=None):
    explicit_arg_names = explicit_arg_names or set()

    dataset = str(getattr(args, "dataset", "")).lower().strip()
    aug_tech = str(getattr(args, "aug_tech", "")).lower().strip()
    preset = MINI_BATCH_AUG_DEFAULTS.get(dataset)

    if preset is None:
        return None, {}, {}

    applied = {}
    skipped = {}

    def set_if_not_explicit(field_name, field_value):
        if field_name in explicit_arg_names:
            skipped[field_name] = getattr(args, field_name)
            return
        setattr(args, field_name, field_value)
        applied[field_name] = field_value

    if aug_tech == "none":
        set_if_not_explicit("dropout", 0.0)
        set_if_not_explicit("aug_mode", "none")
    elif aug_tech == "dropout":
        set_if_not_explicit("dropout", preset["dropout"])
        set_if_not_explicit("aug_mode", "none")
    elif aug_tech == "dropedge":
        set_if_not_explicit("dropedge_rate", preset["dropedge_rate"])
        set_if_not_explicit("p", float(getattr(args, "dropedge_rate", preset["dropedge_rate"])))
        set_if_not_explicit("q", 0.0)
        set_if_not_explicit("aug_mode", "drop")
        set_if_not_explicit("dropout", 0.0)
    elif aug_tech == "dropnode":
        set_if_not_explicit("dropnode_rate", preset["dropnode_rate"])
        set_if_not_explicit("aug_mode", "none")
        set_if_not_explicit("dropout", 0.0)
    elif aug_tech == "dropmessage":
        set_if_not_explicit("dropmessage_rate", preset["dropmessage_rate"])
        set_if_not_explicit("aug_mode", "none")
        set_if_not_explicit("dropout", 0.0)
    elif aug_tech == "rade":
        return (dataset, aug_tech), applied, skipped
    else:
        return None, applied, skipped

    args.ep_correction = False
    args.pq_gradnorm = False
    applied["ep_correction"] = False
    applied["pq_gradnorm"] = False

    return (dataset, aug_tech), applied, skipped


def parse_method(args, num_nodes, num_classes, in_dim, device):
    model = MPNNsMB(
        in_channels=in_dim,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        num_nodes=num_nodes,
        local_layers=args.local_layers,
        dropout=args.dropout,
        pre_linear=args.pre_linear,
        res=args.res,
        ln=args.ln,
        bn=args.bn,
        jk=args.jk,
        gnn=args.gnn,
        ep_correction=args.ep_correction,
        rade_variant=args.rade_variant,
        linear=bool(getattr(args, "linear", False)),
        aug_tech=str(getattr(args, "aug_tech", "none")),
        gat_heads=int(getattr(args, "gat_heads", 1)),
        gat_concat=bool(getattr(args, "gat_concat", True)),
        gat_negative_slope=float(getattr(args, "gat_negative_slope", 0.2)),
        gat_moment_chunk_size=int(getattr(args, "gat_moment_chunk_size", 1024)),
        gat_moment_samples=int(getattr(args, "gat_moment_samples", 256)),
        gat_nonedge_samples=int(getattr(args, "gat_nonedge_samples", 256)),
        gat_moment_seed=int(getattr(args, "gat_moment_seed", 0)),
        gat_ep_corr_clip=float(getattr(args, "gat_ep_corr_clip", 2.0)),
    ).to(device)
    return model


def parser_add_main_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        default="flickr",
        choices=["flickr", "ogbn-arxiv"],
        help="Dataset name",
    )
    parser.add_argument("--data_dir", type=str, default="./data/", help="Root directory")

    parser.add_argument("--device", type=int, default=0, help="GPU id (default: 0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--runs", type=int, default=1, help="number of distinct runs")

    parser.add_argument("--train_prop", type=float, default=0.6, help="training proportion (random-split datasets)")
    parser.add_argument("--valid_prop", type=float, default=0.2, help="validation proportion (random-split datasets)")
    parser.add_argument(
        "--rand_split",
        action="store_true",
        help="Override loader split with a fresh random split (requires main-script support).",
    )
    parser.add_argument(
        "--rand_split_class",
        action="store_true",
        help="Override loader split with class-balanced split (requires main-script support).",
    )
    parser.add_argument("--label_num_per_class", type=int, default=20)
    parser.add_argument("--valid_num", type=int, default=500)
    parser.add_argument("--test_num", type=int, default=1000)

    parser.add_argument("--metric", type=str, default="acc", choices=["acc"], help="evaluation metric")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Number of seed nodes per mini-batch.",
    )
    parser.add_argument(
        "--mb_shuffle",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Shuffle seed nodes each epoch before sampling mini-batches.",
    )
    parser.add_argument("--mb_num_workers", type=int, default=4, help="NeighborLoader workers.")
    parser.add_argument("--mb_pin_memory", action="store_true", help="Pin memory for NeighborLoader.")
    parser.add_argument(
        "--full_eval_cpu",
        action="store_true",
        help="Evaluate on CPU full graph to avoid GPU OOM during evaluation.",
    )
    parser.add_argument("--model", type=str, default="MPNN", choices=["MPNN"])
    parser.add_argument(
        "--gnn",
        type=str,
        default="gcn",
        choices=["gcn", "gin", "gat", "sgc", "gin-linear"],
        help="Backbone: gcn/gin/gat are standard. sgc routes to linear-GCN. gin-linear routes to linear-GIN.",
    )
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--local_layers", type=int, default=2)
    parser.add_argument("--pre_linear", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--res", action="store_true")
    parser.add_argument("--ln", action="store_true")
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--jk", action="store_true")
    parser.add_argument(
        "--linear",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use strictly linear model for supported backbones: disables ReLU/Dropout/BN/LN "
             "and makes GIN MLP linear. Unavailable with --gnn gat.",
    )
    parser.add_argument("--gat_heads", type=int, default=1, help="Number of GAT attention heads. GAT PQ-GradNorm currently requires 1.")
    parser.add_argument("--gat_concat", type=str2bool, default=True, help="Concatenate GAT heads instead of averaging them.")
    parser.add_argument("--gat_negative_slope", type=float, default=0.2, help="LeakyReLU slope for GAT attention.")
    parser.add_argument(
        "--gat_moment_chunk_size",
        type=int,
        default=1024,
        help="Source-node chunk size for mini-batch GAT attention moment computations.",
    )
    parser.add_argument(
        "--gat_moment_samples",
        type=int,
        default=64,
        help="Per-target sampled non-neighbor count for mini-batch GAT denominator moments.",
    )
    parser.add_argument(
        "--gat_nonedge_samples",
        type=int,
        default=64,
        help="Per-target sampled non-neighbor count for mini-batch GAT non-edge corrections and variance.",
    )
    parser.add_argument(
        "--gat_moment_seed",
        type=int,
        default=0,
        help="Deterministic seed for sampled mini-batch GAT moment estimates.",
    )
    parser.add_argument(
        "--gat_ep_corr_clip",
        type=float,
        default=2.0,
        help="GAT RADE EP correction multiplier cap. Use 0 to disable clipping.",
    )

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience in epochs based on validation accuracy. Set <=0 to disable.",
    )

    parser.add_argument(
        "--aug_tech",
        type=str,
        default="rade",
        choices=["rade", "dropout", "dropedge", "dropmessage", "dropnode", "none"],
        help="Which augmentation family to use. "
             "'rade' uses the edge drop/add RADE pipeline. "
             "'dropout' uses feature dropout without graph/message/node augmentation. "
             "'dropedge' drops clean edges without RADE correction. "
             "'dropmessage' uses message dropout. "
             "'dropnode' uses node-wise feature masking. "
             "'none' is clean training without dropout or augmentation.",
    )
    parser.add_argument(
        "--dropedge_rate",
        type=float,
        default=0.2,
        help="DropEdge rate. Used as p when --aug_tech=dropedge.",
    )
    parser.add_argument(
        "--dropmessage_rate",
        type=float,
        default=0.9,
        help="DropMessage rate. Used only when --aug_tech=dropmessage.",
    )
    parser.add_argument(
        "--dropnode_rate",
        type=float,
        default=0.9,
        help="DropNode rate. Used only when --aug_tech=dropnode.",
    )

    parser.add_argument(
        "--aug_mode",
        type=str,
        default="both",
        choices=["none", "drop", "add", "both"],
        help="Graph augmentation mode per epoch.",
    )
    parser.add_argument("--p", type=float, default=0.2, help="Edge drop probability for edges (i,j) in E.")
    parser.add_argument("--q", type=float, default=0.000013446, help="Non-edge add probability.")

    parser.add_argument(
        "--ep_correction",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use expectation-preserving aggregation correction for RADE.",
    )

    parser.add_argument(
        "--pq_gradnorm",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable GradNorm matching for p,q.",
    )
    parser.add_argument(
        "--skip_nonfinite_grad_step",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Skip the optimizer step when the training loss or any model gradient is nonfinite.",
    )
    parser.add_argument(
        "--mask_sharing",
        type=str,
        default="layerwise",
        choices=["shared", "layerwise"],
        help="RADE only: shared or layerwise edge masks.",
    )
    parser.add_argument(
        "--rade_variant",
        type=str,
        default="rade-of",
        choices=["rade-of", "rade-ofs"],
        help="RADE variant: RADE-OF or RADE-OFS.",
    )

    parser.add_argument(
        "--pq_adam_lr",
        type=float,
        default=0.001,
        help="Learning rate for the Adam-based p/q controller.",
    )
    parser.add_argument(
        "--pq_adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1 for the p/q controller.",
    )
    parser.add_argument(
        "--pq_adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 for the p/q controller.",
    )
    parser.add_argument(
        "--pq_adam_eps",
        type=float,
        default=1e-8,
        help="Adam epsilon for the p/q controller.",
    )
    parser.add_argument(
        "--pq_optimize_rho",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="For Adam PQ-GradNorm, optimize rho = q / d directly and convert back to q for G_reg.",
    )
    parser.add_argument(
        "--pq_cap_p",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable an explicit upper bound on the PQ-GradNorm deletion probability p.",
    )
    parser.add_argument(
        "--pq_p_max",
        type=float,
        default=0.9,
        help="Maximum p used when --pq_cap_p is enabled. Must be in [0, 1).",
    )
    parser.add_argument(
        "--pq_cap_rho",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable an explicit upper bound on rho = q / graph_density for PQ-GradNorm.",
    )
    parser.add_argument(
        "--pq_rho_max",
        type=float,
        default=0.2,
        help="Maximum rho used when --pq_cap_rho is enabled. Must be >= 0.",
    )
    parser.add_argument(
        "--pq_deletion_penalty_lambda",
        type=float,
        default=0.0,
        help="Coefficient lambda_p for the soft edge-deletion penalty lambda_p * p^2. "
             "Default 0 disables the penalty.",
    )
    parser.add_argument(
        "--pq_densification_penalty_lambda",
        type=float,
        default=1,
        help="Coefficient lambda for the soft expected-densification penalty applied to rho = q / d.",
    )
    parser.add_argument(
        "--pq_subset_nodes",
        type=int,
        default=0,
        help="Number of train nodes used to estimate GradNorm norms within a batch.",
    )
    parser.add_argument("--pq_update_every", type=int, default=1, help="Update (p,q) every k epochs.")
    parser.add_argument(
        "--pq_update_unit",
        type=str,
        default="batch",
        choices=["batch", "epoch"],
        help="For Adam PQ-GradNorm, update (p,q) by mini-batch interval or once at the end of each eligible epoch.",
    )
    parser.add_argument(
        "--pq_update_every_batches",
        type=int,
        default=1,
        help="For Adam PQ-GradNorm, update (p,q) every K training mini-batches. Default 1 preserves per-batch updates.",
    )
    parser.add_argument(
        "--pq_update_batch_fraction",
        type=float,
        default=0.0,
        help="For Adam PQ-GradNorm, set update interval as ceil(fraction * batches_per_epoch). "
             "Use 0 to rely on --pq_update_every_batches.",
    )
    parser.add_argument("--pq_warmup_epochs", type=int, default=1, help="Number of initial epochs to skip pq updates.")
    parser.add_argument("--pq_eps", type=float, default=1e-12, help="Small epsilon for log/denominator stabilizers.")
    parser.add_argument("--pq_seed", type=int, default=0, help="Seed for selecting the subset of nodes for pq selection.")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="rade-nc-mb")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity/team (optional)")
    parser.add_argument("--wandb_group", type=str, default=None, help="wandb group (optional)")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb run name (optional)")
    parser.add_argument("--wandb_tags", type=str, default="", help="comma-separated tags (optional)")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb mode",
    )
    parser.add_argument("--wandb_log_every", type=int, default=1, help="log every k epochs")

    parser.add_argument("--display_step", type=int, default=1)
    parser.add_argument("--save_model", action="store_true", help="whether to save model (logger.py controls path)")
