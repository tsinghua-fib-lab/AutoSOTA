import argparse

from models import MPNNs


def str2bool(value):
    if isinstance(value, bool):
        return value

    value_norm = str(value).strip().lower()
    if value_norm in {"1", "true", "t", "yes", "y"}:
        return True
    if value_norm in {"0", "false", "f", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


FULL_BATCH_AUTO_DEFAULTS = {
    ("cora", "gcn"): {
        "hidden_channels": 512,
        "epochs": 500,
        "p": 0.5,
        "q": 0.000721,
    },
    ("cora", "gin"): {
        "hidden_channels": 512,
        "epochs": 500,
        "p": 0.5,
        "q": 0.000721,
    },
    ("cora", "gat"): {
        "hidden_channels": 256,
        "epochs": 500,
        "p": 0.5,
        "q": 0.000721,
    },
    ("citeseer", "gcn"): {
        "hidden_channels": 512,
        "epochs": 500,
        "p": 0.5,
        "q": 0.00041,
    },
    ("citeseer", "gin"): {
        "hidden_channels": 512,
        "epochs": 500,
        "p": 0.5,
        "q": 0.00041,
    },
    ("citeseer", "gat"): {
        "hidden_channels": 512,
        "epochs": 500,
        "p": 0.5,
        "q": 0.00041,
    },
    ("pubmed", "gcn"): {
        "hidden_channels": 512,
        "epochs": 500,
        "p": 0.5,
        "q": 0.000114,
    },
    ("pubmed", "gin"): {
        "hidden_channels": 512,
        "epochs": 500,
        "p": 0.5,
        "q": 0.000114,
    },
    ("pubmed", "gat"): {
        "hidden_channels": 256,
        "epochs": 500,
        "p": 0.5,
        "q": 0.0000114,
    },
    ("cs", "gcn"): {
        "hidden_channels": 64,
        "epochs": 1500,
        "p": 0.5,
        "q": 0.00023925,
    },
    ("cs", "gin"): {
        "hidden_channels": 256,
        "epochs": 1500,
        "p": 0.5,
        "q": 0.00023925,
    },
    ("cs", "gat"): {
        "hidden_channels": 128,
        "epochs": 1500,
        "p": 0.5,
        "q": 0.00023925,
    },
    ("computer", "gcn"): {
        "hidden_channels": 128,
        "epochs": 1000,
        "p": 0.5,
        "q": 0.001300138,
        "patience": 500,
    },
    ("computer", "gin"): {
        "hidden_channels": 128,
        "epochs": 1000,
        "p": 0.5,
        "q": 0.001300138,
        "patience": 500,
    },
    ("computer", "gat"): {
        "hidden_channels": 128,
        "epochs": 1000,
        "p": 0.5,
        "q": 0.001300138,
        "patience": 500,
    },
    ("physics", "gcn"): {
        "hidden_channels": 64,
        "epochs": 1500,
        "p": 0.5,
        "q": 0.0002085,
    },
    ("physics", "gin"): {
        "hidden_channels": 64,
        "epochs": 1500,
        "p": 0.5,
        "q": 0.0002085,
    },
    ("physics", "gat"): {
        "hidden_channels": 64,
        "epochs": 1500,
        "p": 0.5,
        "q": 0.0002085,
    },
}


FULL_BATCH_AUG_DEFAULTS = {
    "cora": {
        "dropout": 0.7,
        "dropedge_rate": 0.5,
        "dropnode_rate": 0.9,
        "dropmessage_rate": 0.5,
    },
    "citeseer": {
        "dropout": 0.5,
        "dropedge_rate": 0.2,
        "dropnode_rate": 0.9,
        "dropmessage_rate": 0.9,
    },
    "pubmed": {
        "dropout": 0.7,
        "dropedge_rate": 0.2,
        "dropnode_rate": 0.2,
        "dropmessage_rate": 0.9,
    },
    "computer": {
        "dropout": 0.5,
        "dropedge_rate": 0.5,
        "dropnode_rate": 0.5,
        "dropmessage_rate": 0.5,
    },
    "cs": {
        "dropout": 0.3,
        "dropedge_rate": 0.2,
        "dropnode_rate": 0.5,
        "dropmessage_rate": 0.9,
    },
    "physics": {
        "dropout": 0.3,
        "dropedge_rate": 0.2,
        "dropnode_rate": 0.2,
        "dropmessage_rate": 0.2,
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


def apply_full_batch_auto_defaults(args, explicit_arg_names=None):
    explicit_arg_names = explicit_arg_names or set()

    dataset = str(getattr(args, "dataset", "")).lower().strip()
    gnn = str(getattr(args, "gnn", "")).lower().strip()
    preset_key = (dataset, gnn)
    preset = FULL_BATCH_AUTO_DEFAULTS.get(preset_key)

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


def apply_full_batch_aug_defaults(args, explicit_arg_names=None):
    explicit_arg_names = explicit_arg_names or set()

    dataset = str(getattr(args, "dataset", "")).lower().strip()
    aug_tech = str(getattr(args, "aug_tech", "")).lower().strip()
    preset = FULL_BATCH_AUG_DEFAULTS.get(dataset)

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
        set_if_not_explicit("p", preset["dropedge_rate"])
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

    # Non-RADE baselines must not use RADE correction or p/q tuning. Force these
    # even when a shared W&B sweep passes RADE-only flags.
    args.ep_correction = False
    args.pq_gradnorm = False
    applied["ep_correction"] = False
    applied["pq_gradnorm"] = False

    return (dataset, aug_tech), applied, skipped


def parse_method(args, num_nodes, num_classes, in_dim, device):
    model = MPNNs(
        in_channels=in_dim,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        local_layers=args.local_layers,
        dropout=args.dropout,
        pre_linear=args.pre_linear,
        res=args.res,
        ln=args.ln,
        bn=args.bn,
        jk=args.jk,
        gnn=args.gnn,  # 'gcn' or 'gin'
        ep_correction=args.ep_correction,
        rade_variant=args.rade_variant,
        linear=bool(getattr(args, "linear", False)),
        aug_tech=str(getattr(args, "aug_tech", "rade")),
        gat_heads=int(getattr(args, "gat_heads", 1)),
        gat_concat=bool(getattr(args, "gat_concat", True)),
        gat_negative_slope=float(getattr(args, "gat_negative_slope", 0.2)),
        gat_moment_samples=int(getattr(args, "gat_moment_samples", 256)),
        gat_nonedge_samples=int(getattr(args, "gat_nonedge_samples", 256)),
        gat_moment_seed=int(getattr(args, "gat_moment_seed", 0)),
        gat_ep_corr_clip=float(getattr(args, "gat_ep_corr_clip", 2.0)),
    ).to(device)
    return model


def parser_add_main_args(parser):
    # dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=["cora", "citeseer", "pubmed", "cs", "computer", "physics", "flickr", "ogbn-arxiv"],
        help="Dataset name (must match nc_datasets_simple.py)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/",
        help="Root directory passed to load_nc_dataset(root=...)",
    )
    parser.add_argument(
        "--remove_citeseer_isolated_nodes",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Remove isolated nodes when loading CiteSeer. Ignored for other datasets.",
    )

    # system
    parser.add_argument("--device", type=int, default=3, help="GPU id (default: 0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # training schedule
    parser.add_argument("--epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--runs", type=int, default=1, help="number of distinct runs")

    # split control
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

    # metric
    parser.add_argument("--metric", type=str, default="acc", choices=["acc"], help="evaluation metric")

    # model
    parser.add_argument("--model", type=str, default="MPNN", choices=["MPNN"])
    parser.add_argument(
        "--gnn",
        type=str,
        default="gcn",
        choices=["gcn", "gin", "gat", "sgc", "gin-linear"],
        help="Backbone: gcn/gin/gat are standard. sgc routes to linear-GCN. gin-linear routes to linear-GIN."
    )
    parser.add_argument("--hidden_channels", type=int, default=512)
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
    parser.add_argument(
        "--gat_heads",
        type=int,
        default=1,
        help="Number of GAT attention heads. GAT PQ-GradNorm variance currently requires gat_heads=1.",
    )
    parser.add_argument(
        "--gat_concat",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Concatenate GAT heads. If false, heads are averaged.",
    )
    parser.add_argument("--gat_negative_slope", type=float, default=0.2, help="LeakyReLU slope for GAT attention.")
    parser.add_argument(
        "--gat_moment_samples",
        type=int,
        default=256,
        help="Candidate source samples per target for sampled GAT denominator moments.",
    )
    parser.add_argument(
        "--gat_nonedge_samples",
        type=int,
        default=256,
        help="Candidate source samples per target for sampled GAT non-edge centering, OFS inference, and variance sums.",
    )
    parser.add_argument(
        "--gat_moment_seed",
        type=int,
        default=0,
        help="Deterministic sampling seed for sampled GAT moment estimates.",
    )
    parser.add_argument(
        "--gat_ep_corr_clip",
        type=float,
        default=0.0,
        help="GAT RADE EP correction multiplier cap. Use 0 to disable clipping.",
    )

    # optimization
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience in epochs based on validation accuracy. Set <=0 to disable early stopping.",
    )

    # -----------------------
    # Augmentation selector
    # -----------------------
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

    # RADE perturbation args
    parser.add_argument(
        "--aug_mode",
        type=str,
        default="both",
        choices=["none", "drop", "add", "both"],
        help="Per-epoch graph perturbation mode.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Edge-drop probability for clean edges (i,j) in E.",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.000721,
        help="Non-edge-add probability for clean non-edges (i,j) not in E.",
    )

    parser.add_argument(
        "--ep_correction",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Use expectation-preserving aggregation correction for RADE.",
    )

    parser.add_argument(
        "--pq_gradnorm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable epoch-wise GradNorm matching to adapt (p,q).",
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
        help="RADE only: 'shared' uses one keep/add mask across all layers; "
             "'layerwise' samples independent keep/add masks per layer per epoch.",
    )

    parser.add_argument(
        "--rade_variant",
        type=str,
        default="rade-of",
        choices=["rade-of", "rade-ofs"],
        help="RADE variant. "
             "'rade-of' is the overfitting-oriented variant "
             "(center additions at train; clean inference). "
             "'rade-ofs' is the joint overfitting + over-squashing variant "
             "(uncentered additions at train; inference-time augmentation correction).",
    )

    # ---- GradNorm p/q tuning (epoch-wise) ----

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
        help="For online PQ-GradNorm controllers, optimize rho = q / d directly and convert back to q for G_reg.",
    )
    parser.add_argument(
        "--pq_cap_p",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
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
        help="Coefficient lambda_p for the soft edge-deletion penalty lambda_p * p^2 "
             "in the full-batch PQ-GradNorm objective. Default 0 disables the penalty.",
    )
    parser.add_argument(
        "--pq_densification_penalty_lambda",
        type=float,
        default=1.0,
        help="Coefficient lambda for the soft expected-densification penalty "
             "applied to rho = q / d in the full-batch PQ-GradNorm objective, "
             "where d = |E| / (|E| + |Ebar|) is the undirected graph density.",
    )
    parser.add_argument(
        "--pq_subset_nodes",
        type=int,
        default=0,
        help="Number of train nodes used to estimate GradNorm norms. "
             "Use 0 to use the full train_idx in the full-batch setting.",
    )
    parser.add_argument("--pq_update_every", type=int, default=1, help="Update (p,q) every k epochs.")
    parser.add_argument("--pq_warmup_epochs", type=int, default=0, help="Number of initial epochs to skip pq updates.")
    parser.add_argument("--pq_eps", type=float, default=1e-12, help="Small epsilon for log/denominator stabilizers.")
    parser.add_argument("--pq_seed", type=int, default=0, help="Seed for selecting the subset of nodes for pq selection.")

    # -----------------------
    # Weights & Biases
    # -----------------------
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="rade-nc")
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

    # misc
    parser.add_argument("--display_step", type=int, default=1)
    parser.add_argument("--save_model", action="store_true", help="whether to save model (logger.py controls path)")
