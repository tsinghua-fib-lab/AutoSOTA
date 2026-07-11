from __future__ import annotations

import argparse
import importlib.util
import inspect
from pathlib import Path


def _load_local_models_gc():
    module_path = Path(__file__).with_name("models_gc.py")
    spec = importlib.util.spec_from_file_location("rade_graph_classification_models_gc", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load models_gc from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GraphMPNN, module.GraphMPNNConfig


try:
    GraphMPNN, GraphMPNNConfig = _load_local_models_gc()
except Exception:
    try:
        from .models_gc import GraphMPNN, GraphMPNNConfig
    except Exception:
        from models_gc import GraphMPNN, GraphMPNNConfig


TU_REFERENCE_DEFAULTS = {
    "seed": 0,
    "epochs": 350,
    "lr": 0.01,
    "patience": 0,
}

OGBG_MOLHIV_REFERENCE_DEFAULTS = {
    "epochs": 350,
    "runs": 5,
    "batch_size": 32,
    "lr": 0.001,
    "patience": 0,
    "hidden_channels": 300,
    "local_layers": 5,
    "pooling": "mean",
    "bn": False,
    "dropout": 0.0,
    "use_ogb_encoders": True,
}

PEPTIDES_FUNC_REFERENCE_DEFAULTS = {
    "epochs": 500,
    "runs": 5,
    "batch_size": 128,
    "lr": 0.001,
    "patience": 100,
    "hidden_channels": 300,
    "local_layers": 5,
    "pooling": "mean",
    "bn": False,
    "dropout": 0.0,
    "use_ogb_encoders": True,
}

GRAPH_CLASSIFICATION_FALLBACK_DEFAULTS = {
    "seed": 42,
    "epochs": 500,
    "batch_size": 128,
    "lr": 0.001,
    "patience": 100,
    # "iters_per_epoch": 50,
}


def _tu_defaults(**overrides):
    return {**TU_REFERENCE_DEFAULTS, **overrides}


def _ogbg_molhiv_defaults(**overrides):
    return {**OGBG_MOLHIV_REFERENCE_DEFAULTS, **overrides}


def _peptides_func_defaults(**overrides):
    return {**PEPTIDES_FUNC_REFERENCE_DEFAULTS, **overrides}


GRAPH_CLASSIFICATION_AUTO_DEFAULTS = {
    ("mutag", "gcn"): _tu_defaults(hidden_channels=32, batch_size=16, p=0.50, q=0.0605),
    ("mutag", "gin"): _tu_defaults(hidden_channels=32, batch_size=32, p=0.50, q=0.0605),
    ("mutag", "gat"): _tu_defaults(hidden_channels=32, batch_size=16, p=0.50, q=0.0605),
    ("proteins", "gcn"): _tu_defaults(hidden_channels=64, batch_size=32, p=0.50, q=0.020338),
    ("proteins", "gin"): _tu_defaults(hidden_channels=32, batch_size=32, p=0.50, q=0.020338),
    ("proteins", "gat"): _tu_defaults(hidden_channels=64, batch_size=16, p=0.50, q=0.020338),
    ("imdb-b", "gcn"): _tu_defaults(hidden_channels=32, batch_size=64, p=0.50, q=0.20438),
    ("imdb-b", "gin"): _tu_defaults(hidden_channels=16, batch_size=32, p=0.50, q=0.20438),
    ("imdb-b", "gat"): _tu_defaults(hidden_channels=32, batch_size=64, p=0.50, q=0.20438),
    ("imdb-m", "gcn"): _tu_defaults(hidden_channels=32, batch_size=64, p=0.50, q=0.288465),
    ("imdb-m", "gin"): _tu_defaults(hidden_channels=16, batch_size=32, p=0.50, q=0.288465),
    ("imdb-m", "gat"): _tu_defaults(hidden_channels=32, batch_size=64, p=0.50, q=0.288465),
    ("ogbg-molhiv", "gcn"): _ogbg_molhiv_defaults(p=0.50, q=0.03558149),
    ("ogbg-molhiv", "gin"): _ogbg_molhiv_defaults(p=0.50, q=0.03558149),
    ("ogbg-molhiv", "gat"): _ogbg_molhiv_defaults(p=0.50, q=0.03558149),
    ("peptides-func", "gcn"): _peptides_func_defaults(p=0.50, q=0.002083),
    ("peptides-func", "gin"): _peptides_func_defaults(p=0.50, q=0.002083),
    ("peptides-func", "gat"): _peptides_func_defaults(p=0.50, q=0.002083),
}

GRAPH_CLASSIFICATION_AUG_DEFAULTS = {
    "mutag": {
        "dropout": 0.50,
        "dropedge_rate": 0.20,
        "dropnode_rate": 0.20,
        "dropmessage_rate": 0.20,
    },
    "proteins": {
        "dropout": 0.50,
        "dropedge_rate": 0.80,
        "dropnode_rate": 0.50,
        "dropmessage_rate": 0.80,
    },
    "imdb-b": {
        "dropout": 0.50,
        "dropedge_rate": 0.20,
        "dropnode_rate": 0.50,
        "dropmessage_rate": 0.50,
    },
    "imdb-m": {
        "dropout": 0.80,
        "dropedge_rate": 0.20,
        "dropnode_rate": 0.80,
        "dropmessage_rate": 0.20,
    },
    "ogbg-molhiv": {
        "dropout": 0.50,
        "dropedge_rate": 0.20,
        "dropnode_rate": 0.20,
        "dropmessage_rate": 0.50,
    },
    "peptides-func": {
        "dropout": 0.50,
        "dropedge_rate": 0.20,
        "dropnode_rate": 0.50,
        "dropmessage_rate": 0.50,
    },
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got '{v}'")


def positive_int_or_auto(v):
    if isinstance(v, str) and v.strip().lower() == "auto":
        return "auto"
    try:
        value = int(v)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Expected a positive integer or 'auto'.") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer or 'auto'.")
    return value


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


def apply_gc_auto_defaults(args, explicit_arg_names=None):
    explicit_arg_names = explicit_arg_names or set()
    dataset = str(getattr(args, "dataset", "")).lower().strip()
    gnn = str(getattr(args, "gnn", "")).lower().strip()
    preset = GRAPH_CLASSIFICATION_AUTO_DEFAULTS.get((dataset, gnn))
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

    return (dataset, gnn), applied, skipped


def apply_gc_aug_defaults(args, explicit_arg_names=None):
    explicit_arg_names = explicit_arg_names or set()
    dataset = str(getattr(args, "dataset", "")).lower().strip()
    aug_tech = str(getattr(args, "aug_tech", "")).lower().strip()
    preset = GRAPH_CLASSIFICATION_AUG_DEFAULTS.get(dataset)
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
        dropedge_rate = float(getattr(args, "dropedge_rate", preset["dropedge_rate"]))
        set_if_not_explicit("p", dropedge_rate)
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

    return (dataset, aug_tech), applied, skipped


def parse_method_gc(args, in_channels: int, out_channels: int, device):
    cfg_kwargs = dict(
        gnn=args.gnn,
        pooling=args.pooling,
        local_layers=args.local_layers,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        bn=bool(getattr(args, "bn", False)),
        use_ogb_encoders=bool(getattr(args, "use_ogb_encoders", False)),
        use_edge_attr=bool(getattr(args, "use_edge_attr", False)),
        pre_linear=bool(getattr(args, "pre_linear", False)),
        ep_correction=bool(getattr(args, "ep_correction", False)),
        rade_variant=str(getattr(args, "rade_variant", "rade-of")),
        correct_self_loop=True,
        linear=bool(getattr(args, "linear", False)),
        aug_tech=str(getattr(args, "aug_tech", "rade")),
        global_num_nodes_total=int(getattr(args, "global_num_nodes_total", 0)),
        gat_heads=int(getattr(args, "gat_heads", 1)),
        gat_concat=bool(getattr(args, "gat_concat", True)),
        gat_negative_slope=float(getattr(args, "gat_negative_slope", 0.2)),
        gat_moment_chunk_size=int(getattr(args, "gat_moment_chunk_size", 1024)),
        gat_moment_samples=int(getattr(args, "gat_moment_samples", 256)),
        gat_nonedge_samples=int(getattr(args, "gat_nonedge_samples", 256)),
        gat_moment_seed=int(getattr(args, "gat_moment_seed", 0)),
        gat_ep_corr_clip=float(getattr(args, "gat_ep_corr_clip", 2.0)),
    )
    supported = set(inspect.signature(GraphMPNNConfig).parameters.keys())
    filtered_cfg_kwargs = {key: value for key, value in cfg_kwargs.items() if key in supported}
    cfg = GraphMPNNConfig(**filtered_cfg_kwargs)
    model = GraphMPNN(in_channels=in_channels, out_channels=out_channels, cfg=cfg).to(device)
    return model


def parser_add_gc_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        default="mutag",
        choices=["mutag", "proteins", "imdb-b", "imdb-m", "ogbg-molhiv", "peptides-func"],
    )
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--seed", type=int, default=GRAPH_CLASSIFICATION_FALLBACK_DEFAULTS["seed"])
    parser.add_argument(
        "--tu_raw_root",
        type=str,
        default="./tu_raw_data",
        help="Root containing RADE-local TU raw data as either dataset/<name>/<name>.txt or dataset.zip.",
    )
    parser.add_argument(
        "--iters_per_epoch",
        type=positive_int_or_auto,
        default='auto',
        help="Number of sampled mini-batches per epoch for TU datasets, or 'auto' for ceil(train_graphs / batch_size).",
    )
    parser.add_argument(
        "--tu_train_seed",
        type=int,
        default=0,
        help="Fixed training seed used for TU datasets, matching the reference script.",
    )

    parser.add_argument("--train_prop", type=float, default=0.8)
    parser.add_argument("--valid_prop", type=float, default=0.1)

    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--full_eval_cpu", action="store_true")

    parser.add_argument("--epochs", type=int, default=GRAPH_CLASSIFICATION_FALLBACK_DEFAULTS["epochs"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=GRAPH_CLASSIFICATION_FALLBACK_DEFAULTS["batch_size"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--patience",
        type=int,
        default=GRAPH_CLASSIFICATION_FALLBACK_DEFAULTS["patience"],
        help="Early stopping patience in epochs based on validation accuracy. Set <=0 to disable early stopping.",
    )

    parser.add_argument(
        "--gnn",
        type=str,
        default="gcn",
        choices=["gcn", "gin", "gat", "sgc", "gin-linear"],
        help="Backbone: gcn/gin/gat are standard. sgc routes to linear-GCN. gin-linear routes to linear-GIN.",
    )
    parser.add_argument("--pooling", type=str, default="sum", choices=["mean", "sum"])
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--local_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bn", action="store_true", help="Enable BatchNorm1d after each conv.")
    parser.add_argument(
        "--linear",
        type=str2bool,
        default=False,
        help="Use the linear version of the model (no activations, no BN, no dropout; GIN uses linear MLP).",
    )
    parser.add_argument("--gat_heads", type=int, default=1, help="Number of GAT attention heads. GAT PQ-GradNorm currently requires 1.")
    parser.add_argument("--gat_concat", type=str2bool, default=True, help="Concatenate GAT heads instead of averaging them.")
    parser.add_argument("--gat_negative_slope", type=float, default=0.2, help="LeakyReLU slope for GAT attention.")
    parser.add_argument(
        "--gat_moment_chunk_size",
        type=int,
        default=1024,
        help="Source-node chunk size for graph-classification GAT attention moment computations.",
    )
    parser.add_argument(
        "--gat_moment_samples",
        type=int,
        default=128,
        help="Per-target sampled non-neighbor count for graph-classification GAT denominator moments.",
    )
    parser.add_argument(
        "--gat_nonedge_samples",
        type=int,
        default=128,
        help="Per-target sampled non-neighbor count for graph-classification GAT non-edge corrections and variance.",
    )
    parser.add_argument(
        "--gat_moment_seed",
        type=int,
        default=0,
        help="Deterministic seed for sampled graph-classification GAT moment estimates.",
    )
    parser.add_argument(
        "--gat_ep_corr_clip",
        type=float,
        default=2.0,
        help="GAT RADE EP correction multiplier cap. Use <=0 to disable.",
    )
    parser.add_argument(
        "--pre_linear",
        action="store_true",
        help="Apply a pre-linear projection from input dim to hidden dim before message passing.",
    )

    parser.add_argument("--use_ogb_encoders", type=str2bool, default=False)
    parser.add_argument("--use_edge_attr", action="store_true")

    parser.add_argument(
        "--aug_tech",
        type=str,
        default="rade",
        choices=["rade", "dropout", "dropedge", "dropmessage", "dropnode", "none"],
        help="Which augmentation family to use.",
    )
    parser.add_argument(
        "--dropedge_rate",
        type=float,
        default=0.2,
        help="DropEdge rate. Used as p when --aug_tech=dropedge unless --p is set explicitly.",
    )
    parser.add_argument("--dropmessage_rate", type=float, default=0.0)
    parser.add_argument("--dropnode_rate", type=float, default=0.0)

    parser.add_argument("--aug_mode", type=str, default="both", choices=["none", "drop", "add", "both"])
    parser.add_argument("--p", type=float, default=0.2, help="Edge drop probability for clean edges (0<=p<1).")
    parser.add_argument("--q", type=float, default=0.01, help="Non-edge add probability over clean complement (0<=q<=1).")
    parser.add_argument(
        "--safety_max_additions",
        type=int,
        default=2_000_000,
        help="Safety cap on expected number of added undirected non-edges per graph.",
    )

    parser.add_argument("--ep_correction", type=str2bool, default=True)
    parser.add_argument(
        "--rade_variant",
        type=str,
        default="rade-of",
        choices=["rade-of", "rade-ofs"],
        help="RADE-OF uses centered additions with clean inference. RADE-OFS uses inference-time correction.",
    )
    parser.add_argument("--mask_sharing", type=str, default="shared", choices=["shared", "layerwise"])

    parser.add_argument("--pq_gradnorm", type=str2bool, default=True)
    parser.add_argument("--pq_warmup_epochs", type=int, default=0)
    parser.add_argument("--pq_update_every", type=int, default=1)
    parser.add_argument(
        "--pq_update_unit",
        type=str,
        default="epoch",
        choices=["batch", "epoch"],
        help="Update Adam PQ-GradNorm (p,q) per mini-batch interval or once at the end of each eligible epoch. "
             "Epoch mode uses the final training mini-batch as the GradNorm probe.",
    )
    parser.add_argument(
        "--pq_update_every_batches",
        type=int,
        default=1,
        help="Update Adam PQ-GradNorm (p,q) every K training mini-batches. Default 1 preserves per-batch updates.",
    )
    parser.add_argument(
        "--pq_update_batch_fraction",
        type=float,
        default=0.0,
        help="Set Adam PQ-GradNorm update interval as ceil(fraction * batches_per_epoch). "
             "Use 0 to rely on --pq_update_every_batches.",
    )
    parser.add_argument("--pq_data_anchor", type=str, default="clean", choices=["clean", "aug"])
    parser.add_argument("--pq_adam_lr", type=float, default=0.001)
    parser.add_argument("--pq_adam_beta1", type=float, default=0.9)
    parser.add_argument("--pq_adam_beta2", type=float, default=0.999)
    parser.add_argument("--pq_adam_eps", type=float, default=1e-8)
    parser.add_argument(
        "--pq_p_max",
        type=float,
        default=-1.0,
        help="Hard upper bound for GradNorm's learned edge-drop probability p. "
             "Use a negative value for auto: 0.9.",
    )
    parser.add_argument(
        "--pq_gat_denom_penalty_lambda",
        type=float,
        default=0.0,
        help="GAT-only penalty weight for unstable random attention denominators. "
             "Set 0 to disable.",
    )
    parser.add_argument(
        "--pq_gat_denom_cv2_threshold",
        type=float,
        default=1.0,
        help="GAT-only hinge threshold for Var[attention denominator] / E[attention denominator]^2.",
    )
    parser.add_argument(
        "--pq_gat_corr_penalty_lambda",
        type=float,
        default=50.0,
        help="GAT-only penalty weight for large EP clean-edge correction gains. Set 0 to disable.",
    )
    parser.add_argument(
        "--pq_gat_corr_threshold",
        type=float,
        default=1.5,
        help="GAT-only hinge threshold for clean_alpha / E[retained clean-edge attention].",
    )
    parser.add_argument(
        "--pq_optimize_rho",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="For Adam PQ-GradNorm, optimize the add-side controller as rho=q/density instead of raw q.",
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
        default=0.1,
        help="Penalty weight applied to rho in the GradNorm objective.",
    )
    parser.add_argument("--lr", type=float, default=GRAPH_CLASSIFICATION_FALLBACK_DEFAULTS["lr"])
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--metric", type=str, default="auto", choices=["auto", "acc", "rocauc", "ap"])
    parser.add_argument("--loss", type=str, default="auto", choices=["auto", "ce", "bce"])

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rade-gc")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_log_every", type=int, default=1)

    parser.add_argument("--display_step", type=int, default=1)
    parser.add_argument("--save_model", action="store_true")
