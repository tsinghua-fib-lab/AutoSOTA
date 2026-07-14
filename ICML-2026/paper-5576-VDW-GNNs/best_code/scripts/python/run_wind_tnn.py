#!/usr/bin/env python3
"""
Run Tangent Bundle (TNN) baseline on the wind reconstruction task with
the same sampling/seed conventions as the VDW experiments, and 
experiment-wide configuration via the (e.g.) 'wind/experiment.yaml' YAML 
config file.

This script mirrors the combo + replication loop in
`run_wind_experiments.py` but uses the hardcoded hyperparameters from
`models/comparisons/tangent_bundle_nn.py` and avoids the config manager.
"""

from __future__ import annotations
import argparse
import random
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple
import yaml
import numpy as np
import torch
import torch.nn as nn

# Ensure project root and code directory are on the path
DIR_PARENTS = Path(__file__).resolve().parents
PROJECT_ROOT, CODE_DIR = DIR_PARENTS[3], DIR_PARENTS[2]
if str(PROJECT_ROOT) not in sys.path:
    print(f"Adding project root ({PROJECT_ROOT}) to sys.path")
    sys.path.append(str(PROJECT_ROOT))
if str(CODE_DIR) not in sys.path:
    print(f"Adding code directory ({CODE_DIR}) to sys.path")
    sys.path.append(str(CODE_DIR))

from data_processing.wind import (  # noqa: E402
    WIND_ROT_MASK_PROP,
    WIND_ROT_SAMPLE_N,
    apply_random_2d_rotation,
    apply_random_3d_rotation,
    create_dataset,
    create_wind_rot_datasets,
    wind_rot_seed_rng,
)
import results_utils  # noqa: E402
from models.nn_utilities import ProjectionMLP  # noqa: E402
from os_utilities import get_unique_path_with_suffix  # noqa: E402

# Hardcoded hyperparameters from their repo ('mainWindSampling.py'), 
# plus some defaults for the runner, to work with our experiment config logic.
from models.comparisons.tangent_bundle_nn import (  # noqa: E402
    MODEL_NAME,
    EPSILON,
    EPSILON_PCA,
    GAMMA,
    FEATURES,
    KAPPA,
    LEARN_RATE,
    LOSS_FUNCTION,
    READOUT_SIGMA,
    SIGMA,
    DEFAULT_REPLICATIONS,
    DEFAULT_KNN_K,
    DEFAULT_SEED,
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_PLATEAU_PATIENCE,
    DEFAULT_PLATEAU_FACTOR,
    DEFAULT_PLATEAU_MIN_LR,
    DEFAULT_PLATEAU_MAX_RESTARTS,
    DEFAULT_VECTOR_FEAT_KEY,
    WEIGHT_DECAY,
    TNN,
    get_laplacians,
    project_data,
    set_hparams,
    READOUT_MLP_HIDDEN_DIMS,
    USE_READOUT_MLP,
)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the TNN wind baseline.
    """
    parser = argparse.ArgumentParser(
        description="Run Tangent Bundle NN baseline for wind reconstruction."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional experiment YAML; fills missing args.",
    )
    parser.add_argument(
        "--use_experiment_params",
        action="store_true",
        help="When set, override TNN hyperparams (lr, weight_decay, scheduler) from the experiment config.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Directory containing wind NetCDF files.",
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=None,
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset key override (e.g., wind or wind_rot). Defaults to config.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=MODEL_NAME,
        help="Folder name under experiments/wind to place all combinations.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=5,
        help="Number of replications per (sample_n, mask_prop) combo.",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=None,
        help="k for k-NN graph construction.",
    )
    parser.add_argument(
        "--sample_n",
        type=str,
        default=None,
        help="Comma-separated ints, e.g., 100,200,300",
    )
    parser.add_argument(
        "--mask_prop",
        type=str,
        default=None,
        help="Comma-separated floats, e.g., 0.5,0.3,0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed (matches VDW runs).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience on validation MSE.",
    )
    parser.add_argument(
        "--plateau_patience",
        type=int,
        default=None,
        help="Patience (epochs) before a plateau restart (LR reduce + reload best).",
    )
    parser.add_argument(
        "--plateau_factor",
        type=float,
        default=None,
        help="LR multiply factor on plateau restart.",
    )
    parser.add_argument(
        "--plateau_min_lr",
        type=float,
        default=None,
        help="Minimum LR clamp on plateau restarts.",
    )
    parser.add_argument(
        "--plateau_max_restarts",
        type=int,
        default=None,
        help="Maximum number of plateau restarts (reload-best + LR reduce).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string; defaults to CUDA if available else CPU.",
    )
    parser.add_argument(
        "--vector_feat_key",
        type=str,
        default=None,
        help="Feature key used in the PyG Data object.",
    )
    parser.add_argument(
        "--vector_feat_dim",
        type=int,
        default=None,
        help="Vector feature dimension (2 or 3). Defaults to config.",
    )
    parser.add_argument(
        "--target_key",
        type=str,
        default=None,
        help="Regression target attribute name (e.g., 'y'); defaults to config.",
    )
    parser.add_argument(
        "--tnn_mode",
        type=str,
        default="tnn",
        choices=["tnn", "gnn"],
        help="Shift-operator mode. If dimensions mismatch, fallback to 'gnn'.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate; defaults to TNN constant.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay; defaults to TNN constant.",
    )
    parser.add_argument(
        "--rotation_seed",
        type=int,
        default=None,
        help="Seed for rotation equivariance evaluation.",
    )
    parser.add_argument(
        "--do_rotation_eval",
        action="store_true",
        help="If set, run post-training rotation evaluation; disabled by default.",
    )
    return parser.parse_args()


def parse_list(arg: str, cast_fn) -> List:
    """
    Parse a comma-separated list string into a list with casting.
    """
    return [cast_fn(x.strip()) for x in arg.split(",") if x.strip()]


def is_wind_rot_dataset(dataset_key: str | None) -> bool:
    """
    Helper to identify the wind_rot dataset selection.
    """
    if dataset_key is None:
        return False
    return dataset_key.lower() == "wind_rot"


def _get_cfg(cfg: dict, path: List[str], default=None):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """
    If a config YAML is provided, fill any None args from the config,
    else fall back to hardcoded defaults.
    """
    cfg = {}
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        if not config_path.exists():
            alt_config = CODE_DIR / "config" / "yaml_files" / Path(args.config)
            if alt_config.exists():
                config_path = alt_config
            else:
                raise FileNotFoundError(
                    f"Config not found at '{config_path}' or '{alt_config}'."
                )
        args.config = config_path
        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            raise RuntimeError(f"Failed to load config yaml: {config_path}") from exc

    defaults = {
        "data_dir": PROJECT_ROOT / "data" / "wind",
        "root_dir": PROJECT_ROOT,
        "replications": DEFAULT_REPLICATIONS,
        "knn_k": DEFAULT_KNN_K,
        "seed": DEFAULT_SEED,
        "epochs": DEFAULT_EPOCHS,
        "patience": DEFAULT_PATIENCE,
        "plateau_patience": DEFAULT_PLATEAU_PATIENCE,
        "plateau_factor": DEFAULT_PLATEAU_FACTOR,
        "plateau_min_lr": DEFAULT_PLATEAU_MIN_LR,
        "plateau_max_restarts": DEFAULT_PLATEAU_MAX_RESTARTS,
        "device": None,
        "vector_feat_key": DEFAULT_VECTOR_FEAT_KEY,
        "target_key": "y",
        "lr": LEARN_RATE,
        "weight_decay": WEIGHT_DECAY,
        "rotation_seed": DEFAULT_SEED,
    }

    dataset_key = args.dataset if args.dataset is not None else _get_cfg(cfg, ["dataset", "dataset"])
    if dataset_key is None:
        dataset_key = "wind"
    args.dataset = dataset_key
    dataset_is_wind_rot = is_wind_rot_dataset(args.dataset)

    def pick(name, cfg_path):
        val = getattr(args, name)
        if val is not None:
            return val
        from_cfg = _get_cfg(cfg, cfg_path)
        if from_cfg is not None:
            return from_cfg
        return defaults[name]

    args.data_dir = pick("data_dir", ["dataset", "data_dir"])
    args.root_dir = pick("root_dir", ["training", "root_dir"])
    args.replications = pick("replications", ["training", "replications"])
    args.knn_k = pick("knn_k", ["dataset", "max_num_neighbors"])
    args.seed = pick("seed", ["dataset", "split_seed"])
    args.epochs = pick("epochs", ["training", "n_epochs"])
    args.patience = pick("patience", ["training", "no_valid_metric_improve_patience"])
    # Optimizer/scheduler pulled from config only if requested
    if args.use_experiment_params:
        args.lr = pick("lr", ["optimizer", "learn_rate"])
        args.weight_decay = pick("weight_decay", ["optimizer", "weight_decay"])
        args.plateau_patience = pick("plateau_patience", ["scheduler", "patience"])
        args.plateau_factor = pick("plateau_factor", ["scheduler", "factor"])
        args.plateau_min_lr = pick("plateau_min_lr", ["scheduler", "min_lr"])
        args.plateau_max_restarts = pick("plateau_max_restarts", ["scheduler", "plateau_max_restarts"])
    else:
        args.lr = defaults["lr"] if args.lr is None else args.lr
        args.weight_decay = defaults["weight_decay"] if args.weight_decay is None else args.weight_decay
        args.plateau_patience = defaults["plateau_patience"] if args.plateau_patience is None else args.plateau_patience
        args.plateau_factor = defaults["plateau_factor"] if args.plateau_factor is None else args.plateau_factor
        args.plateau_min_lr = defaults["plateau_min_lr"] if args.plateau_min_lr is None else args.plateau_min_lr
        args.plateau_max_restarts = defaults["plateau_max_restarts"] if args.plateau_max_restarts is None else args.plateau_max_restarts
    args.device = pick("device", ["training", "device"])
    args.vector_feat_key = pick("vector_feat_key", ["dataset", "vector_feat_key"])
    args.vector_feat_dim = pick("vector_feat_dim", ["dataset", "vector_feat_dim"])
    args.target_key = pick("target_key", ["dataset", "target_key"])
    args.rotation_seed = pick("rotation_seed", ["training", "rotation_seed"])

    # sample_n and mask_prop: allow config to supply lists; else require CLI
    if args.sample_n is None:
        cfg_sample_n = _get_cfg(cfg, ["training", "sample_n"])
        if cfg_sample_n is not None:
            if isinstance(cfg_sample_n, (list, tuple)):
                args.sample_n = ",".join(str(x) for x in cfg_sample_n)
            else:
                args.sample_n = str(cfg_sample_n)
    if dataset_is_wind_rot and args.sample_n is None:
        args.sample_n = str(WIND_ROT_SAMPLE_N)
    if args.mask_prop is None:
        cfg_mask = _get_cfg(cfg, ["training", "mask_prop"])
        if cfg_mask is not None:
            if isinstance(cfg_mask, (list, tuple)):
                args.mask_prop = ",".join(str(x) for x in cfg_mask)
            else:
                args.mask_prop = str(cfg_mask)
    if dataset_is_wind_rot and args.mask_prop is None:
        args.mask_prop = str(WIND_ROT_MASK_PROP)

    if args.sample_n is None or args.mask_prop is None:
        raise ValueError("sample_n and mask_prop must be provided via CLI or config.")

    # Resolve paths relative to project root if not absolute
    if args.data_dir is not None:
        args.data_dir = Path(args.data_dir)
        if not args.data_dir.is_absolute():
            args.data_dir = PROJECT_ROOT / args.data_dir
    if args.root_dir is not None:
        args.root_dir = Path(args.root_dir)
        if not args.root_dir.is_absolute():
            args.root_dir = PROJECT_ROOT / args.root_dir
    if args.config is not None:
        args.config = Path(args.config)
        if not args.config.is_absolute():
            args.config = PROJECT_ROOT / args.config

    return args


def format_combo(sample_n: int, mask_prop: float) -> str:
    """
    Generate a slug for the (sample_n, mask_prop) combo.
    """
    pct = int(round(mask_prop * 100))
    return f"n{sample_n}p{pct:02d}"


def seed_rng(base_seed: int, combo_idx: int, rep_idx: int) -> np.random.RandomState:
    """
    Match the VDW seeding strategy for consistent sampling/splits.
    """
    return np.random.RandomState(base_seed + combo_idx * 1000 + rep_idx)


def set_global_seeds(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str | None) -> torch.device:
    """
    Resolve device preference.
    """
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """
    Count learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_shift_operators(
    coords: np.ndarray,
    mode: str,
) -> Tuple[List[torch.Tensor], List[np.ndarray], int]:
    """
    Build shift operators via sheaf Laplacian, returning (operators, O_i_collection, d_hat).
    Accepts the sheaf Laplacian shape (n * d_hat, n * d_hat); rejects mismatches
    that are not integer multiples of node count.
    """
    try:
        laplacian_result = get_laplacians(
            coords,
            epsilon=EPSILON,
            epsilon_pca=EPSILON_PCA,
            gamma_svd=GAMMA,
            tnn_or_gnn=mode,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"get_laplacians failed for mode='{mode}' with coords shape {coords.shape}"
        ) from exc

    if not isinstance(laplacian_result, (list, tuple)):
        raise RuntimeError(
            f"Unexpected get_laplacians return type: {type(laplacian_result)}"
        )

    if len(laplacian_result) >= 5:
        Delta_n_numpy = laplacian_result[0]
        O_i_collection = laplacian_result[3]
        d_hat = laplacian_result[4]
    else:
        raise RuntimeError(
            f"get_laplacians returned {len(laplacian_result)} values; expected >=5 for mode '{mode}'."
        )

    operator = torch.from_numpy(Delta_n_numpy).to(torch.float32)
    node_count = coords.shape[0]
    op_dim = operator.shape[0]
    if op_dim != node_count and (op_dim % node_count) != 0:
        raise RuntimeError(
            f"Unexpected shift operator dimension {op_dim} for {node_count} nodes; "
            f"cannot validate sheaf Laplacian."
        )
    if op_dim % node_count == 0 and op_dim != node_count:
        d_hat = op_dim // node_count
        print(
            f"[INFO] Shift operator dimension ({op_dim}) implies d_hat={d_hat} "
            f"for {node_count} nodes; accepting sheaf Laplacian."
        )
    operators = len(FEATURES) * [operator]
    return operators, O_i_collection, d_hat


def train_single_replication(
    combo_dir: Path,
    args: argparse.Namespace,
    sample_n: int,
    mask_prop: float,
    knn_k: int,
    seed: int,
    combo_idx: int,
    rep_idx: int,
) -> Path:
    """
    Run one replication for a given (sample_n, mask_prop) combo.
    """
    combo_dir.mkdir(parents=True, exist_ok=True)
    results_path = combo_dir / f"results_{rep_idx}.yaml"
    if results_path.exists():
        raise RuntimeError(
            f"Results already exist for combo {combo_idx} replication {rep_idx}."
        )

    replication_seed = seed + combo_idx * 1000 + rep_idx
    set_global_seeds(replication_seed)

    dataset_is_wind_rot = is_wind_rot_dataset(getattr(args, "dataset", None))
    offline_comp_time = float("nan")
    test_data = None

    if dataset_is_wind_rot:
        rotation_seed_val = args.rotation_seed if args.rotation_seed is not None else seed
        rng = wind_rot_seed_rng(rotation_seed_val, combo_idx=combo_idx, rep_idx=rep_idx)
        train_data, test_data = create_wind_rot_datasets(
            root=str(args.data_dir),
            rng=rng,
            knn_k=knn_k,
            sample_n=sample_n,
            mask_prop=mask_prop,
            vector_feat_key=args.vector_feat_key,
        )
    else:
        rng = np.random.RandomState(replication_seed)
        train_data = create_dataset(
            root=str(args.data_dir),
            rng=rng,
            knn_k=knn_k,
            sample_n=sample_n,
            mask_prop=mask_prop,
            vector_feat_key=args.vector_feat_key,
        )

    coords = train_data.pos.cpu().numpy()
    offline_t0 = time.time()
    operators, O_i_collection, d_hat = build_shift_operators(
        coords=coords,
        mode=args.tnn_mode,
    )
    offline_comp_time = float(time.time() - offline_t0)

    device = select_device(args.device)
    x = getattr(train_data, args.vector_feat_key).to(device).float()
    y = train_data.y.to(device).float()
    train_mask = train_data.train_mask.to(device).bool()
    valid_mask = train_data.valid_mask.to(device).bool()
    test_mask = train_data.test_mask.to(device).bool()

    def project_features_to_sheaf(feat: torch.Tensor, O_i: List[np.ndarray]) -> torch.Tensor:
        """
        Project node features into the sheaf basis using the authors' utility.
        Always embed in the ambient coordinate dim (e.g., 3 for xyz) by zero-padding
        or truncating the full feature vector per node, then apply `project_data`
        once to map each per-node ambient vector via the local orthonormal frames
        O_i into a stacked (n * d_hat, 1) signal aligned with the sheaf Laplacian.
        """
        feat_np = feat.detach().cpu().numpy()
        ambient_dim = O_i[0].shape[0]  # e.g., 3 for xyz
        feat_dim = feat_np.shape[1]
        if feat_dim < ambient_dim:
            feat_np = np.pad(feat_np, ((0, 0), (0, ambient_dim - feat_dim)), mode="constant")
        elif feat_dim > ambient_dim:
            feat_np = feat_np[:, :ambient_dim]

        proj = project_data(feat_np, O_i)  # (n*d_hat,1)
        return torch.from_numpy(proj).to(torch.float32)

    def set_model_operators(model_obj: nn.Module, operators_list: List[torch.Tensor]) -> None:
        """
        Update model and layer shift operators in-place.
        """
        if hasattr(model_obj, "L"):
            model_obj.L = [op.to(device) for op in operators_list]
        for layer_idx, layer in enumerate(model_obj.tnn):
            if hasattr(layer, "L") and layer_idx < len(operators_list):
                layer.L = operators_list[layer_idx]

    x_proj = project_features_to_sheaf(x, O_i_collection).to(device)
    y_proj = project_features_to_sheaf(y, O_i_collection).to(device)
    train_mask_proj = train_mask.repeat_interleave(
        d_hat if train_mask.shape[0] * d_hat == operators[0].shape[0] else 1
    ).to(device)
    valid_mask_proj = valid_mask.repeat_interleave(
        d_hat if valid_mask.shape[0] * d_hat == operators[0].shape[0] else 1
    ).to(device)
    test_mask_proj = test_mask.repeat_interleave(
        d_hat if test_mask.shape[0] * d_hat == operators[0].shape[0] else 1
    ).to(device)

    def reshape_sheaf_output(
        preds: torch.Tensor,
        node_count: int,
        d_hat_val: int,
    ) -> torch.Tensor:
        """
        Reshape stacked sheaf output (n*d_hat, c) into per-node features (n, d_hat).
        """
        if preds.shape[0] != node_count * d_hat_val:
            raise RuntimeError(
                f"Unexpected sheaf output size {preds.shape[0]} for n={node_count}, d_hat={d_hat_val}."
            )
        return preds.view(node_count, d_hat_val, -1).squeeze(-1)

    features_out = FEATURES.copy()
    if features_out[-1] != y_proj.shape[1]:
        features_out = features_out[:-1] + [y_proj.shape[1]]

    train_operator_tensors = [op.to(device) for op in operators]

    hparams = set_hparams(
        n=int(operators[0].shape[0]),
        L=operators,
        in_features=int(x_proj.shape[1]),
        features=features_out,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        sigma=SIGMA,
        readout_sigma=READOUT_SIGMA,
        kappa=KAPPA,
        loss_function=LOSS_FUNCTION,
        device=device,
    )
    model = TNN(**hparams).to(device)
    set_model_operators(model, train_operator_tensors)

    readout_mlp = None
    if USE_READOUT_MLP:
        readout_mlp = ProjectionMLP(
            in_dim=d_hat,
            hidden_dim=READOUT_MLP_HIDDEN_DIMS,
            embedding_dim=int(y.shape[1]),
            activation=nn.ReLU,
            use_batch_norm=False,
            dropout_p=None,
            residual_style=False,
        ).to(device)

    optim_params = list(model.parameters())
    if readout_mlp is not None:
        optim_params.extend(list(readout_mlp.parameters()))
    optimizer = torch.optim.Adam(
        optim_params,
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )
    criterion = LOSS_FUNCTION

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_ctr = 0
    plateau_restarts = 0
    train_time_total = 0.0
    infer_time_total = 0.0
    epochs_run = 0
    last_train_mse = float("nan")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        optimizer.zero_grad()
        preds = model(x_proj)
        if readout_mlp is not None:
            preds_sheaf_nodes = reshape_sheaf_output(
                preds,
                node_count=int(y.shape[0]),
                d_hat_val=d_hat,
            )
            preds = readout_mlp(preds_sheaf_nodes)
            train_numel = max(int(train_mask.sum()) * y.shape[1], 1)
            train_loss_sum = criterion(preds[train_mask], y[train_mask])
        else:
            train_numel = max(int(train_mask_proj.sum()) * y_proj.shape[1], 1)
            train_loss_sum = criterion(preds[train_mask_proj], y_proj[train_mask_proj])
        train_loss = train_loss_sum / float(train_numel)
        last_train_mse = float(train_loss.item())
        train_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_time_total += time.time() - t0

        model.eval()
        with torch.no_grad():
            t1 = time.time()
            val_preds = model(x_proj)
            infer_time_total += time.time() - t1
            if readout_mlp is not None:
                val_sheaf_nodes = reshape_sheaf_output(
                    val_preds,
                    node_count=int(y.shape[0]),
                    d_hat_val=d_hat,
                )
                val_preds = readout_mlp(val_sheaf_nodes)
                val_numel = max(int(valid_mask.sum()) * y.shape[1], 1)
                val_loss_sum = criterion(val_preds[valid_mask], y[valid_mask])
            else:
                val_numel = max(int(valid_mask_proj.sum()) * y_proj.shape[1], 1)
                val_loss_sum = criterion(val_preds[valid_mask_proj], y_proj[valid_mask_proj])
            val_loss = val_loss_sum / float(val_numel)

        # Lightweight epoch log
        print(
            f"[epoch {epoch:04d}] "
            f"train_mse={last_train_mse:.6f} "
            f"val_mse={float(val_loss.item()):.6f} "
            f"best_val={best_val:.6f} "
            f"patience={patience_ctr}/{args.patience}"
        )

        epochs_run += 1
        if val_loss.item() + 1e-12 < best_val:
            best_val = float(val_loss.item())
            best_epoch = epoch
            patience_ctr = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        # Plateau restart: reduce LR and reload best weights after plateau_patience epochs without improvement
        if patience_ctr >= args.plateau_patience and plateau_restarts < args.plateau_max_restarts:
            if best_state is not None:
                model.load_state_dict(best_state)
            for pg in optimizer.param_groups:
                new_lr = max(pg["lr"] * args.plateau_factor, args.plateau_min_lr)
                pg["lr"] = new_lr
            plateau_restarts += 1
            print(
                f"[plateau] restart {plateau_restarts}/{args.plateau_max_restarts} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} best_val={best_val:.6f}"
            )
            patience_ctr = 0

        # Early stopping after max patience with no more restarts available
        if patience_ctr >= args.patience and plateau_restarts >= args.plateau_max_restarts:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"[rep {rep_idx}] best_epoch={best_epoch} "
        f"best_val_mse={best_val:.6f} "
        f"train_mse_last={last_train_mse:.6f}"
    )

    mean_train_time = train_time_total / epochs_run if epochs_run else float("nan")
    mean_infer_time = infer_time_total / epochs_run if epochs_run else float("nan")
    param_count = count_parameters(model)
    if readout_mlp is not None:
        param_count += count_parameters(readout_mlp)
    test_mse = float("nan")
    region_test_mse = float("nan")
    if int(test_mask.sum()) > 0:
        model.eval()
        with torch.no_grad():
            preds_test = model(x_proj)
            if readout_mlp is not None:
                preds_test_nodes = reshape_sheaf_output(
                    preds_test,
                    node_count=int(y.shape[0]),
                    d_hat_val=d_hat,
                )
                preds_test = readout_mlp(preds_test_nodes)
                test_numel = max(int(test_mask.sum()) * y.shape[1], 1)
                test_loss_sum = criterion(preds_test[test_mask], y[test_mask])
            else:
                test_numel = max(int(test_mask_proj.sum()) * y_proj.shape[1], 1)
                test_loss_sum = criterion(preds_test[test_mask_proj], y_proj[test_mask_proj])
            test_mse = float((test_loss_sum / float(test_numel)).item())

    if dataset_is_wind_rot and test_data is not None:
        test_coords = test_data.pos.cpu().numpy()
        operators_test, O_i_collection_test, d_hat_test = build_shift_operators(
            coords=test_coords,
            mode=args.tnn_mode,
        )
        test_L_tensors = [op.to(device) for op in operators_test]
        set_model_operators(model, test_L_tensors)

        x_test = getattr(test_data, args.vector_feat_key).to(device).float()
        y_test = test_data.y.to(device).float()
        mask_test = test_data.test_mask.to(device).bool()
        x_proj_test = project_features_to_sheaf(x_test, O_i_collection_test).to(device)
        y_proj_test = project_features_to_sheaf(y_test, O_i_collection_test).to(device)
        mask_proj_test = mask_test.repeat_interleave(
            d_hat_test if mask_test.shape[0] * d_hat_test == operators_test[0].shape[0] else 1
        ).to(device)

        model.eval()
        with torch.no_grad():
            preds_test = model(x_proj_test)
            if readout_mlp is not None:
                preds_test_nodes = reshape_sheaf_output(
                    preds_test,
                    node_count=int(y_test.shape[0]),
                    d_hat_val=d_hat_test,
                )
                preds_test = readout_mlp(preds_test_nodes)
                test_numel = max(int(mask_test.sum()) * y_test.shape[1], 1)
                test_loss_sum = criterion(preds_test[mask_test], y_test[mask_test])
            else:
                test_numel = max(int(mask_proj_test.sum()) * y_proj_test.shape[1], 1)
                test_loss_sum = criterion(preds_test[mask_proj_test], y_proj_test[mask_proj_test])
            region_test_mse = float((test_loss_sum / float(test_numel)).item())

        # Restore training operators for subsequent evaluations
        set_model_operators(model, train_operator_tensors)

    # --------------------------------------------------------------
    # Rotation equivariance evaluation (no retraining)
    # --------------------------------------------------------------
    rot_deg = float("nan")
    rotation_mse = float("nan")
    if args.do_rotation_eval:
        rotation_seed_val = args.rotation_seed if args.rotation_seed is not None else args.seed
        rot_rng = np.random.RandomState(rotation_seed_val + combo_idx * 1000 + rep_idx)
        vector_dim = int(args.vector_feat_dim or 2)
        if vector_dim == 3:
            rot_data, rot_deg = apply_random_3d_rotation(
                data=train_data,
                rng=rot_rng,
                vector_feat_key=args.vector_feat_key,
                target_key=args.target_key,
            )
        else:
            rot_data, rot_deg = apply_random_2d_rotation(
                data=train_data,
                rng=rot_rng,
                vector_feat_key=args.vector_feat_key,
                target_key=args.target_key,
            )

        # Rebuild sheaf Laplacians and local frames for rotated geometry
        rot_coords = rot_data.pos.cpu().numpy()
        operators_rot, O_i_collection_rot, d_hat_rot = build_shift_operators(
            coords=rot_coords,
            mode=args.tnn_mode,
        )
        # Update model shift operators to the rotated ones (preserve learned weights)
        rotated_L_tensors = [op.to(device) for op in operators_rot]
        set_model_operators(model, rotated_L_tensors)

        x_rot = getattr(rot_data, args.vector_feat_key).to(device).float()
        y_rot = rot_data.y.to(device).float()
        mask_rot = rot_data.test_mask.to(device).bool()
        x_proj_rot = project_features_to_sheaf(x_rot, O_i_collection_rot).to(device)
        y_proj_rot = project_features_to_sheaf(y_rot, O_i_collection_rot).to(device)
        mask_proj_rot = mask_rot.repeat_interleave(
            d_hat_rot if mask_rot.shape[0] * d_hat_rot == operators_rot[0].shape[0] else 1
        ).to(device)

        model.eval()
        with torch.no_grad():
            preds_rot = model(x_proj_rot)
            if readout_mlp is not None:
                preds_rot_nodes = reshape_sheaf_output(
                    preds_rot,
                    node_count=int(mask_rot.shape[0]),
                    d_hat_val=d_hat_rot,
                )
                preds_rot = readout_mlp(preds_rot_nodes)
                rot_numel = max(int(mask_rot.sum()) * y_rot.shape[1], 1)
                rot_loss_sum = criterion(preds_rot[mask_rot], y_rot[mask_rot])
            else:
                rot_numel = max(int(mask_proj_rot.sum()) * y_proj_rot.shape[1], 1)
                rot_loss_sum = criterion(preds_rot[mask_proj_rot], y_proj_rot[mask_proj_rot])
            rotation_mse = float((rot_loss_sum / float(rot_numel)).item())

    results = {
        "sample_n": sample_n,
        "mask_prop": mask_prop,
        "replication": rep_idx,
        "parameter_count": param_count,
        "best_epoch": int(best_epoch),
        "mean_train_time": float(mean_train_time),
        "mean_infer_time": float(mean_infer_time),
        "offline_comp_time": float(offline_comp_time),
        "test_mse": float(test_mse),
        "region_test_mse": float(region_test_mse),
        "train_mse": float(last_train_mse),
        "valid_mse": float(best_val),
        "rotation_mse": float(rotation_mse),
        "rotation_degree": float(rot_deg),
    }
    results_utils.write_yaml(results, results_path)
    return results_path


def main() -> None:
    args = parse_args()
    args = apply_config_overrides(args)
    dataset_is_wind_rot = is_wind_rot_dataset(getattr(args, "dataset", None))
    if dataset_is_wind_rot:
        if args.sample_n is None or args.mask_prop is None:
            sample_ns = [WIND_ROT_SAMPLE_N]
            mask_props = [WIND_ROT_MASK_PROP]
        else:
            sample_ns = parse_list(args.sample_n, int)
            mask_props = parse_list(args.mask_prop, float)
    else:
        sample_ns = parse_list(args.sample_n, int)
        mask_props = parse_list(args.mask_prop, float)
    combos = list(enumerate((s, m) for s in sample_ns for m in mask_props))

    root_dir = args.root_dir
    experiment_root = args.dataset if args.dataset is not None else ("wind_rot" if dataset_is_wind_rot else "wind")
    base_dir = root_dir / "experiments" / experiment_root / args.exp_name
    base_dir.mkdir(parents=True, exist_ok=True)

    for combo_idx, (n_val, p_val) in combos:
        combo_slug = format_combo(n_val, p_val)
        combo_dir = base_dir / combo_slug
        result_paths: List[Path] = []
        for rep_idx in range(args.replications):
            rep_seed = args.seed + combo_idx
            path = train_single_replication(
                combo_dir=combo_dir,
                args=args,
                sample_n=n_val,
                mask_prop=p_val,
                knn_k=args.knn_k,
                seed=rep_seed,
                combo_idx=combo_idx,
                rep_idx=rep_idx,
            )
            result_paths.append(path)

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
        aggregated["sample_n"] = n_val
        aggregated["mask_prop"] = p_val
        results_utils.write_yaml(
            aggregated,
            combo_dir / "reps_results.yaml",
        )

    # Print best overall val_mse across combos (mean of means)
    all_vals = [
        v["valid_mse"].get("mean")
        for v in aggregated.values()
        if isinstance(v, dict) and "valid_mse" in v and isinstance(v["valid_mse"], dict)
    ]
    finite_vals = [float(x) for x in all_vals if x is not None and np.isfinite(x)]
    if finite_vals:
        best_overall = min(finite_vals)
        print(f"[DONE] Best overall val_mse (mean across reps): {best_overall:.6f}")
    else:
        print("[DONE] No finite val_mse values to report.")


if __name__ == "__main__":
    main()

