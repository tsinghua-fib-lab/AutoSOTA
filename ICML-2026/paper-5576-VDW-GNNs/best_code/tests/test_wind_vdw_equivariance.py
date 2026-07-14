#!/usr/bin/env python3
"""
End-to-end equivariance probe for VDWModular on the 3D wind dataset.

Minimal usage:
  python3 tests/test_wind_vdw_equivariance.py \
    --root_dir /path/to/code \
    --weights_dir /path/to/experiment/models/best \
    --reconstruct_Q_rot \
    --exact_data_seed 1234 \
    --exact_rotation_seed 298357

Example with explicit config + tolerances:
  python3 tests/test_wind_vdw_equivariance.py \
    --config config/yaml_files/wind/vdw.yaml \
    --weights_dir /path/to/experiment/models/best \
    --sample_n 400 --mask_prop 0.1 --knn_k 3 --local_pca_k 10 \
    --tol_rel 1e-2 --tol_cos 1e-2 --top_k 30
  # Legacy behavior (rebuild Q from rotated geometry):
  #   add --reconstruct_Q_rot

This script:
- Loads a wind config (layered with experiment.yaml via ConfigManager through run_wind_experiments.prep_config).
- Builds a single wind graph A (default n=400, p=0.1, 3D).
- Builds B by applying a single random 3D rotation R to vector features and y.
- Forms the rotated operator by conjugation: Q_B = (I⊗R) Q_A (I⊗R)^T (blockwise R Q_ij R^T).
- Loads a trained model from a directory containing `model.safetensors`.
- Runs forward passes on A and B while registering hooks on all modules, then checks
  vector-valued intermediate outputs for rotation equivariance.

Vector-valued outputs are detected heuristically (an axis of size 3). Scalars/invariants are ignored.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

# ---------------------------------------------------------------------
# Ensure imports work when running as a script from repo root.
# This repo is organized as <root>/code/<modules>.
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
CODE_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse existing project functions (per user request)
from scripts.python.run_wind_experiments import (  # noqa: E402
    attach_operators,
    prep_config,
    rotate_sparse_block_operator,
)
from data_processing.wind import (  # noqa: E402
    apply_random_3d_rotation,
    create_dataset,
)
from training.prep_model import (  # noqa: E402
    prepare_vdw_model,
    _load_pretrained_weights,
)
from tests.wind_equivariance_helpers import generate_equivariance_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe rotation equivariance layer-by-layer for VDWModular on wind."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/yaml_files/wind/vdw.yaml"),
        help="Path to model YAML (experiment.yaml layered automatically if present).",
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=None,
        help="Optional root override (mirrors run_wind_experiments).",
    )
    parser.add_argument(
        "--weights_dir",
        type=Path,
        required=True,
        help="Directory containing trained weights (must include model.safetensors).",
    )
    parser.add_argument("--knn_k", type=int, default=3)
    parser.add_argument("--sample_n", type=int, default=400)
    parser.add_argument("--mask_prop", type=float, default=0.1)
    parser.add_argument(
        "--local_pca_k",
        type=int,
        default=10,
        help="Neighbor count used for local PCA O-frame construction (matches wind scripts).",
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
        "--device",
        type=str,
        default=None,
        help="Device string (e.g. cpu/cuda). Default: auto.",
    )
    parser.add_argument("--tol_rel", type=float, default=1e-2)
    parser.add_argument("--tol_cos", type=float, default=1e-2)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed for dataset sampling and rotation.",
    )
    parser.add_argument(
        "--rep_idx",
        type=int,
        default=0,
        help=(
            "Replication index used to recreate the sampled graph and rotation RNG "
            "(matches run_wind_experiments.py)."
        ),
    )
    parser.add_argument(
        "--base_split_seed",
        type=int,
        default=None,
        help=(
            "Base seed used for dataset sampling. If omitted, defaults to the "
            "config dataset split_seed. Used to match run_wind_experiments seeding."
        ),
    )
    parser.add_argument(
        "--base_rot_seed",
        type=int,
        default=None,
        help=(
            "Base seed used for rotation evaluation. If omitted, defaults to the "
            "config rotation_seed. Used to match run_wind_experiments seeding."
        ),
    )
    parser.add_argument(
        "--exact_data_seed",
        type=int,
        default=None,
        help=(
            "Exact data split seed to use (skips combo/rep seed reconstruction). "
            "Matches data_split_seed saved in run_wind_experiments."
        ),
    )
    parser.add_argument(
        "--exact_rotation_seed",
        type=int,
        default=None,
        help=(
            "Exact rotation seed to use (skips combo/rep seed reconstruction). "
            "Matches rotation_eval_seed saved in run_wind_experiments."
        ),
    )
    args = parser.parse_args()
    return args


def _random_rotation_matrix_3d(
    *,
    device: torch.device,
    dtype: torch.dtype,
    rng: np.random.RandomState,
) -> torch.Tensor:
    """
    Generate a random 3D rotation matrix using QR decomposition.
    Ensures det(R)=+1.
    """
    A = torch.tensor(rng.randn(3, 3), device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    det = torch.linalg.det(Q)
    if det < 0:
        Q[:, -1] = -Q[:, -1]
    return Q


def main() -> None:
    args = parse_args()

    if args.root_dir is not None:
        root_dir = args.root_dir.expanduser()
        if not root_dir.is_absolute():
            candidate = Path("/" + str(root_dir))
            root_dir = candidate if candidate.exists() else root_dir.resolve()
        args.root_dir = root_dir

    # Build config via the same wind experiment pipeline
    # This expects args to have at least: config, root_dir, dataset, vector_feat_key
    # (prep_config resolves layering and returns TrainingConfig).
    args.dataset = "wind"  # enforce
    args.vector_feat_key = None
    _, config = prep_config(args)

    # Device selection via Accelerate, matching project patterns
    if args.device is not None:
        os.environ["ACCELERATE_USE_CPU"] = "1" if args.device.lower() == "cpu" else "0"

    acc = Accelerator(
        device_placement=False if config.using_pytorch_geo else config.device,
        mixed_precision="no" if config.mixed_precision == "none" else config.mixed_precision,
    )
    device = acc.device

    # ------------------------------------------------------------
    # Match run_wind_experiments.py combo indexing and seeding
    # (or use exact seeds if provided)
    # ------------------------------------------------------------
    combo_idx = None
    base_split_seed = None
    base_rot_seed = None
    dataset_seed = None
    rot_seed_used = None

    if args.exact_data_seed is not None:
        dataset_seed = int(args.exact_data_seed)
    if args.exact_rotation_seed is not None:
        rot_seed_used = int(args.exact_rotation_seed)

    if (dataset_seed is None) or (rot_seed_used is None):
        # run_wind.sh typically enumerates sample_n outer and mask_prop inner:
        # for n in [100,200,300,400]:
        #   for p in [0.1,0.3,0.5]:
        #     combo_idx increments
        sample_n_list: List[int] = [100, 200, 300, 400]
        mask_prop_list: List[float] = [0.1, 0.3, 0.5]

        if int(args.sample_n) not in sample_n_list:
            raise ValueError(
                f"sample_n={args.sample_n} not in expected list {sample_n_list}; "
                "cannot recreate combo_idx."
            )
        # Float matching with tolerance
        mp = float(args.mask_prop)
        mp_idx = None
        for i, p in enumerate(mask_prop_list):
            if abs(float(p) - mp) <= 1e-12:
                mp_idx = i
                break
        if mp_idx is None:
            raise ValueError(
                f"mask_prop={args.mask_prop} not in expected list {mask_prop_list}; "
                "cannot recreate combo_idx."
            )

        n_idx = sample_n_list.index(int(args.sample_n))
        combo_idx = int(n_idx * len(mask_prop_list) + mp_idx)

        # Dataset sampling seed (matches: seed_rng(seed=args.seed+combo_idx, combo_idx, rep_idx))
        # => base_seed + combo_idx*1001 + rep_idx, where base_seed corresponds to args.seed.
        base_split_seed = (
            int(args.base_split_seed)
            if args.base_split_seed is not None
            else int(getattr(config.dataset_config, "split_seed", int(args.seed)))
        )
        if dataset_seed is None:
            dataset_seed = int(base_split_seed + combo_idx * 1001 + int(args.rep_idx))

    # Build dataset A and attach operators
    rng = np.random.RandomState(dataset_seed)
    data_raw = create_dataset(
        root=config.dataset_config.data_dir,
        rng=rng,
        knn_k=int(args.knn_k),
        sample_n=int(args.sample_n),
        mask_prop=float(args.mask_prop),
        vector_feat_key=config.dataset_config.vector_feat_key,
    )
    geom_feat_key = getattr(config.dataset_config, "geom_feat_key", None)
    data_a = attach_operators(
        data_raw,
        vector_feat_key=config.dataset_config.vector_feat_key,
        geom_feat_key=geom_feat_key,
        ablate_scalar_track=getattr(config, "ablate_scalar_track"),
        local_pca_k=int(args.local_pca_k),
    )

    # Build dataset B by rotation; rotate operator by conjugation
    # Rotation seeding (matches: adj_rot_seed = rotation_seed_val + combo_idx*1000 + rep_idx)
    # where rotation_seed_val defaults to config.rotation_seed when present, else falls back to seed (args.seed+combo_idx).
    if rot_seed_used is None:
        base_rot_seed = (
            int(args.base_rot_seed)
            if args.base_rot_seed is not None
            else getattr(config, "rotation_seed", None)
        )
        if base_rot_seed is None:
            # Fallback to the run_wind_experiments behavior when no rotation_seed is set:
            # use 'seed' (which is args.seed + combo_idx). Here we approximate with base_split_seed + combo_idx.
            base_rot_seed = int(base_split_seed + (combo_idx or 0))
        rot_seed_used = int(base_rot_seed + (combo_idx or 0) * 1000 + int(args.rep_idx))
    rot_rng = np.random.RandomState(rot_seed_used)
    data_b, rot_deg, R = apply_random_3d_rotation(
        data=data_a,
        rng=rot_rng,
        vector_feat_key=config.dataset_config.vector_feat_key,
        target_key=config.dataset_config.target_key,
        return_matrix=True,
    )
    if args.reconstruct_Q_rot:
        # Legacy behavior: rebuild operators from scratch on rotated geometry.
        data_b = attach_operators(
            data_b,
            vector_feat_key=config.dataset_config.vector_feat_key,
            geom_feat_key=geom_feat_key,
            ablate_scalar_track=getattr(config, "ablate_scalar_track"),
            local_pca_k=int(args.local_pca_k),
        )
    else:
        # Default: rotate the existing operator by conjugation.
        if not hasattr(data_a, "Q"):
            raise RuntimeError("Expected data_a to have attribute 'Q' after operator attachment.")
        Q_rot = rotate_sparse_block_operator(getattr(data_a, "Q"), R)
        setattr(data_b, "Q", Q_rot)

    # Build a single-graph dataloader for model preparation (reuses prepare_vdw_model)
    loader = PyGDataLoader([data_a], batch_size=1, shuffle=False)
    dataloader_dict: Dict[str, PyGDataLoader] = {"train": loader}
    config, model, _ = prepare_vdw_model(config, dataloader_dict, acc=acc)
    model = model.to(device)

    # Load weights from weights_dir (expects model.safetensors)
    weights_dir = Path(args.weights_dir).expanduser().resolve()
    if not weights_dir.exists():
        raise FileNotFoundError(f"--weights_dir does not exist: {weights_dir}")
    _load_pretrained_weights(model, str(weights_dir), verbosity=1)

    # Make sure lazy modules are materialized
    if hasattr(model, "run_epoch_zero_methods"):
        model.run_epoch_zero_methods(data_a.to(device))

    report_lines = [
        f"weights_dir: {weights_dir}",
        f"dataset: wind  sample_n={args.sample_n}  mask_prop={args.mask_prop}  knn_k={args.knn_k}",
        f"vector_feat_key: {config.dataset_config.vector_feat_key}  target_key: {config.dataset_config.target_key}",
        f"combo_idx: {combo_idx}  rep_idx: {args.rep_idx}",
        f"dataset_seed_used: {dataset_seed}  base_split_seed: {base_split_seed}",
        f"rotation_seed_used: {rot_seed_used}  base_rot_seed: {base_rot_seed}",
    ]
    if args.exact_data_seed is not None:
        report_lines.append(f"exact_data_seed: {args.exact_data_seed}")
    if args.exact_rotation_seed is not None:
        report_lines.append(f"exact_rotation_seed: {args.exact_rotation_seed}")

    report = generate_equivariance_report(
        model=model,
        data_a=data_a,
        data_b=data_b,
        rot_mat=R,
        device=device,
        target_key=str(config.dataset_config.target_key),
        tol_rel=float(args.tol_rel),
        tol_cos=float(args.tol_cos),
        top_k=int(args.top_k),
        rot_deg=float(rot_deg),
        rotation_seed=rot_seed_used,
        extra_lines=report_lines,
    )
    print("\n" + report)


if __name__ == "__main__":
    main()

