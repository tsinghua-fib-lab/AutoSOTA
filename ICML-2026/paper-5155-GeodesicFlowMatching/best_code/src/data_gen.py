"""
Check for an existing dataset matching the config; generate if missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cleanup_ssps.dataset_registry import DatasetSpec, ensure_target_dataset
from cleanup_ssps.space_factory import build_ssp_space, resolve_encoded_dim

from src.utils import get_project_root, resolve_config_paths


def ensure_training_data(
    cfg: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root or get_project_root()
    cfg_resolved = resolve_config_paths(cfg, root)

    ssp_section = cfg_resolved["ssp"]
    data_section = cfg_resolved["data"]

    ssp_cfg: dict[str, Any] = dict(ssp_section)
    enc_dim = resolve_encoded_dim(ssp_cfg)
    ssp_cfg["encoded_dim"] = enc_dim

    domain_dim = int(ssp_cfg.get("domain_dim", 2))
    bounds = np.asarray(
        ssp_cfg.get("domain_bounds", [[-1, 1], [-1, 1]]),
        dtype=np.float64,
    )

    ssp_space = build_ssp_space(
        ssp_cfg,
        domain_dim=domain_dim,
        domain_bounds=bounds,
    )

    data_root = cfg_resolved["paths"]["data_root"]
    assert isinstance(data_root, Path)

    spec = DatasetSpec(
        data_root=data_root,
        dataset_type=str(data_section["target_dataset_type"]),
        encoded_dim=enc_dim,
        length_scale=float(ssp_cfg["length_scale"]),
        train_samples=int(data_section["train_samples"]),
        test_samples=int(data_section["test_samples"]),
        sampling_method=str(data_section["dataset_sampling_method"]),
        train_subdir=str(data_section["train_subdir"]),
        test_subdir=str(data_section["test_subdir"]),
    )

    dataset_info = ensure_target_dataset(spec, ssp_cfg, ssp_space)

    return {
        "ssp_space": ssp_space,
        "ssp_config": ssp_cfg,
        "encoded_dim": enc_dim,
        "dataset": dataset_info,
        "paths": cfg_resolved["paths"],
        "project_root": root,
    }

