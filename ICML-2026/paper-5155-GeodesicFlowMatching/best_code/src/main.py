"""
Single-config pipeline: load YAML, data, train, evaluate, optional Weights and Biases logging.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data_gen import ensure_training_data
from src.evaluate import run_evaluation
from src.train import build_trainer_config, run_training
from src.utils import get_project_root, load_config
from utils.wandb_utils import initialize_wandb


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_pipeline(
    config_path: Path | str | None = None,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root or get_project_root()
    cfg = load_config(config_path, project_root=root)
    _set_seed(cfg.get("project", {}).get("seed"))

    data_out = ensure_training_data(cfg, project_root=root)
    ssp_space = data_out["ssp_space"]
    ssp_cfg = data_out["ssp_config"]
    ds = data_out["dataset"]
    paths = data_out["paths"]

    print(
        f"Dataset {'created' if ds['created'] else 'reused'}: "
        f"group={ds.get('dataset_group', '?')} id={ds['dataset_id']}\n"
        f"  dir={ds['dataset_dir']}"
    )
    print(
        "Training loads target .npy files from disk; z0 noise is sampled each batch in SSPDataset "
        "(no on-the-fly SSP encoding for targets)."
    )

    tr = cfg["trainer"]
    device = resolve_device(tr["device"])
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("Using CPU")

    wb = cfg.get("wandb", {})
    if wb.get("enabled", True):
        tags = list(wb.get("tags", [])) + list(cfg.get("experiment", {}).get("tags", []))
        flat_cfg = {**cfg.get("ssp", {}), **tr, **cfg.get("eval", {})}
        initialize_wandb(
            project_name=wb["project"],
            experiment_name=cfg["experiment"]["name"],
            tags=tags,
            config=flat_cfg,
            entity=wb.get("entity"),
        )

    ck_root = paths["checkpoint_dir"]
    if not isinstance(ck_root, Path):
        ck_root = Path(ck_root)
    group = ds.get("dataset_group") or "unknown_dataset_group"
    checkpoint_run_dir = (ck_root / group).resolve()
    checkpoint_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints directory: {checkpoint_run_dir}")

    trainer_cfg = build_trainer_config(
        cfg,
        ds["train_dir"],
        ds["test_dir"],
        checkpoint_dir=checkpoint_run_dir,
    )
    trainer_cfg["device"] = device
    nw = trainer_cfg.get("dataloader_num_workers")
    print(
        f"DataLoader num_workers={nw!r} (None uses OS default: 0 on Windows, 4 on Linux/macOS)"
    )

    t0 = time.time()
    results = run_training(ssp_space, ssp_cfg, trainer_cfg)
    print(f"Training finished in {time.time() - t0:.2f}s")

    run_evaluation(
        results,
        cfg,
        ssp_space,
        test_dir=ds["test_dir"],
        device=device,
        batch_size=tr["batch_size"],
    )

    return {
        "config": cfg,
        "dataset": ds,
        "training_results": results,
        "paths": paths,
        "device": device,
        "checkpoint_dir": checkpoint_run_dir,
    }


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
