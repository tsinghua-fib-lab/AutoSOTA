"""
Training orchestration driven by ``configs/config.yaml`` trainer section.
"""
from __future__ import annotations

from typing import Any

from utils.training import TrainingManager


def build_trainer_config(
    cfg: dict[str, Any],
    train_dir: str,
    test_dir: str,
    *,
    checkpoint_dir: Any | None = None,
) -> dict[str, Any]:
    t = cfg["trainer"]
    out = {
        "data_dir": train_dir,
        "test_dir": test_dir,
        "sampling_modes": t["sampling_modes"],
        "device": t["device"],
        "batch_size": t["batch_size"],
        "epochs": t["epochs"],
        "lr": t["lr"],
        "weight_decay": t["weight_decay"],
        "val_split": t["val_split"],
        "noise_type": t["noise_type"],
        "target_type": t["target_type"],
        "signal_strength": t.get("signal_strength", 0.0),
        "sigma_min": t.get("sigma_min", 0.1),
        "beta_min": t.get("beta_min", 0.1),
        "beta_max": t.get("beta_max", 20.0),
        "ot_method": t.get("ot_method", "sinkhorn"),
        "ot_reg": t.get("ot_reg", 0.005),
        "ot_reg_sb": t.get("ot_reg_sb", 0.05),
        "train_feedforward": t.get("train_feedforward", True),
        "dataloader_num_workers": t.get("dataloader_num_workers"),
        "dataloader_prefetch_factor": t.get("dataloader_prefetch_factor", 2),
    }
    if checkpoint_dir is not None:
        out["checkpoint_dir"] = checkpoint_dir
    return out


def run_training(ssp_space, ssp_cfg: dict[str, Any], trainer_cfg: dict[str, Any]):
    tm = TrainingManager(ssp_space, trainer_cfg, ssp_cfg)
    return tm.train()
