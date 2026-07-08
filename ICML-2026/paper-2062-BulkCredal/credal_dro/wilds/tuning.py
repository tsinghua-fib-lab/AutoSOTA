from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np

from .heads import HeadConfig, build_head


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinism knobs: may reduce throughput, but helps reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # torch not installed or CPU-only environment
        pass


def _to_torch(x, *, device: str):
    try:
        import torch
    except Exception as e:
        raise ImportError("PyTorch is required for ERM tuning utilities.") from e

    if isinstance(x, torch.Tensor):
        return x.to(device)
    arr = np.asarray(x)
    return torch.from_numpy(arr).to(device)


def _eval_classifier(
    model,
    X,
    y,
    groups=None,
    num_groups: Optional[int] = None,
    metric: Literal["avg_acc", "worst_group_acc"] = "avg_acc",
) -> float:
    try:
        import torch
    except Exception as e:
        raise ImportError("PyTorch is required for evaluation.") from e

    model.eval()
    with torch.no_grad():
        logits = model(X)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float()

        if metric == "avg_acc":
            return float(acc.mean().item())

        if metric == "worst_group_acc":
            if groups is None or num_groups is None:
                raise ValueError("worst_group_acc requires groups and num_groups.")
            g = groups.long()
            worst = 1.0
            for gg in range(int(num_groups)):
                mask = (g == gg)
                if int(mask.sum().item()) == 0:
                    continue
                gacc = float(acc[mask].mean().item())
                worst = min(worst, gacc)
            return float(worst)

        raise ValueError(f"Unknown metric={metric!r}.")


@dataclass(frozen=True)
class ERMTuningConfig:
    """Hyperparameter tuning config for *frozen* ERM selection."""
    # What we tune (small grid only)
    weight_decays: Tuple[float, ...] = (0.0, 1e-5, 1e-4, 1e-3)
    head_configs: Tuple[HeadConfig, ...] = (HeadConfig(kind="linear"), HeadConfig(kind="mlp", hidden_dim=256, dropout=0.0))

    # Training settings for the tuning runs
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 10
    patience: Optional[int] = 3  # early stopping on the selection metric
    metric: Literal["avg_acc", "worst_group_acc"] = "avg_acc"

    # Determinism/perf
    num_workers: int = 0


def tune_erm_hparams(
    *,
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes: int,
    input_dim: int,
    seed: int,
    device: str,
    groups_val=None,
    num_groups: Optional[int] = None,
    cfg: ERMTuningConfig = ERMTuningConfig(),
) -> Dict[str, Any]:
    """Tune (head config, weight decay) using ERM on train and selection on validation.

    This is the *only* place you should select λ/model capacity. All robust methods
    must re-use the returned hyperparameters.

    Importantly: calibration split must NOT be used here.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:
        raise ImportError("PyTorch is required for ERM tuning.") from e

    seed_everything(seed)

    Xtr = _to_torch(X_train, device=device).float()
    ytr = _to_torch(y_train, device=device).long()
    Xva = _to_torch(X_val, device=device).float()
    yva = _to_torch(y_val, device=device).long()

    gva = None
    if groups_val is not None:
        gva = _to_torch(groups_val, device=device).long()

    ds_tr = TensorDataset(Xtr, ytr)
    loader_tr = DataLoader(
        ds_tr,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )

    best = {
        "metric": -1e18,
        "weight_decay": None,
        "head_config": None,
    }
    all_rows: List[Dict[str, Any]] = []

    for head_cfg in cfg.head_configs:
        for wd in cfg.weight_decays:
            seed_everything(seed)  # make runs comparable

            model = build_head(input_dim=input_dim, num_classes=num_classes, cfg=head_cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(wd))
            loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

            best_local = -1e18
            best_state = None
            bad_epochs = 0

            for epoch in range(int(cfg.epochs)):
                model.train()
                for xb, yb in loader_tr:
                    opt.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()

                score = _eval_classifier(
                    model, Xva, yva,
                    groups=gva, num_groups=num_groups,
                    metric=cfg.metric,
                )

                if score > best_local + 1e-12:
                    best_local = score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if cfg.patience is not None and bad_epochs >= int(cfg.patience):
                        break

            # restore best local state (so metric is meaningful if you later export)
            if best_state is not None:
                model.load_state_dict(best_state)

            row = {
                "head_kind": head_cfg.kind,
                "head_hidden_dim": int(head_cfg.hidden_dim),
                "head_dropout": float(head_cfg.dropout),
                "weight_decay": float(wd),
                "val_metric": float(best_local),
                "metric_name": cfg.metric,
            }
            all_rows.append(row)

            if best_local > best["metric"]:
                best = {
                    "metric": float(best_local),
                    "weight_decay": float(wd),
                    "head_config": head_cfg.to_dict(),
                    "metric_name": cfg.metric,
                }

    return {
        "best": best,
        "grid": all_rows,
    }


def get_or_tune_frozen_erm_hparams(
    *,
    cache_dir: Path,
    cache_key: str,
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes: int,
    input_dim: int,
    seed: int,
    device: str,
    groups_val=None,
    num_groups: Optional[int] = None,
    cfg: ERMTuningConfig = ERMTuningConfig(),
) -> Dict[str, Any]:
    """Load tuned ERM hyperparams from disk if present; otherwise tune and cache.

    Recommended cache_key fields:
      dataset, scenario, eps_true, split_seed, embedder_name, (optionally) label_space_version
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key}.json"

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    out = tune_erm_hparams(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        num_classes=num_classes, input_dim=input_dim,
        seed=seed, device=device,
        groups_val=groups_val, num_groups=num_groups,
        cfg=cfg,
    )

    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out
