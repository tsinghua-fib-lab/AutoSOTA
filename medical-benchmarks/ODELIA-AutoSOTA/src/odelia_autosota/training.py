"""Train seven independent heads and save validation-selected checkpoints."""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from . import heads as candidate
from .heads import FEATURE_DIM, NUM_FOLDS
from .data import _normalize, _softmax, load_training_archive
from .inference import _assert_independent, _predict
from .metrics import compute_metrics

SEED_BASE = 3107


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _checkpoint_payload(
    head: nn.Module, fold: int, best_epoch: int, best_metrics: dict[str, float]
) -> dict[str, Any]:
    return {
        "method": candidate.METHOD_NAME,
        "fold": fold,
        "feature_dim": FEATURE_DIM,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in head.state_dict().items()
        },
        "candidate_metadata": candidate.candidate_metadata(),
    }


def train_command(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays = load_training_archive(args.packed_cache.resolve())
    if args.strict_historical and (
        len(arrays["train_labels"]),
        len(arrays["val_labels"]),
        args.epochs,
        args.batch_size,
    ) != (1208, 206, 300, 128):
        raise ValueError(
            "historical training requires 1208/206 ROIs, 300 epochs, batch size 128"
        )
    device = torch.device(args.device)
    heads = [
        candidate.build_fold_head(fold, FEATURE_DIM).to(device)
        for fold in range(NUM_FOLDS)
    ]
    _assert_independent(heads)
    reports = []
    val_predictions = []
    released_val_predictions = _softmax(arrays["val_released_logits"])
    for fold, head in enumerate(heads):
        seed = SEED_BASE + fold
        set_deterministic(seed)
        optimizer = candidate.make_optimizer(head)
        train_features = torch.from_numpy(arrays["train_features"][:, fold]).float()
        train_logits = torch.from_numpy(
            arrays["train_released_logits"][:, fold]
        ).float()
        train_labels = torch.from_numpy(arrays["train_labels"]).long()
        loader_generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(train_features, train_logits, train_labels),
            batch_size=args.batch_size,
            shuffle=True,
            generator=loader_generator,
            num_workers=0,
            drop_last=False,
        )
        best_epoch = 0
        best_probabilities = _predict(
            head,
            arrays["val_features"][:, fold],
            arrays["val_released_logits"][:, fold],
            fold,
            device,
        )
        best_metrics = compute_metrics(arrays["val_labels"], best_probabilities)
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in head.state_dict().items()
        }
        history = [{"epoch": 0, **best_metrics}]
        if optimizer is not None:
            for epoch in range(1, args.epochs + 1):
                head.train()
                running = 0.0
                for batch_features, batch_logits, batch_labels in loader:
                    batch_features = batch_features.to(device)
                    batch_logits = batch_logits.to(device)
                    batch_labels = batch_labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = candidate.training_loss(
                        head, batch_features, batch_logits, batch_labels, fold
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss fold={fold} epoch={epoch}")
                    loss.backward()
                    optimizer.step()
                    candidate.after_optimizer_step(head, fold)
                    running += float(loss.item()) * len(batch_labels)
                probabilities = _predict(
                    head,
                    arrays["val_features"][:, fold],
                    arrays["val_released_logits"][:, fold],
                    fold,
                    device,
                )
                metrics = compute_metrics(arrays["val_labels"], probabilities)
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": running / len(train_labels),
                        **metrics,
                    }
                )
                better = (
                    metrics["val_challenge_score"]
                    > best_metrics["val_challenge_score"] + 1e-12
                )
                tied = (
                    abs(
                        metrics["val_challenge_score"]
                        - best_metrics["val_challenge_score"]
                    )
                    <= 1e-12
                )
                if better or (tied and metrics["val_nll"] < best_metrics["val_nll"]):
                    best_epoch = epoch
                    best_metrics = dict(metrics)
                    best_probabilities = probabilities.copy()
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in head.state_dict().items()
                    }
                if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
                    print(
                        f"PER_FOLD_TRAIN_PROGRESS fold={fold} epoch={epoch}/{args.epochs} "
                        f"val={metrics['val_challenge_score']:.8f} best={best_metrics['val_challenge_score']:.8f}",
                        flush=True,
                    )
        head.load_state_dict(best_state, strict=True)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=False)
        torch.save(
            _checkpoint_payload(head, fold, best_epoch, best_metrics),
            fold_dir / "head_best.pth",
        )
        report = {
            "fold": fold,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "released_val_metrics": compute_metrics(
                arrays["val_labels"], released_val_predictions[:, fold]
            ),
            "history": history,
        }
        (fold_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        reports.append(report)
        val_predictions.append(best_probabilities)
        print(
            f"PER_FOLD_TRAIN_COMPLETE fold={fold} best_epoch={best_epoch} "
            f"val={best_metrics['val_challenge_score']:.12f}",
            flush=True,
        )
    stacked = np.stack(val_predictions, axis=1)
    ensemble = _normalize(stacked.mean(axis=1))
    released_ensemble = _normalize(released_val_predictions.mean(axis=1))
    summary = {
        "status": "success",
        "method": candidate.METHOD_NAME,
        "parameter_sharing": False,
        "num_fold_heads": NUM_FOLDS,
        "fold_reports": reports,
        "ensemble_metrics": compute_metrics(arrays["val_labels"], ensemble),
        "released_ensemble_metrics": compute_metrics(
            arrays["val_labels"], released_ensemble
        ),
        "candidate_metadata": candidate.candidate_metadata(),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "PER_FOLD_TRAINING_COMPLETE "
        f"validation={summary['ensemble_metrics']['val_challenge_score']:.12f}",
        flush=True,
    )
    return 0
