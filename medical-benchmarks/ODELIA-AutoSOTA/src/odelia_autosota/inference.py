"""Predict with seven saved heads or the fixed released-model baseline."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from . import heads as candidate
from .heads import FEATURE_DIM, NUM_CLASSES, NUM_FOLDS
from .data import _normalize, inference_inputs, sha256_file


def _assert_independent(heads: list[nn.Module]) -> None:
    if len({id(head) for head in heads}) != NUM_FOLDS:
        raise RuntimeError("fold heads are not distinct module objects")
    owners: dict[int, int] = {}
    for fold, head in enumerate(heads):
        for parameter in head.parameters():
            pointer = int(parameter.untyped_storage().data_ptr())
            previous = owners.setdefault(pointer, fold)
            if previous != fold:
                raise RuntimeError(
                    f"parameter storage shared by folds {previous} and {fold}"
                )


def _predict(
    head: nn.Module,
    features: np.ndarray,
    logits: np.ndarray,
    fold: int,
    device: torch.device,
) -> np.ndarray:
    head.eval()
    feature_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    logit_tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32)).to(device)
    with torch.inference_mode():
        probabilities = candidate.prediction_probabilities(
            head, feature_tensor, logit_tensor, fold
        )
    result = probabilities.detach().cpu().numpy()
    if result.shape != (len(features), NUM_CLASSES):
        raise RuntimeError(f"fold {fold} prediction shape mismatch: {result.shape}")
    return _normalize(result)


def infer_command(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    run_dir = args.run_dir.resolve()
    device = torch.device(args.device)
    sample_ids, features, released_logits, released_probabilities = inference_inputs(
        args
    )
    heads = [
        candidate.build_fold_head(fold, FEATURE_DIM).to(device)
        for fold in range(NUM_FOLDS)
    ]
    _assert_independent(heads)
    fold_predictions = []
    checkpoint_hashes = []
    for fold, head in enumerate(heads):
        checkpoint_path = run_dir / f"fold_{fold}" / "head_best.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("method") != candidate.METHOD_NAME
            or int(checkpoint["fold"]) != fold
            or int(checkpoint["feature_dim"]) != FEATURE_DIM
        ):
            raise RuntimeError(f"fold checkpoint contract mismatch: {checkpoint_path}")
        head.load_state_dict(checkpoint["model_state_dict"], strict=True)
        fold_predictions.append(
            _predict(head, features[:, fold], released_logits[:, fold], fold, device)
        )
        checkpoint_hashes.append(sha256_file(checkpoint_path))
    stacked = np.stack(fold_predictions, axis=1)
    ensemble = _normalize(stacked.mean(axis=1))
    np.savez(
        output_dir / "predictions.npz",
        sample_ids=np.asarray(sample_ids),
        per_fold_probabilities=stacked,
        ensemble_probabilities=ensemble,
        released_per_fold_probabilities=released_probabilities,
    )
    flat_payload = {
        sample_id: {"probabilities": ensemble[position].tolist()}
        for position, sample_id in enumerate(sample_ids)
    }
    predictions_path = output_dir / "ensemble_predictions.json"
    predictions_path.write_text(
        json.dumps(flat_payload, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "status": "inference_complete",
        "method": candidate.METHOD_NAME,
        "test_labels_read": False,
        "num_samples": len(sample_ids),
        "num_fold_heads": NUM_FOLDS,
        "parameter_sharing": False,
        "input_sha256": (
            sha256_file(args.inputs.resolve())
            if args.inputs
            else {
                "predictions": sha256_file(args.predictions.resolve()),
                "features": sha256_file(args.block_features.resolve()),
            }
        ),
        "checkpoint_sha256": checkpoint_hashes,
        "output_predictions_sha256": sha256_file(predictions_path),
    }
    (output_dir / "inference_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PER_FOLD_INFERENCE_COMPLETE samples={len(sample_ids)}", flush=True)
    return 0


def baseline_command(args):
    output = args.output_dir.resolve()
    ids, _, _, probabilities = inference_inputs(args)
    output.mkdir(parents=True, exist_ok=False)
    ensemble = _normalize(probabilities.mean(axis=1))
    np.savez(
        output / "predictions.npz",
        sample_ids=np.asarray(ids),
        per_fold_probabilities=probabilities,
        ensemble_probabilities=ensemble,
    )
    (output / "baseline_report.json").write_text(
        json.dumps(
            {
                "method": "released_seven_fold_identity_baseline",
                "num_samples": len(ids),
                "equal_weight_ensemble": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"BASELINE_COMPLETE samples={len(ids)}")
    return 0
