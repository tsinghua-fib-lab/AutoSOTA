"""Validate, pack and align data archives; no model training happens here."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from .heads import FEATURE_DIM, NUM_CLASSES, NUM_FOLDS

FEATURE_KEYS = (
    "stage3_mean",
    "stage3_max",
    "stage3_std",
    "stage4_mean",
    "stage4_max",
    "stage4_std",
    "stage5_mean",
    "stage5_max",
    "stage5_std",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape[-1] != NUM_CLASSES:
        raise RuntimeError(f"invalid class dimension: {probabilities.shape}")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise RuntimeError("invalid candidate probabilities")
    mass = probabilities.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise RuntimeError("zero probability mass")
    return probabilities / mass


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=-1, keepdims=True)
    exponential = np.exp(values)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def _records_by_partition(
    manifest: dict[str, Any], partition: str
) -> list[dict[str, Any]]:
    return [row for row in manifest["records"] if row["partition"] == partition]


def _pack_partition(
    cache_root: Path, manifests: list[dict[str, Any]], partition: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = _records_by_partition(manifests[0], partition)
    sample_ids = [row["sample_id"] for row in records]
    labels = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    if not records:
        raise ValueError(f"empty {partition} partition")
    validate_ids(np.asarray(sample_ids), len(records))
    features_by_fold = []
    logits_by_fold = []
    for fold in range(NUM_FOLDS):
        fold_records = _records_by_partition(manifests[fold], partition)
        if [row["sample_id"] for row in fold_records] != sample_ids:
            raise RuntimeError(f"fold {fold} {partition} sample order mismatch")
        if [int(row["label"]) for row in fold_records] != labels.tolist():
            raise RuntimeError(f"fold {fold} {partition} labels mismatch")
        fold_features = []
        fold_logits = []
        for sample_id in sample_ids:
            sample_path = cache_root / f"fold_{fold}" / "samples" / f"{sample_id}.npz"
            with np.load(sample_path, allow_pickle=False) as payload:
                vector = np.concatenate(
                    [np.asarray(payload[key], dtype=np.float32) for key in FEATURE_KEYS]
                )
                logits = np.asarray(payload["released_logits"], dtype=np.float32)
            if vector.shape != (FEATURE_DIM,) or logits.shape != (NUM_CLASSES,):
                raise RuntimeError(f"invalid cached shapes for fold {fold}")
            fold_features.append(vector)
            fold_logits.append(logits)
        features_by_fold.append(np.stack(fold_features))
        logits_by_fold.append(np.stack(fold_logits))
    features = np.stack(features_by_fold, axis=1)
    logits = np.stack(logits_by_fold, axis=1)
    return np.asarray(sample_ids), labels, features, logits


def pack_command(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite packed cache: {output}")
    cache_root = args.cache_root.resolve()
    manifests = []
    for fold in range(NUM_FOLDS):
        manifest_path = cache_root / f"fold_{fold}" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(payload["fold"]) != fold:
            raise RuntimeError(f"manifest fold mismatch: {manifest_path}")
        manifests.append(payload)
    train_ids, train_labels, train_features, train_logits = _pack_partition(
        cache_root, manifests, "train"
    )
    val_ids, val_labels, val_features, val_logits = _pack_partition(
        cache_root, manifests, "val"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            train_sample_ids=train_ids,
            train_labels=train_labels,
            train_features=train_features,
            train_released_logits=train_logits,
            val_sample_ids=val_ids,
            val_labels=val_labels,
            val_features=val_features,
            val_released_logits=val_logits,
            train_case_ids=np.asarray(
                [r["case_id"] for r in _records_by_partition(manifests[0], "train")]
            ),
            val_case_ids=np.asarray(
                [r["case_id"] for r in _records_by_partition(manifests[0], "val")]
            ),
        )
    os.replace(temporary, output)
    print(
        "PER_FOLD_PACK_COMPLETE "
        f"train={len(train_ids)} val={len(val_ids)} shape={train_features.shape} "
        f"sha256={sha256_file(output)}",
        flush=True,
    )
    return 0


def parse_released_predictions(path: Path) -> tuple[list[str], np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, list[list[float]]] = {}
    for case_id, case_payload in payload["cases"].items():
        for roi_name, roi_payload in case_payload["rois"].items():
            sample_id = f"{case_id}_{roi_name}"
            rows[sample_id] = [
                roi_payload["per_fold"][str(fold)]["probabilities"]
                for fold in range(NUM_FOLDS)
            ]
    sample_ids = sorted(rows)
    probabilities = _normalize(
        np.asarray([rows[sample_id] for sample_id in sample_ids], dtype=np.float64)
    )
    if probabilities.shape != (len(sample_ids), NUM_FOLDS, NUM_CLASSES):
        raise RuntimeError(
            f"test released probability shape mismatch: {probabilities.shape}"
        )
    return sample_ids, probabilities


def inference_inputs(args):
    if args.inputs:
        if args.predictions or args.block_features:
            raise ValueError("use --inputs OR the two legacy input arguments")
        arrays = load_inference_archive(args.inputs.resolve())
        logits = arrays["released_logits"]
        return (
            arrays["sample_ids"].tolist(),
            arrays["features"],
            logits,
            _softmax(logits),
        )
    if not args.predictions or not args.block_features:
        raise ValueError("provide --inputs, or both --predictions and --block-features")
    ids, probabilities = parse_released_predictions(args.predictions.resolve())
    with np.load(args.block_features, allow_pickle=False) as payload:
        feature_ids = payload["sample_ids"].astype(str)
        features = np.asarray(payload["block_features"], dtype=np.float32)
    validate_ids(feature_ids, len(ids))
    if (
        set(feature_ids) != set(ids)
        or features.shape != (len(ids), 7, 2688)
        or not np.isfinite(features).all()
    ):
        raise ValueError("legacy feature identifiers or shape do not match predictions")
    lookup = {value: i for i, value in enumerate(feature_ids)}
    features = np.stack([features[lookup[value]] for value in ids])
    # Historical test path uses log(probability), not re-extracted raw logits.
    logits = np.log(probabilities.clip(1e-8, 1.0)).astype(np.float32)
    return ids, features, logits, probabilities


def validate_ids(ids, n):
    ids = np.asarray(ids)
    if (
        ids.shape != (n,)
        or ids.dtype.kind not in "US"
        or any(not str(x) for x in ids)
        or len(set(ids.tolist())) != n
    ):
        raise ValueError("sample IDs must be unique nonempty strings")


def validate_partition(arrays, prefix):
    labels = arrays[f"{prefix}_labels"]
    n = len(labels)
    if (
        labels.shape != (n,)
        or n == 0
        or labels.dtype.kind not in "iu"
        or not np.isin(labels, [0, 1, 2]).all()
    ):
        raise ValueError("labels must be nonempty integer class IDs 0/1/2")
    validate_ids(arrays[f"{prefix}_sample_ids"], n)
    cases = arrays[f"{prefix}_case_ids"]
    if (
        cases.shape != (n,)
        or cases.dtype.kind not in "US"
        or any(not str(x) for x in cases)
    ):
        raise ValueError("case IDs are required to audit bilateral case overlap")
    for key, shape in (("features", (n, 7, 2688)), ("released_logits", (n, 7, 3))):
        value = arrays[f"{prefix}_{key}"]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid {prefix}_{key} shape or nonfinite value")
        arrays[f"{prefix}_{key}"] = value.astype(np.float32)


def load_training_archive(path: Path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    for prefix in ("train", "val"):
        validate_partition(arrays, prefix)
    for key in ("sample_ids", "case_ids"):
        if set(arrays[f"train_{key}"]) & set(arrays[f"val_{key}"]):
            raise ValueError(f"training/validation {key} overlap")
    if set(arrays["val_labels"].tolist()) != {0, 1, 2}:
        raise ValueError("validation must contain all three classes for macro AUC")
    return arrays


def load_inference_archive(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "labels" in data.files:
            raise ValueError(
                "inference inputs must not include labels; use a separate evaluation archive"
            )
        arrays = {
            key: data[key].copy()
            for key in ("sample_ids", "features", "released_logits")
        }
    n = len(arrays["sample_ids"])
    if n == 0:
        raise ValueError("empty inference cohort")
    validate_ids(arrays["sample_ids"], n)
    for key, shape in (("features", (n, 7, 2688)), ("released_logits", (n, 7, 3))):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"invalid {key} shape or nonfinite value")
        arrays[key] = arrays[key].astype(np.float32)
    return arrays
