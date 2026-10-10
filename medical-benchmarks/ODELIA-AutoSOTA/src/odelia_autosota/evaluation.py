"""Evaluate saved predictions using separately supplied labels."""

from __future__ import annotations
import json
import numpy as np
from .data import sha256_file, validate_ids
from .metrics import compute_metrics


def evaluate_command(args):
    with np.load(args.prediction_archive, allow_pickle=False) as data:
        ids = data["sample_ids"].astype(str)
        probabilities = data["ensemble_probabilities"].copy()
    with np.load(args.labels, allow_pickle=False) as data:
        label_ids = data["sample_ids"].astype(str)
        labels = data["labels"].copy()
    validate_ids(ids, len(probabilities))
    validate_ids(label_ids, len(labels))
    if set(ids) != set(label_ids):
        raise ValueError("prediction and label identifiers do not match")
    lookup = {value: i for i, value in enumerate(label_ids)}
    labels = labels[[lookup[value] for value in ids]]
    result = {
        key.removeprefix("val_"): value
        for key, value in compute_metrics(labels, probabilities).items()
    }
    report = {
        "num_rois": len(ids),
        "metrics": result,
        "predictions_sha256": sha256_file(args.prediction_archive),
        "labels_sha256": sha256_file(args.labels),
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as file:
            file.write(encoded)
    print(encoded, end="")
    return 0
