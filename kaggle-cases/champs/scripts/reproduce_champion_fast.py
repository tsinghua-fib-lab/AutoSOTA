#!/usr/bin/env python3
"""Run the winner's original ensemble on its archived component predictions.

This reproduces postprocessing, not network inference or training.
"""
import concurrent.futures
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import validate_submission

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Compatibility with NumPy >=1.24; algorithm is unchanged.
    np.bool = np.bool_
    np.int = int
    source = ROOT / "repos/BCAI_kaggle_CHAMPS/src/predictor.py"
    os.chdir(source.parent)
    spec = importlib.util.spec_from_file_location("champion_predictor", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original.settings["RAW_DATA_DIR"] = str(ROOT / "data/raw")
    original.settings["SUBMISSION_DIR"] = str(ROOT / "artifacts/champion_predictions")
    # Parallel decompression changes only loading, not prediction or ensembling.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        loaded = dict(zip(original.models["names"], pool.map(original.load_submission, original.models["names"])))
    original.load_submission = loaded.__getitem__
    ids, predictions = original.ensemble(original.models["names"])
    result = pd.DataFrame({"id": ids.astype(np.int64), "scalar_coupling_constant": predictions})
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    checks = validate_submission(result, test)
    published = pd.read_csv(ROOT / "artifacts/champion_predictions/published_final.csv.gz")
    if not np.array_equal(result.id.values, published.id.values):
        raise ValueError("Archived final submission IDs differ")
    delta = np.abs(result.scalar_coupling_constant.values - published.scalar_coupling_constant.values)
    checks.update({"mode": "original_ensemble_of_archived_component_predictions",
                   "max_absolute_difference_to_published_final": float(delta.max()),
                   "mean_absolute_difference_to_published_final": float(delta.mean()),
                   "exact_within_1e-12": bool((delta < 1e-12).all()),
                   "component_models": len(loaded), "test_labels_used": False})
    if not checks["exact_within_1e-12"]:
        raise ValueError("Winner ensemble did not reproduce archived final predictions: %s" % checks)
    output = ROOT / "submissions/champion_reproduced.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    (ROOT / "artifacts/champion_fast_verification.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
