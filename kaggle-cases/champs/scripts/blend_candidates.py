#!/usr/bin/env python3
"""Predeclared architecture blends; no test targets or tuned test weights."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import validate_submission, TYPES

ROOT = Path(__file__).resolve().parents[1]


def main():
    winner = pd.read_csv(ROOT / "submissions/champion_reproduced.csv")
    runnerup = pd.read_csv(ROOT / "submissions/runnerup_single_reproduced.csv")
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    validate_submission(winner, test)
    validate_submission(runnerup, test)
    reports = []
    for weight in (0.1, 0.25, 0.5):
        output = winner.copy()
        output.scalar_coupling_constant = ((1 - weight) * winner.scalar_coupling_constant.values
                                          + weight * runnerup.scalar_coupling_constant.values)
        filename = "blend_champion_%03d_atomic_%03d.csv" % (round(100 * (1 - weight)), round(100 * weight))
        checks = validate_submission(output, test)
        output.to_csv(ROOT / "submissions" / filename, index=False, float_format="%.10g")
        checks.update({"file": filename, "champion_weight": 1 - weight, "atomic_weight": weight,
                       "weight_selection": "predeclared; no test targets or leaderboard tuning",
                       "kaggle_score": None, "beats_champion": "not yet evaluated"})
        reports.append(checks)
    disagreement = {}
    for t in TYPES:
        mask = test.type.values == t
        delta = winner.scalar_coupling_constant.values[mask] - runnerup.scalar_coupling_constant.values[mask]
        disagreement[t] = {"mean_absolute_prediction_difference": float(np.abs(delta).mean()),
                           "median_absolute_prediction_difference": float(np.median(np.abs(delta)))}
    manifest = {"candidates": reports, "prediction_disagreement_not_error": disagreement,
                "test_labels_used": False,
                "primary_candidate": "blend_champion_075_atomic_025.csv",
                "limitation": "No unbiased OOF predictions from the all-train pretrained checkpoints. Improvement requires Kaggle scoring or fresh molecule-holdout training."}
    (ROOT / "artifacts/blend_candidates.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
