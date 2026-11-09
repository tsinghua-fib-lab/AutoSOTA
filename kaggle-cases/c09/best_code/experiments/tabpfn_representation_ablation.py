from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from tabpfn_component_cv import (
    CELLS,
    DATA_PATH,
    SEED,
    SOURCE_CELLS,
    TARGET_CELLS,
    build_anchor,
    build_source_features,
    component_prediction,
    mr_rmse,
)


ROOT = Path("/workspace/c09-single-cell")
OUTPUT_PATH = ROOT / "experiments" / "tabpfn_representation_ablation_results.json"
PREDICTION_PATH = ROOT / "experiments" / "tabpfn_representation_ablation_predictions.npz"
REPRESENTATIONS = (
    "full_target",
    "source_mean_residual",
    "source_median_residual",
    "adaptive_consensus_residual",
)
BOOTSTRAPS = 20_000


def bootstrap_interval(values: np.ndarray, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(BOOTSTRAPS)]
    )
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    frame = pd.read_parquet(DATA_PATH)
    metadata = {"cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"}
    genes = [column for column in frame.columns if column not in metadata]
    drugs = sorted(frame["sm_name"].unique())
    drug_index = {drug: position for position, drug in enumerate(drugs)}
    cell_index = {cell: position for position, cell in enumerate(CELLS)}
    source_indices = [cell_index[cell] for cell in SOURCE_CELLS]
    target_indices = [cell_index[cell] for cell in TARGET_CELLS]

    values = np.full((len(drugs), len(CELLS), len(genes)), np.nan, dtype=np.float32)
    for _, row in frame.iterrows():
        cell = row["cell_type"]
        if cell in cell_index:
            values[drug_index[row["sm_name"]], cell_index[cell]] = row[genes].to_numpy(
                dtype=np.float32
            )
    observed = np.isfinite(values[..., 0])
    adaptive_anchor = build_anchor(values, observed, source_indices)
    source_mean = np.nanmean(values[:, source_indices], axis=1).astype(np.float32)
    source_median = np.nanmedian(values[:, source_indices], axis=1).astype(np.float32)
    zero_anchor = np.zeros_like(adaptive_anchor)
    anchors = {
        "full_target": zero_anchor,
        "source_mean_residual": source_mean,
        "source_median_residual": source_median,
        "adaptive_consensus_residual": adaptive_anchor,
    }
    features = build_source_features(values, observed, source_indices, adaptive_anchor)

    complete = np.asarray(
        [
            drug
            for drug in drugs
            if all(observed[drug_index[drug], target] for target in target_indices)
        ]
    )
    complete_indices = np.asarray([drug_index[drug] for drug in complete])
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions = {
        representation: np.zeros((len(complete), len(TARGET_CELLS), len(genes)), dtype=np.float32)
        for representation in REPRESENTATIONS
    }
    fold_records = []

    for fold, (train_position, valid_position) in enumerate(splitter.split(complete)):
        train_indices = complete_indices[train_position]
        valid_indices = complete_indices[valid_position]
        for representation_position, representation in enumerate(REPRESENTATIONS):
            anchor = anchors[representation]
            for target_position, target_index in enumerate(target_indices):
                train_output = values[train_indices, target_index] - anchor[train_indices]
                predicted_output, components = component_prediction(
                    features[train_indices],
                    train_output,
                    features[valid_indices],
                    SEED + 100_000 * representation_position + 1_000 * fold + 100 * target_position,
                )
                predictions[representation][valid_position, target_position] = (
                    anchor[valid_indices] + predicted_output
                )
        fold_records.append(
            {
                "fold": fold,
                "train_compounds": complete[train_position].tolist(),
                "valid_compounds": complete[valid_position].tolist(),
                "components": components,
            }
        )
        print(f"completed fold {fold}", flush=True)

    truth = values[complete_indices][:, target_indices]
    results = {}
    for representation in REPRESENTATIONS:
        errors = mr_rmse(
            predictions[representation].reshape(-1, len(genes)),
            truth.reshape(-1, len(genes)),
        ).reshape(len(complete), len(TARGET_CELLS))
        results[representation] = {
            "combined_mrrmse": float(errors.mean()),
            "combined_ci95_by_compound": bootstrap_interval(errors.mean(axis=1)),
            "targets": {
                target: {
                    "mrrmse": float(errors[:, position].mean()),
                    "ci95_by_compound": bootstrap_interval(errors[:, position], SEED + position + 1),
                }
                for position, target in enumerate(TARGET_CELLS)
            },
        }

    adaptive_errors = mr_rmse(
        predictions["adaptive_consensus_residual"].reshape(-1, len(genes)),
        truth.reshape(-1, len(genes)),
    ).reshape(len(complete), len(TARGET_CELLS)).mean(axis=1)
    pairwise = {}
    for representation in REPRESENTATIONS:
        errors = mr_rmse(
            predictions[representation].reshape(-1, len(genes)),
            truth.reshape(-1, len(genes)),
        ).reshape(len(complete), len(TARGET_CELLS)).mean(axis=1)
        gain = errors - adaptive_errors
        pairwise[representation] = {
            "adaptive_consensus_gain": float(gain.mean()),
            "gain_ci95_by_compound": bootstrap_interval(gain, SEED + 11),
            "adaptive_wins": int((gain > 0).sum()),
            "compound_count": int(len(gain)),
        }

    report = {
        "method": "Matched componentwise TabPFN representation ablation",
        "protocol": {
            "split": "5-fold compound-grouped cross-validation",
            "matched_conditions": "Same source features, folds, TabPFN estimator count, PCA output dimension, and direct correction strength; only the output representation changes.",
            "seed": SEED,
            "complete_compounds": complete.tolist(),
            "representations": list(REPRESENTATIONS),
        },
        "folds": fold_records,
        "results": results,
        "pairwise_against_adaptive_consensus": pairwise,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(PREDICTION_PATH, compounds=complete, truth=truth, **predictions)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
