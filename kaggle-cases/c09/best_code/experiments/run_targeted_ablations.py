from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


C09_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = C09_ROOT / "data" / "de_train.parquet"
OOF_PATH = C09_ROOT / "experiments" / "ramrt_bayesian_abstention_oof_predictions.npz"
OUTPUT_PATH = Path(__file__).with_name("targeted_ablation_results.json")

SOURCE_CELLS = ["NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"]
TARGET_CELLS = ["B cells", "Myeloid cells"]
SOURCE_COMPONENTS = 16
TARGET_COMPONENTS = 8
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
FOLDS = 5
SEED = 17
BOOTSTRAPS = 20_000


def mr_rmse(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((prediction - truth) ** 2, axis=-1))


def bootstrap_interval(values: np.ndarray, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(BOOTSTRAPS)]
    )
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def adaptive_consensus(source: np.ndarray) -> np.ndarray:
    mean = source.mean(axis=1)
    median = np.median(source, axis=1)
    nk = source[:, 0]
    strength = np.sqrt(np.mean(mean * mean, axis=1))
    nk_strength = np.sqrt(np.mean(nk * nk, axis=1))
    anchor = mean.copy()
    low_strength = strength < 1.987
    high_nk_ratio = nk_strength / np.maximum(strength, 1e-8) >= 1.5
    anchor[low_strength] = median[low_strength]
    use_nk = (~low_strength) & high_nk_ratio
    anchor[use_nk] = 0.75 * nk[use_nk] + 0.25 * mean[use_nk]
    return anchor.astype(np.float32)


def load_complete_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(DATA_PATH)
    metadata = {"cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"}
    genes = [column for column in frame.columns if column not in metadata]
    compounds = sorted(frame["sm_name"].unique())
    cell_order = SOURCE_CELLS + TARGET_CELLS
    values = np.full((len(compounds), len(cell_order), len(genes)), np.nan, dtype=np.float32)
    compound_index = {compound: position for position, compound in enumerate(compounds)}
    cell_index = {cell: position for position, cell in enumerate(cell_order)}
    for _, row in frame.iterrows():
        cell = row["cell_type"]
        if cell in cell_index:
            values[compound_index[row["sm_name"]], cell_index[cell]] = row[genes].to_numpy(
                dtype=np.float32
            )
    observed = np.isfinite(values[..., 0])
    complete_mask = observed.all(axis=1)
    return np.asarray(compounds)[complete_mask], values[complete_mask, :4], values[complete_mask, 4:]


def source_features(train_source: np.ndarray, valid_source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    component_count = min(SOURCE_COMPONENTS, train_source.shape[0] * train_source.shape[1] - 1)
    pca = PCA(n_components=component_count, random_state=SEED)
    pca.fit(train_source.reshape(-1, train_source.shape[-1]))

    def transform(source: np.ndarray) -> np.ndarray:
        scores = pca.transform(source.reshape(-1, source.shape[-1])).reshape(
            source.shape[0], source.shape[1], -1
        )
        strengths = np.sqrt(np.mean(source * source, axis=-1))
        return np.concatenate([scores.reshape(source.shape[0], -1), strengths], axis=1)

    return transform(train_source), transform(valid_source)


def representation_targets(target: np.ndarray, source: np.ndarray) -> dict[str, np.ndarray]:
    anchors = {
        "full_target": np.zeros_like(target),
        "source_mean_residual": source.mean(axis=1),
        "source_median_residual": np.median(source, axis=1),
        "adaptive_consensus_residual": adaptive_consensus(source),
    }
    return {name: target - anchor[:, None, :] for name, anchor in anchors.items()}


def representation_anchors(source: np.ndarray) -> dict[str, np.ndarray]:
    zero = np.zeros((len(source), source.shape[-1]), dtype=np.float32)
    return {
        "full_target": zero,
        "source_mean_residual": source.mean(axis=1),
        "source_median_residual": np.median(source, axis=1),
        "adaptive_consensus_residual": adaptive_consensus(source),
    }


def run_representation_ablation() -> dict:
    compounds, source, target = load_complete_data()
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    representations = list(representation_anchors(source).keys())
    predictions = {
        alpha: {representation: np.zeros_like(target) for representation in representations}
        for alpha in RIDGE_ALPHAS
    }
    fold_records = []
    for fold, (train_index, valid_index) in enumerate(splitter.split(compounds)):
        train_x, valid_x = source_features(source[train_index], source[valid_index])
        train_targets = representation_targets(target[train_index], source[train_index])
        valid_anchors = representation_anchors(source[valid_index])
        for representation in representations:
            train_output = train_targets[representation]
            anchor = valid_anchors[representation]
            for target_position, target_name in enumerate(TARGET_CELLS):
                component_count = min(TARGET_COMPONENTS, len(train_index) - 1)
                output_pca = PCA(n_components=component_count, random_state=SEED)
                train_scores = output_pca.fit_transform(train_output[:, target_position])
                for alpha in RIDGE_ALPHAS:
                    model = Ridge(alpha=alpha)
                    model.fit(train_x, train_scores)
                    predicted = output_pca.inverse_transform(model.predict(valid_x))
                    predictions[alpha][representation][valid_index, target_position] = predicted + anchor
        fold_records.append(
            {
                "fold": fold,
                "train_compounds": compounds[train_index].tolist(),
                "valid_compounds": compounds[valid_index].tolist(),
            }
        )

    results = {}
    for alpha in RIDGE_ALPHAS:
        alpha_result = {}
        for representation in representations:
            errors = mr_rmse(predictions[alpha][representation], target)
            alpha_result[representation] = {
                "combined_mrrmse": float(errors.mean()),
                "target_mrrmse": {
                    target_name: float(errors[:, position].mean())
                    for position, target_name in enumerate(TARGET_CELLS)
                },
                "combined_ci95_by_compound": bootstrap_interval(errors.mean(axis=1), SEED),
            }
        results[str(alpha)] = alpha_result

    primary_alpha = "1.0"
    primary = results[primary_alpha]
    adaptive_error = np.asarray(
        mr_rmse(predictions[1.0]["adaptive_consensus_residual"], target).mean(axis=1)
    )
    comparisons = {}
    for representation in representations:
        other_error = mr_rmse(predictions[1.0][representation], target).mean(axis=1)
        gain = other_error - adaptive_error
        comparisons[representation] = {
            "adaptive_consensus_gain": float(gain.mean()),
            "gain_ci95_by_compound": bootstrap_interval(gain, SEED + 1),
            "adaptive_wins": int((gain > 0).sum()),
            "compound_count": int(len(gain)),
        }
    return {
        "design": {
            "purpose": "Controlled representation probe; not a rerun of the TabPFN final method.",
            "split": f"{FOLDS}-fold compound-grouped CV over complete compounds",
            "matched_conditions": "Identical source features, PCA dimensions, Ridge model, folds, and alpha for every representation.",
            "primary_alpha": 1.0,
            "sensitivity_alphas": list(RIDGE_ALPHAS),
            "source_components": SOURCE_COMPONENTS,
            "target_components": TARGET_COMPONENTS,
        },
        "folds": fold_records,
        "results": results,
        "primary_results": primary,
        "primary_pairwise_comparisons": comparisons,
    }


def summarize_prediction(base: np.ndarray, prediction: np.ndarray, truth: np.ndarray) -> dict:
    base_error = mr_rmse(base, truth)
    model_error = mr_rmse(prediction, truth)
    delta = base_error - model_error
    return {
        "combined_mrrmse": float(model_error.mean()),
        "combined_delta_vs_frozen": float(delta.mean()),
        "combined_ci95_delta_by_compound": bootstrap_interval(delta.mean(axis=0), SEED + 2),
        "combined_wins": int((delta.mean(axis=0) > 0).sum()),
        "targets": {
            target: {
                "mrrmse": float(model_error[position].mean()),
                "delta_vs_frozen": float(delta[position].mean()),
                "ci95_delta_by_compound": bootstrap_interval(delta[position], SEED + 3 + position),
                "wins": int((delta[position] > 0).sum()),
            }
            for position, target in enumerate(TARGET_CELLS)
        },
    }


def run_correction_ablation() -> dict:
    arrays = np.load(OOF_PATH, allow_pickle=True)
    base = arrays["base"]
    truth = arrays["truth"]
    correction = arrays["corrections"].mean(axis=0)
    adaptive_lambdas = arrays["lambdas"]
    fixed_lambdas = np.full(base.shape[:2], 0.05, dtype=np.float64)
    predictions = {
        "frozen_baseline": base,
        "fixed_residual_correction_0.05": base + fixed_lambdas[..., None] * correction,
        "bounded_adaptive_correction": base + adaptive_lambdas[..., None] * correction,
    }
    return {
        "design": {
            "source": OOF_PATH.name,
            "protocol": "Reanalysis of fully nested compound-held-out OOF predictions; target activation and adaptive bounds were selected on inner folds only.",
            "fixed_lambda": 0.05,
            "adaptive_lambda_range": [float(adaptive_lambdas.min()), float(adaptive_lambdas.max())],
            "caution": "This tests the complete nested decision rule. It does not treat the final all-OOF refit as independent evidence.",
        },
        "results": {
            name: summarize_prediction(base, prediction, truth)
            for name, prediction in predictions.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=("all", "ridge", "correction"),
        default="all",
    )
    args = parser.parse_args()
    output = {}
    if args.experiment in {"all", "ridge"}:
        output["representation_ablation"] = run_representation_ablation()
    if args.experiment in {"all", "correction"}:
        output["correction_ablation"] = run_correction_ablation()
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
