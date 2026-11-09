from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor


ROOT = Path("/workspace/c09-single-cell")
DATA_PATH = ROOT / "data" / "de_train.parquet"
MODEL_PATH = ROOT / "models" / "tabpfn-v2-regressor.ckpt"
OUTPUT_PATH = ROOT / "experiments" / "tabpfn_component_cv_results.json"
PREDICTION_PATH = ROOT / "experiments" / "tabpfn_component_cv_predictions.npz"
SEED = 17
SOURCE_COMPONENTS = 16
TARGET_COMPONENTS = 8
SHRINKAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
SOURCE_CELLS = [
    "NK cells",
    "T cells CD4+",
    "T cells CD8+",
    "T regulatory cells",
]
TARGET_CELLS = ["B cells", "Myeloid cells"]
CELLS = SOURCE_CELLS + TARGET_CELLS


def mr_rmse(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((prediction - truth) ** 2, axis=1))


def bootstrap_interval(values: np.ndarray, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(20000)]
    )
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def build_anchor(values: np.ndarray, observed: np.ndarray, source_indices: list[int]) -> np.ndarray:
    rows = []
    nk_position = source_indices[0]
    for drug in range(len(values)):
        available = observed[drug, source_indices]
        source = values[drug, np.asarray(source_indices)[available]]
        mean = source.mean(axis=0)
        median = np.median(source, axis=0)
        strength = float(np.sqrt(np.mean(mean * mean)))
        nk = values[drug, nk_position] if observed[drug, nk_position] else mean
        nk_strength = float(np.sqrt(np.mean(nk * nk)))
        if strength < 1.987:
            anchor = median
        elif nk_strength / max(strength, 1e-8) >= 1.5:
            anchor = 0.75 * nk + 0.25 * mean
        else:
            anchor = mean
        rows.append(anchor)
    return np.stack(rows).astype(np.float32)


def build_source_features(
    values: np.ndarray,
    observed: np.ndarray,
    source_indices: list[int],
    anchors: np.ndarray,
) -> np.ndarray:
    profiles = []
    for drug in range(len(values)):
        for cell in source_indices:
            if observed[drug, cell]:
                profiles.append(values[drug, cell])
    pca = PCA(n_components=SOURCE_COMPONENTS, random_state=SEED)
    pca.fit(np.stack(profiles))
    anchor_scores = pca.transform(anchors)
    features = []
    for drug in range(len(values)):
        parts = []
        masks = []
        strengths = []
        for cell in source_indices:
            if observed[drug, cell]:
                profile = values[drug, cell][None]
                parts.append(pca.transform(profile)[0])
                masks.append(1.0)
                strengths.append(float(np.sqrt(np.mean(profile * profile))))
            else:
                parts.append(anchor_scores[drug])
                masks.append(0.0)
                strengths.append(float(np.sqrt(np.mean(anchors[drug] * anchors[drug]))))
        features.append(np.concatenate(parts + [np.asarray(masks), np.asarray(strengths)]))
    return np.asarray(features, dtype=np.float32)


def component_prediction(
    train_features: np.ndarray,
    train_residuals: np.ndarray,
    valid_features: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, int]:
    component_count = min(TARGET_COMPONENTS, len(train_residuals) - 1)
    pca = PCA(n_components=component_count, random_state=random_state)
    scores = pca.fit_transform(train_residuals)
    predicted_scores = np.zeros((len(valid_features), component_count), dtype=np.float32)
    for component in range(component_count):
        model = TabPFNRegressor(
            n_estimators=2,
            device="cuda",
            model_path=MODEL_PATH,
            random_state=random_state + component,
            n_jobs=1,
            fit_mode="fit_preprocessors",
        )
        model.fit(train_features, scores[:, component])
        predicted_scores[:, component] = model.predict(valid_features)
    return pca.inverse_transform(predicted_scores).astype(np.float32), component_count


def main() -> None:
    frame = pd.read_parquet(DATA_PATH)
    metadata = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]
    genes = [column for column in frame.columns if column not in metadata]
    drugs = sorted(frame["sm_name"].unique())
    drug_index = {drug: idx for idx, drug in enumerate(drugs)}
    cell_index = {cell: idx for idx, cell in enumerate(CELLS)}
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
    anchors = build_anchor(values, observed, source_indices)
    features = build_source_features(values, observed, source_indices, anchors)

    complete = [
        drug
        for drug in drugs
        if all(observed[drug_index[drug], target] for target in target_indices)
    ]
    complete_array = np.asarray(complete)
    complete_indices = np.asarray([drug_index[drug] for drug in complete])
    outer = KFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions: dict[tuple[str, str], np.ndarray] = {}
    fold_records = []

    for outer_fold, (train_pos, valid_pos) in enumerate(outer.split(complete_array)):
        outer_train = complete_array[train_pos]
        outer_valid = complete_array[valid_pos]
        inner = KFold(n_splits=3, shuffle=True, random_state=SEED + outer_fold)
        target_records = {}
        selected_shrinkages = {}

        for target_name, target in zip(TARGET_CELLS, target_indices):
            shrinkage_errors = {shrinkage: [] for shrinkage in SHRINKAGES}
            for inner_fold, (inner_train_pos, inner_valid_pos) in enumerate(inner.split(outer_train)):
                inner_train = outer_train[inner_train_pos]
                inner_valid = outer_train[inner_valid_pos]
                train_indices = np.asarray([drug_index[drug] for drug in inner_train])
                valid_indices = np.asarray([drug_index[drug] for drug in inner_valid])
                train_residuals = values[train_indices, target] - anchors[train_indices]
                correction, _ = component_prediction(
                    features[train_indices],
                    train_residuals,
                    features[valid_indices],
                    SEED + 1000 * outer_fold + 100 * target + inner_fold,
                )
                truth = values[valid_indices, target]
                for shrinkage in SHRINKAGES:
                    prediction = anchors[valid_indices] + shrinkage * correction
                    shrinkage_errors[shrinkage].extend(mr_rmse(prediction, truth).tolist())

            inner_scores = {
                str(shrinkage): float(np.mean(errors))
                for shrinkage, errors in shrinkage_errors.items()
            }
            selected = min(SHRINKAGES, key=lambda value: inner_scores[str(value)])
            selected_shrinkages[target_name] = selected

            train_indices = np.asarray([drug_index[drug] for drug in outer_train])
            valid_indices = np.asarray([drug_index[drug] for drug in outer_valid])
            train_residuals = values[train_indices, target] - anchors[train_indices]
            correction, component_count = component_prediction(
                features[train_indices],
                train_residuals,
                features[valid_indices],
                SEED + 10000 + 1000 * outer_fold + 100 * target,
            )
            for position, drug in enumerate(outer_valid):
                predictions[(drug, target_name)] = (
                    anchors[valid_indices[position]] + selected * correction[position]
                ).astype(np.float32)
            target_records[target_name] = {
                "selected_shrinkage": selected,
                "inner_scores": inner_scores,
                "components": component_count,
            }

        fold_records.append(
            {
                "fold": outer_fold,
                "train": outer_train.tolist(),
                "valid": outer_valid.tolist(),
                "targets": target_records,
            }
        )
        print(
            f"fold={outer_fold} valid={outer_valid.tolist()} selected={selected_shrinkages}",
            flush=True,
        )

    report: dict[str, object] = {
        "method": "source-PCA features plus componentwise TabPFN residual transfer",
        "source_components": SOURCE_COMPONENTS,
        "max_target_components": TARGET_COMPONENTS,
        "estimators": 2,
        "complete_compounds": complete,
        "folds": fold_records,
        "targets": {},
    }
    combined_delta = []
    saved_predictions = {}
    for target_name, target in zip(TARGET_CELLS, target_indices):
        prediction = np.stack([predictions[(drug, target_name)] for drug in complete])
        truth = values[complete_indices, target]
        anchor = anchors[complete_indices]
        anchor_error = mr_rmse(anchor, truth)
        model_error = mr_rmse(prediction, truth)
        delta = anchor_error - model_error
        combined_delta.extend(delta.tolist())
        saved_predictions[target_name] = prediction
        report["targets"][target_name] = {
            "anchor": float(anchor_error.mean()),
            "model": float(model_error.mean()),
            "delta": float(delta.mean()),
            "wins": int((delta > 0).sum()),
            "ci95": bootstrap_interval(delta),
            "per_compound": dict(zip(complete, map(float, delta))),
        }

    combined_delta_array = np.asarray(combined_delta)
    combined_ci = bootstrap_interval(combined_delta_array)
    target_reports = report["targets"]
    report["combined"] = {
        "delta": float(combined_delta_array.mean()),
        "wins": int((combined_delta_array > 0).sum()),
        "ci95": combined_ci,
        "accept": bool(
            combined_ci[0] > 0
            and all(target_reports[target]["delta"] > 0 for target in TARGET_CELLS)
            and all(target_reports[target]["wins"] >= 10 for target in TARGET_CELLS)
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        PREDICTION_PATH,
        compounds=np.asarray(complete),
        **saved_predictions,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
