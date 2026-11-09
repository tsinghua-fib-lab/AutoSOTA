from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, LeaveOneOut

from ramrt_nested_cv import (
    OUTER_SEED,
    SOURCE_CELLS,
    TARGET_CELLS,
    bootstrap_interval,
    load_data,
    mr_rmse,
    strong_base_prediction,
    truths_and_anchors,
)


ROOT = Path("/workspace/c09-single-cell")
REPORT_PATH = ROOT / "experiments" / "mgcc_nested_cv_results.json"
PREDICTION_PATH = ROOT / "experiments" / "mgcc_nested_cv_predictions.npz"
BASE_SEEDS = (17, 29, 43)
DIRECTION_COUNT = 3


def source_geometry(data) -> tuple[np.ndarray, np.ndarray]:
    source_indices = [data.cell_index[cell] for cell in SOURCE_CELLS]
    features = []
    directions = []
    for drug_position in range(len(data.drugs)):
        raw_profiles = data.values[drug_position, source_indices].astype(np.float64)
        available = np.isfinite(raw_profiles[:, 0])
        profiles = raw_profiles[available]
        anchor = data.anchors[drug_position].astype(np.float64)
        anchor_norm = np.linalg.norm(anchor) + 1e-8
        basis = [anchor / anchor_norm]
        centered = profiles - profiles.mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        reference_rows = centered
        for candidate_position in range(min(len(right), DIRECTION_COUNT + 1)):
            candidate = right[candidate_position].copy()
            for vector in basis:
                candidate -= np.dot(candidate, vector) * vector
            norm = np.linalg.norm(candidate)
            if norm < 1e-8:
                continue
            candidate /= norm
            reference = reference_rows[min(len(basis) - 1, len(reference_rows) - 1)]
            if np.dot(candidate, reference) < 0.0:
                candidate *= -1.0
            basis.append(candidate)
            if len(basis) == DIRECTION_COUNT:
                break
        while len(basis) < DIRECTION_COUNT:
            basis.append(np.zeros_like(anchor))

        strengths = np.sqrt(np.mean(profiles * profiles, axis=1))
        all_strengths = np.full(len(SOURCE_CELLS), strengths.mean(), dtype=np.float64)
        all_strengths[available] = strengths
        normalized_strengths = all_strengths / (strengths.mean() + 1e-8)
        unit = profiles / (np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-8)
        similarities = unit @ unit.T
        upper = similarities[np.triu_indices(len(profiles), 1)]
        disagreement = np.sqrt(np.mean(centered * centered)) / (
            np.sqrt(np.mean(anchor * anchor)) + 1e-8
        )
        features.append(
            np.concatenate(
                [
                    normalized_strengths,
                    available.astype(np.float64),
                    np.asarray(
                        [
                            np.sqrt(np.mean(anchor * anchor)),
                            disagreement,
                            upper.mean(),
                            upper.std(),
                        ]
                    ),
                ]
            )
        )
        directions.append(np.stack(basis))
    return np.asarray(features, dtype=np.float64), np.asarray(directions, dtype=np.float64)


def standardized(train_x: np.ndarray, valid_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0) + 1e-6
    return (train_x - center) / scale, (valid_x - center) / scale


def fit_coefficients(
    train_x: np.ndarray,
    train_coefficients: np.ndarray,
    valid_x: np.ndarray,
) -> np.ndarray:
    train_scaled, valid_scaled = standardized(train_x, valid_x)
    predicted = np.zeros((len(valid_x), train_coefficients.shape[1]), dtype=np.float64)
    for direction in range(train_coefficients.shape[1]):
        model = BayesianRidge(compute_score=False)
        model.fit(train_scaled, train_coefficients[:, direction])
        predicted[:, direction] = model.predict(valid_scaled)
    return predicted


def projected_coefficients(errors: np.ndarray, directions: np.ndarray) -> np.ndarray:
    return np.einsum("ng,nkg->nk", errors, directions)


def reconstruct(coefficients: np.ndarray, directions: np.ndarray) -> np.ndarray:
    return np.einsum("nk,nkg->ng", coefficients, directions)


def cross_fitted_attenuation(
    train_x: np.ndarray,
    errors: np.ndarray,
    directions: np.ndarray,
) -> tuple[float, np.ndarray]:
    coefficients = projected_coefficients(errors, directions)
    predicted = np.zeros_like(coefficients)
    for fit_positions, valid_positions in LeaveOneOut().split(train_x):
        predicted[valid_positions] = fit_coefficients(
            train_x[fit_positions], coefficients[fit_positions], train_x[valid_positions]
        )
    correction = reconstruct(predicted, directions)
    numerator = float(np.sum(errors * correction))
    denominator = float(np.sum(correction * correction)) + 1e-12
    attenuation = float(np.clip(numerator / denominator, 0.0, 1.0))
    return attenuation, coefficients


def summarize(base: np.ndarray, prediction: np.ndarray, truth: np.ndarray, compounds) -> dict:
    base_error = mr_rmse(
        base.reshape(-1, base.shape[-1]), truth.reshape(-1, truth.shape[-1])
    ).reshape(len(TARGET_CELLS), -1)
    model_error = mr_rmse(
        prediction.reshape(-1, prediction.shape[-1]), truth.reshape(-1, truth.shape[-1])
    ).reshape(len(TARGET_CELLS), -1)
    delta = base_error - model_error
    per_compound = delta.mean(axis=0)
    positive_sum = float(np.maximum(per_compound, 0.0).sum())
    max_gain_share = (
        float(np.maximum(per_compound, 0.0).max() / positive_sum)
        if positive_sum > 0.0
        else 1.0
    )
    targets = {}
    for position, target in enumerate(TARGET_CELLS):
        values = delta[position]
        targets[target] = {
            "base": float(base_error[position].mean()),
            "model": float(model_error[position].mean()),
            "delta": float(values.mean()),
            "wins": int((values > 0.0).sum()),
            "ci95": bootstrap_interval(values, OUTER_SEED + position),
            "per_compound": dict(zip(compounds.tolist(), map(float, values))),
        }
    combined_ci = bootstrap_interval(per_compound)
    return {
        "combined_delta": float(per_compound.mean()),
        "combined_wins": int((per_compound > 0.0).sum()),
        "combined_ci95_by_compound": combined_ci,
        "max_single_compound_positive_gain_share": max_gain_share,
        "targets": targets,
        "accept": bool(
            combined_ci[0] > 0.0
            and all(targets[target]["delta"] > 0.0 for target in TARGET_CELLS)
            and all(targets[target]["wins"] >= 10 for target in TARGET_CELLS)
            and max_gain_share <= 0.5
        ),
    }


def run_seed(data, features, directions, base_seed: int) -> tuple[dict, dict]:
    outer = KFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    stored = {}
    fold_records = []
    for outer_fold, (train_positions, valid_positions) in enumerate(outer.split(data.complete)):
        train_drugs = data.complete[train_positions]
        valid_drugs = data.complete[valid_positions]
        train_indices = np.asarray([data.drug_index[drug] for drug in train_drugs])
        valid_indices = np.asarray([data.drug_index[drug] for drug in valid_drugs])

        oof_base = np.zeros(
            (len(TARGET_CELLS), len(train_drugs), len(data.genes)), dtype=np.float32
        )
        inner = KFold(n_splits=3, shuffle=True, random_state=OUTER_SEED + outer_fold)
        for inner_fold, (fit_positions, held_positions) in enumerate(inner.split(train_drugs)):
            oof_base[:, held_positions] = strong_base_prediction(
                data,
                train_drugs[fit_positions],
                train_drugs[held_positions],
                base_seed + 100000 * outer_fold + 10000 * inner_fold,
            )
        train_truth, _ = truths_and_anchors(data, train_drugs)
        valid_truth, _ = truths_and_anchors(data, valid_drugs)
        valid_base = strong_base_prediction(
            data,
            train_drugs,
            valid_drugs,
            base_seed + 900000 + 10000 * outer_fold,
        )

        prediction = valid_base.copy()
        attenuations = {}
        for target_position, target in enumerate(TARGET_CELLS):
            train_errors = train_truth[target_position] - oof_base[target_position]
            attenuation, train_coefficients = cross_fitted_attenuation(
                features[train_indices], train_errors, directions[train_indices]
            )
            valid_coefficients = fit_coefficients(
                features[train_indices], train_coefficients, features[valid_indices]
            )
            correction = reconstruct(valid_coefficients, directions[valid_indices])
            prediction[target_position] += attenuation * correction.astype(np.float32)
            attenuations[target] = attenuation

        for position, drug in enumerate(valid_drugs):
            stored[drug] = {
                "base": valid_base[:, position],
                "truth": valid_truth[:, position],
                "prediction": prediction[:, position],
            }
        fold_summary = summarize(valid_base, prediction, valid_truth, valid_drugs)
        fold_records.append(
            {
                "fold": outer_fold,
                "train": train_drugs.tolist(),
                "valid": valid_drugs.tolist(),
                "attenuations": attenuations,
                "combined_delta": fold_summary["combined_delta"],
            }
        )
        print(
            f"seed={base_seed} outer={outer_fold} attenuation={attenuations} "
            f"delta={fold_summary['combined_delta']:.6f}",
            flush=True,
        )

    arrays = {
        name: np.stack([stored[drug][name] for drug in data.complete], axis=1)
        for name in ("base", "truth", "prediction")
    }
    result = summarize(arrays["base"], arrays["prediction"], arrays["truth"], data.complete)
    return {"seed": base_seed, "folds": fold_records, "summary": result}, arrays


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = load_data()
    features, directions = source_geometry(data)
    seed_reports = []
    seed_arrays = []
    for position, seed in enumerate(BASE_SEEDS):
        report, arrays = run_seed(data, features, directions, seed)
        seed_reports.append(report)
        seed_arrays.append(arrays)
        if position == 0:
            target_results = report["summary"]["targets"]
            if (
                report["summary"]["combined_delta"] <= 0.0
                or any(target_results[target]["delta"] <= 0.0 for target in TARGET_CELLS)
            ):
                break

    ensemble_summary = None
    accepted = False
    if len(seed_arrays) == len(BASE_SEEDS):
        ensemble_base = np.mean([arrays["base"] for arrays in seed_arrays], axis=0)
        ensemble_prediction = np.mean(
            [arrays["prediction"] for arrays in seed_arrays], axis=0
        )
        truth = seed_arrays[0]["truth"]
        ensemble_summary = summarize(
            ensemble_base, ensemble_prediction, truth, data.complete
        )
        accepted = bool(
            ensemble_summary["accept"]
            and all(report["summary"]["combined_delta"] > 0.0 for report in seed_reports)
        )
        np.savez_compressed(
            PREDICTION_PATH,
            compounds=data.complete,
            base=ensemble_base,
            truth=truth,
            prediction=ensemble_prediction,
            seeds=np.asarray(BASE_SEEDS),
        )

    result = {
        "method": "Metric-Geometric Cross-Fit Calibration",
        "correction_space": "drug-specific orthonormal source-response geometry",
        "coefficient_model": "Bayesian ridge on invariant source-geometry features",
        "attenuation": "closed-form risk minimizer from leave-one-compound-out corrections",
        "validation": "5-fold compound-held-out CV; calibration inputs cross-fitted in 3 folds",
        "direction_count": DIRECTION_COUNT,
        "seed_reports": seed_reports,
        "ensemble_summary": ensemble_summary,
        "submission_gate_passed": accepted,
    }
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not accepted:
        print("SUBMISSION_GATE_FAILED", flush=True)


if __name__ == "__main__":
    main()
