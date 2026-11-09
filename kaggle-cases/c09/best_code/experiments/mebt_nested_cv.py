from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

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
REPORT_PATH = ROOT / "experiments" / "mebt_nested_cv_results.json"
PREDICTION_PATH = ROOT / "experiments" / "mebt_nested_cv_predictions.npz"

MAX_COMPONENTS = 32
COMPONENT_GRID = (8, 16, 32)
PRIOR_PRECISION_GRID = (1.0, 10.0, 100.0)
BLEND_GRID = (0.0, 0.025, 0.05, 0.1, 0.2)
BASE_SEEDS = (17, 29, 43)
SOURCE_RIDGE = 1.0


@dataclass(frozen=True)
class Config:
    components: int
    prior_precision: float
    blend: float


def stable_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def fit_source_space(data) -> tuple[PCA, np.ndarray, np.ndarray]:
    source_indices = [data.cell_index[cell] for cell in SOURCE_CELLS]
    profiles = []
    for drug_position in range(len(data.drugs)):
        for cell in source_indices:
            if data.observed[drug_position, cell]:
                profiles.append(data.values[drug_position, cell])
    pca = PCA(n_components=MAX_COMPONENTS, svd_solver="randomized", random_state=OUTER_SEED)
    pca.fit(np.stack(profiles))
    scale = np.sqrt(pca.explained_variance_).astype(np.float64) + 1e-6
    scores = np.full(
        (len(data.drugs), len(SOURCE_CELLS), MAX_COMPONENTS),
        np.nan,
        dtype=np.float64,
    )
    for drug_position in range(len(data.drugs)):
        for source_position, cell in enumerate(source_indices):
            if data.observed[drug_position, cell]:
                scores[drug_position, source_position] = (
                    pca.transform(data.values[drug_position, cell][None])[0] / scale
                )
    return pca, scale, scores


def subset_features(scores: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            scores.mean(axis=0),
            scores.std(axis=0),
            scores.max(axis=0) - scores.min(axis=0),
            np.ones(scores.shape[1], dtype=np.float64),
        ],
        axis=1,
    )


def source_prior(
    data,
    source_scores: np.ndarray,
    allowed_drugs: set[str],
    components: int,
) -> np.ndarray:
    feature_parts = []
    target_parts = []
    for drug in sorted(allowed_drugs):
        drug_position = data.drug_index[drug]
        available = np.flatnonzero(np.isfinite(source_scores[drug_position, :, 0]))
        if len(available) < 4:
            continue
        for held_position in available:
            context = np.delete(available, np.where(available == held_position))
            if len(context) < 3:
                continue
            feature_parts.append(
                subset_features(source_scores[drug_position, context, :components])
            )
            target_parts.append(source_scores[drug_position, held_position, :components])
    features = np.stack(feature_parts)
    targets = np.stack(target_parts)
    prior = np.empty((components, features.shape[-1]), dtype=np.float64)
    identity = SOURCE_RIDGE * np.eye(features.shape[-1], dtype=np.float64)
    for component in range(components):
        x = features[:, component]
        y = targets[:, component]
        prior[component] = stable_solve(x.T @ x + identity, x.T @ y)
    return prior


def target_examples(
    data,
    source_scores: np.ndarray,
    target: str,
    drugs: np.ndarray,
    pca: PCA,
    scale: np.ndarray,
    components: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_parts = []
    target_parts = []
    target_index = data.cell_index[target]
    for drug in drugs:
        drug_position = data.drug_index[drug]
        available = np.flatnonzero(np.isfinite(source_scores[drug_position, :, 0]))
        if len(available) < 3:
            continue
        target_score = pca.transform(data.values[drug_position, target_index][None])[0]
        target_score = target_score[:components] / scale[:components]
        for subset in combinations(available.tolist(), 3):
            feature_parts.append(
                subset_features(source_scores[drug_position, list(subset), :components])
            )
            target_parts.append(target_score)
    return np.stack(feature_parts), np.stack(target_parts)


def fit_target_adapter(
    features: np.ndarray,
    targets: np.ndarray,
    prior: np.ndarray,
    prior_precision: float,
) -> np.ndarray:
    result = np.empty_like(prior)
    identity = prior_precision * np.eye(features.shape[-1], dtype=np.float64)
    for component in range(features.shape[1]):
        x = features[:, component]
        y = targets[:, component]
        result[component] = stable_solve(
            x.T @ x + identity,
            x.T @ y + prior_precision * prior[component],
        )
    return result


def predict_target(
    data,
    source_scores: np.ndarray,
    drugs: np.ndarray,
    adapter: np.ndarray,
    pca: PCA,
    scale: np.ndarray,
    components: int,
) -> np.ndarray:
    predictions = []
    for drug in drugs:
        drug_position = data.drug_index[drug]
        available = np.flatnonzero(np.isfinite(source_scores[drug_position, :, 0]))
        subset_predictions = []
        for subset in combinations(available.tolist(), 3):
            features = subset_features(
                source_scores[drug_position, list(subset), :components]
            )
            subset_predictions.append(np.sum(features * adapter, axis=1))
        latent = np.mean(subset_predictions, axis=0) * scale[:components]
        full_latent = np.zeros((1, MAX_COMPONENTS), dtype=np.float64)
        full_latent[0, :components] = latent
        predictions.append(pca.inverse_transform(full_latent)[0])
    return np.asarray(predictions, dtype=np.float32)


def mebt_prediction(
    data,
    pca: PCA,
    scale: np.ndarray,
    source_scores: np.ndarray,
    target_train: np.ndarray,
    valid_drugs: np.ndarray,
    source_training_drugs: set[str],
    config: Config,
) -> np.ndarray:
    prior = source_prior(
        data,
        source_scores,
        source_training_drugs,
        config.components,
    )
    outputs = []
    for target in TARGET_CELLS:
        features, targets = target_examples(
            data,
            source_scores,
            target,
            target_train,
            pca,
            scale,
            config.components,
        )
        adapter = fit_target_adapter(
            features,
            targets,
            prior,
            config.prior_precision,
        )
        outputs.append(
            predict_target(
                data,
                source_scores,
                valid_drugs,
                adapter,
                pca,
                scale,
                config.components,
            )
        )
    return np.stack(outputs)


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
        if positive_sum > 0
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


def run_seed(data, pca, scale, source_scores, base_seed: int) -> tuple[dict, dict]:
    configs = [Config(*values) for values in product(COMPONENT_GRID, PRIOR_PRECISION_GRID, BLEND_GRID)]
    outer = KFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    stored = {}
    fold_records = []
    all_drugs = set(data.drugs)

    for outer_fold, (outer_train_pos, outer_valid_pos) in enumerate(outer.split(data.complete)):
        outer_train = data.complete[outer_train_pos]
        outer_valid = data.complete[outer_valid_pos]
        inner = KFold(n_splits=3, shuffle=True, random_state=OUTER_SEED + outer_fold)
        config_scores = np.zeros(len(configs), dtype=np.float64)

        for inner_fold, (inner_train_pos, inner_valid_pos) in enumerate(inner.split(outer_train)):
            inner_train = outer_train[inner_train_pos]
            inner_valid = outer_train[inner_valid_pos]
            base = strong_base_prediction(
                data,
                inner_train,
                inner_valid,
                base_seed + 100000 * outer_fold + 10000 * inner_fold,
            )
            truth, _ = truths_and_anchors(data, inner_valid)
            base_errors = mr_rmse(
                base.reshape(-1, base.shape[-1]), truth.reshape(-1, truth.shape[-1])
            ).reshape(len(TARGET_CELLS), -1)
            source_training = all_drugs.difference(outer_valid.tolist()).difference(inner_valid.tolist())
            raw_cache = {}
            for config_position, config in enumerate(configs):
                key = (config.components, config.prior_precision)
                if key not in raw_cache:
                    raw_cache[key] = mebt_prediction(
                        data,
                        pca,
                        scale,
                        source_scores,
                        inner_train,
                        inner_valid,
                        source_training,
                        config,
                    )
                prediction = base + config.blend * (raw_cache[key] - base)
                errors = mr_rmse(
                    prediction.reshape(-1, prediction.shape[-1]),
                    truth.reshape(-1, truth.shape[-1]),
                ).reshape(len(TARGET_CELLS), -1)
                regression = np.maximum(errors.mean(axis=1) - base_errors.mean(axis=1), 0.0)
                config_scores[config_position] += float(errors.mean() + 4.0 * regression.sum())

        best = configs[int(np.argmin(config_scores))]
        base = strong_base_prediction(
            data,
            outer_train,
            outer_valid,
            base_seed + 900000 + 10000 * outer_fold,
        )
        truth, _ = truths_and_anchors(data, outer_valid)
        source_training = all_drugs.difference(outer_valid.tolist())
        raw = mebt_prediction(
            data,
            pca,
            scale,
            source_scores,
            outer_train,
            outer_valid,
            source_training,
            best,
        )
        prediction = base + best.blend * (raw - base)
        for position, drug in enumerate(outer_valid):
            stored[drug] = {
                "base": base[:, position],
                "truth": truth[:, position],
                "raw": raw[:, position],
                "prediction": prediction[:, position],
            }
        fold_summary = summarize(base, prediction, truth, outer_valid)
        fold_records.append(
            {
                "fold": outer_fold,
                "train": outer_train.tolist(),
                "valid": outer_valid.tolist(),
                "selected": asdict(best),
                "combined_delta": fold_summary["combined_delta"],
            }
        )
        print(
            f"seed={base_seed} outer={outer_fold} selected={best} "
            f"delta={fold_summary['combined_delta']:.6f}",
            flush=True,
        )

    arrays = {
        name: np.stack([stored[drug][name] for drug in data.complete], axis=1)
        for name in ("base", "truth", "raw", "prediction")
    }
    summary = summarize(arrays["base"], arrays["prediction"], arrays["truth"], data.complete)
    return {"seed": base_seed, "folds": fold_records, "summary": summary}, arrays


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = load_data()
    pca, scale, source_scores = fit_source_space(data)
    seed_reports = []
    seed_arrays = []
    for position, seed in enumerate(BASE_SEEDS):
        report, arrays = run_seed(data, pca, scale, source_scores, seed)
        seed_reports.append(report)
        seed_arrays.append(arrays)
        if position == 0:
            targets = report["summary"]["targets"]
            if (
                report["summary"]["combined_delta"] <= 0.0
                or any(targets[target]["delta"] <= 0.0 for target in TARGET_CELLS)
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
            and all(
                report["summary"]["combined_delta"] > 0.0
                for report in seed_reports
            )
        )
        np.savez_compressed(
            PREDICTION_PATH,
            compounds=data.complete,
            base=ensemble_base,
            truth=truth,
            prediction=ensemble_prediction,
            raw=np.mean([arrays["raw"] for arrays in seed_arrays], axis=0),
            seeds=np.asarray(BASE_SEEDS),
        )

    result = {
        "method": "Masked-cell Empirical-Bayes Transport",
        "source_pretraining": "leave-one-source-cell reconstruction with held-out compounds excluded",
        "target_adaptation": "componentwise ridge adapter with source-operator prior",
        "validation": "5x3 nested compound-held-out CV against a strong neural base",
        "configs": {
            "components": list(COMPONENT_GRID),
            "prior_precision": list(PRIOR_PRECISION_GRID),
            "blend": list(BLEND_GRID),
        },
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
