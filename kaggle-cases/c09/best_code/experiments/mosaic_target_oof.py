from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from mosaic_action_cv import (
    LOWER_QUANTILE,
    TAUS,
    fit_model,
    make_episodes,
    normalized_actions,
)
from mosaic_pseudotarget_cv import (
    MODULE_COUNT,
    RANDOM_SEED,
    SOURCE_CELLS,
    build_modules,
    load_source_tensor,
    module_mse,
    module_summary,
    pairwise_gene_invariance,
    prepare_context,
    row_rmse_from_modules,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "de_train.parquet"
OOF_PATH = ROOT / "experiments" / "ramrt_oof_predictions.npz"
REPORT_PATH = ROOT / "experiments" / "mosaic_target_oof_results.json"
PREDICTION_PATH = ROOT / "experiments" / "mosaic_target_oof_predictions.npz"
TARGET_CELLS = ["B cells", "Myeloid cells"]
BOOTSTRAPS = 20000


def build_anchor(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    rows = []
    for drug_position in range(len(values)):
        available = values[drug_position, observed[drug_position]]
        mean = available.mean(axis=0)
        median = np.median(available, axis=0)
        strength = float(np.sqrt(np.mean(mean * mean)))
        nk = values[drug_position, 0] if observed[drug_position, 0] else mean
        nk_strength = float(np.sqrt(np.mean(nk * nk)))
        if strength < 1.987:
            anchor = median
        elif nk_strength / max(strength, 1e-8) >= 1.5:
            anchor = 0.75 * nk + 0.25 * mean
        else:
            anchor = mean
        rows.append(anchor)
    return np.stack(rows).astype(np.float32)


def load_target_geometry(
    compounds: list[str],
    genes: list[str],
    complete_source_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    frame = pd.read_parquet(DATA_PATH)
    tables = {
        cell: frame[frame["cell_type"].eq(cell)].set_index("sm_name")[genes]
        for cell in SOURCE_CELLS
    }
    values = np.full(
        (len(compounds), len(SOURCE_CELLS), len(genes)), np.nan, dtype=np.float32
    )
    observed = np.zeros((len(compounds), len(SOURCE_CELLS)), dtype=bool)
    for drug_position, compound in enumerate(compounds):
        for cell_position, cell in enumerate(SOURCE_CELLS):
            if compound in tables[cell].index:
                values[drug_position, cell_position] = tables[cell].loc[
                    compound
                ].to_numpy(dtype=np.float32)
                observed[drug_position, cell_position] = True
    anchor = build_anchor(values, observed)
    invariance = pairwise_gene_invariance(complete_source_values)
    modules = build_modules(invariance)
    module_features = np.stack(
        [
            np.asarray(
                [
                    float(invariance[mask].mean()),
                    float(invariance[mask].std()),
                    float(mask.mean()),
                ],
                dtype=np.float32,
            )
            for mask in modules
        ]
    )
    source_disagreement = np.zeros((len(compounds), MODULE_COUNT), dtype=np.float32)
    for drug_position in range(len(compounds)):
        deviations = (
            values[drug_position, observed[drug_position]] - anchor[drug_position]
        )
        source_disagreement[drug_position] = module_summary(
            deviations, modules
        ).mean(axis=0)

    reducer = PCA(n_components=16, svd_solver="randomized", random_state=RANDOM_SEED)
    reducer.fit(complete_source_values.reshape(-1, len(genes)))
    anchor_scores = reducer.transform(anchor)
    score_parts = []
    mask_parts = []
    strength_parts = []
    for drug_position in range(len(compounds)):
        scores = []
        strengths = []
        for cell_position in range(len(SOURCE_CELLS)):
            if observed[drug_position, cell_position]:
                profile = values[drug_position, cell_position]
                scores.append(reducer.transform(profile[None])[0])
                strengths.append(float(np.sqrt(np.mean(profile * profile))))
            else:
                scores.append(anchor_scores[drug_position])
                strengths.append(float(np.sqrt(np.mean(anchor[drug_position] ** 2))))
        score_parts.append(np.concatenate(scores))
        mask_parts.append(observed[drug_position].astype(np.float32))
        strength_array = np.asarray(strengths)
        strength_parts.append(strength_array)
    geometry = np.concatenate(
        [
            np.asarray(score_parts),
            np.asarray(mask_parts),
            np.asarray(strength_parts),
        ],
        axis=1,
    )
    return anchor, source_disagreement, modules, module_features, geometry


def nearest_heldout_distance(geometry: np.ndarray) -> np.ndarray:
    distances = np.zeros(len(geometry), dtype=np.float32)
    for position in range(len(geometry)):
        support = np.arange(len(geometry)) != position
        center = geometry[support].mean(axis=0)
        scale = geometry[support].std(axis=0) + 1e-4
        support_scaled = (geometry[support] - center) / scale
        query_scaled = (geometry[position] - center) / scale
        difference = support_scaled - query_scaled
        distances[position] = float(np.sqrt(np.min(np.mean(difference * difference, axis=1))))
    return distances


def target_action_data(
    base: np.ndarray,
    truth: np.ndarray,
    correction: np.ndarray,
    anchor: np.ndarray,
    source_disagreement: np.ndarray,
    modules: list[np.ndarray],
    module_features: np.ndarray,
    geometry: np.ndarray,
) -> dict[str, np.ndarray]:
    residual = truth - base
    base_mse = module_mse(residual, modules)
    actions = normalized_actions(correction, anchor, modules)
    action_mse = np.stack(
        [module_mse(residual - action, modules) for action in actions], axis=2
    )
    correction_sizes = [module_summary(action, modules) for action in actions]
    anchor_size = module_summary(anchor, modules)
    distance = nearest_heldout_distance(geometry)
    features_by_action = []
    all_indices = np.arange(len(base))
    for action_position, (tau, action_size) in enumerate(zip(TAUS, correction_sizes)):
        gain = base_mse - action_mse[:, :, action_position]
        samples = []
        for query_position in range(len(base)):
            support = all_indices != query_position
            for module_position in range(MODULE_COUNT):
                samples.append(
                    np.concatenate(
                        [
                            np.asarray(
                                [
                                    source_disagreement[
                                        query_position, module_position
                                    ],
                                    action_size[query_position, module_position],
                                    anchor_size[query_position, module_position],
                                    action_size[query_position, module_position]
                                    / (
                                        anchor_size[query_position, module_position]
                                        + 1e-8
                                    ),
                                    distance[query_position],
                                    gain[support, module_position].mean(),
                                    gain[support, module_position].std(),
                                    base_mse[support, module_position].mean(),
                                    float(tau),
                                ],
                                dtype=np.float32,
                            ),
                            module_features[module_position],
                        ]
                    )
                )
        features_by_action.append(np.asarray(samples, dtype=np.float32))
    return {
        "features": np.stack(features_by_action, axis=1),
        "base_mse": base_mse,
        "action_mse": action_mse,
        "actions": np.stack(actions, axis=0),
    }


def bootstrap_interval(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(BOOTSTRAPS)]
    )
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def evaluate_target(item, model, base, truth, correction, modules, seed):
    features = item["features"]
    flat = features.reshape(-1, features.shape[-1])
    tree_predictions = np.stack(
        [tree.predict(flat) for tree in model.estimators_], axis=1
    )
    lower = np.quantile(tree_predictions, LOWER_QUANTILE, axis=1).reshape(
        features.shape[:2]
    )
    best_action = np.argmax(lower, axis=1).reshape(item["base_mse"].shape)
    best_lower = np.max(lower, axis=1).reshape(item["base_mse"].shape)
    selected = best_lower > 0.0
    prediction = base.copy()
    for action_position in range(len(TAUS)):
        for module_position, mask in enumerate(modules):
            rows = selected[:, module_position] & (
                best_action[:, module_position] == action_position
            )
            prediction[np.ix_(rows, mask)] += item["actions"][
                action_position
            ][np.ix_(rows, mask)]
    base_rmse = np.sqrt(np.mean((base - truth) ** 2, axis=1))
    model_rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=1))
    fixed_prediction = base + 0.05 * correction
    fixed_rmse = np.sqrt(np.mean((fixed_prediction - truth) ** 2, axis=1))
    delta = base_rmse - model_rmse
    positive = np.maximum(delta, 0.0)
    concentration = float(positive.max() / positive.sum()) if positive.sum() else 1.0
    report = {
        "base_rmse": float(base_rmse.mean()),
        "fixed_alpha_0_05_rmse": float(fixed_rmse.mean()),
        "fixed_alpha_0_05_delta": float((base_rmse - fixed_rmse).mean()),
        "mosaic_rmse": float(model_rmse.mean()),
        "mosaic_delta": float(delta.mean()),
        "mosaic_delta_ci": bootstrap_interval(delta, seed),
        "wins": int((delta > 0.0).sum()),
        "losses": int((delta < 0.0).sum()),
        "selected_fraction": float(selected.mean()),
        "positive_gain_concentration": concentration,
    }
    return prediction, report, delta, selected, best_action, best_lower


def main() -> None:
    _, genes, complete_source_values = load_source_tensor()
    contexts = [prepare_context(complete_source_values, position) for position in range(4)]
    source_episodes = [
        item
        for position, context in enumerate(contexts)
        for item in make_episodes(context, position)
    ]
    model = fit_model(source_episodes, RANDOM_SEED + 100)

    oof = np.load(OOF_PATH, allow_pickle=True)
    compounds = [str(value) for value in oof["compounds"]]
    base = oof["base"].astype(np.float32)
    truth = oof["truth"].astype(np.float32)
    correction = oof["corrections"].mean(axis=0).astype(np.float32)
    anchor, disagreement, modules, module_features, geometry = load_target_geometry(
        compounds, genes, complete_source_values
    )
    predictions = []
    reports = []
    deltas = []
    selections = []
    actions = []
    lowers = []
    for target_position, target_name in enumerate(TARGET_CELLS):
        item = target_action_data(
            base[target_position],
            truth[target_position],
            correction[target_position],
            anchor,
            disagreement,
            modules,
            module_features,
            geometry,
        )
        prediction, report, delta, selected, best_action, best_lower = evaluate_target(
            item,
            model,
            base[target_position],
            truth[target_position],
            correction[target_position],
            modules,
            RANDOM_SEED + target_position,
        )
        reports.append({"target": target_name, **report})
        predictions.append(prediction)
        deltas.append(delta)
        selections.append(selected)
        actions.append(best_action)
        lowers.append(best_lower)
        print(json.dumps(reports[-1]), flush=True)
    combined_delta = np.concatenate(deltas)
    combined = {
        "mosaic_delta": float(combined_delta.mean()),
        "mosaic_delta_ci": bootstrap_interval(combined_delta, RANDOM_SEED + 10),
        "wins": int((combined_delta > 0.0).sum()),
        "losses": int((combined_delta < 0.0).sum()),
    }
    passed = bool(
        all(report["mosaic_delta"] > 0.0 for report in reports)
        and all(report["wins"] >= 10 for report in reports)
        and combined["mosaic_delta_ci"][0] > 0.0
        and all(report["positive_gain_concentration"] <= 0.5 for report in reports)
    )
    result = {
        "method": "MOSAIC source-meta-trained normalized action selection",
        "protocol": "source-only meta-training plus compound-held-out OOF stacking feasibility",
        "lower_quantile": LOWER_QUANTILE,
        "taus": TAUS.tolist(),
        "targets": reports,
        "combined": combined,
        "strict_feasibility_gate_passed": passed,
        "limitations": [
            "OOF corrections are cross-fitted, but support gain features are stacked across outer folds.",
            "A positive result still requires full outer-fold reconstruction and multiple base seeds.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        PREDICTION_PATH,
        compounds=np.asarray(compounds),
        prediction=np.stack(predictions),
        selected=np.stack(selections),
        best_action=np.stack(actions),
        best_lower=np.stack(lowers),
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
