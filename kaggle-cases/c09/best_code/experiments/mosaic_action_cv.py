from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from mosaic_pseudotarget_cv import (
    MODULE_COUNT,
    RANDOM_SEED,
    build_modules,
    context_episodes,
    load_source_tensor,
    module_mse,
    module_summary,
    nearest_distance,
    prepare_context,
    row_rmse_from_modules,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "experiments" / "mosaic_action_cv_results.json"
TAUS = np.asarray([0.025, 0.05, 0.10, 0.20, 0.40, 0.80], dtype=np.float32)
LOWER_QUANTILE = 0.05


def normalized_actions(
    correction: np.ndarray,
    anchor: np.ndarray,
    modules: list[np.ndarray],
) -> list[np.ndarray]:
    outputs = []
    correction_size = module_summary(correction, modules)
    anchor_size = module_summary(anchor, modules)
    for tau in TAUS:
        action = np.zeros_like(correction)
        for module_position, mask in enumerate(modules):
            scale = np.minimum(
                1.0,
                tau * anchor_size[:, module_position]
                / (correction_size[:, module_position] + 1e-8),
            )
            action[:, mask] = correction[:, mask] * scale[:, None]
        outputs.append(action)
    return outputs


def action_episode(context, support: np.ndarray, query: np.ndarray) -> dict[str, np.ndarray]:
    from mosaic_pseudotarget_cv import fit_transport

    support_correction, query_correction = fit_transport(context, support, query)
    support_base_mse = module_mse(context.residual[support], context.modules)
    query_base_mse = module_mse(context.residual[query], context.modules)
    distance = nearest_distance(context.features, support, query)
    query_anchor_size = module_summary(context.anchor[query], context.modules)

    support_actions = normalized_actions(
        support_correction, context.anchor[support], context.modules
    )
    query_actions = normalized_actions(
        query_correction, context.anchor[query], context.modules
    )
    action_features = []
    action_gains = []
    action_mse = []
    for tau, support_action, query_action in zip(TAUS, support_actions, query_actions):
        support_candidate_mse = module_mse(
            context.residual[support] - support_action, context.modules
        )
        support_gain = support_base_mse - support_candidate_mse
        query_candidate_mse = module_mse(
            context.residual[query] - query_action, context.modules
        )
        query_gain = query_base_mse - query_candidate_mse
        query_correction_size = module_summary(query_action, context.modules)
        samples = []
        for query_position in range(len(query)):
            for module_position in range(MODULE_COUNT):
                samples.append(
                    np.concatenate(
                        [
                            np.asarray(
                                [
                                    context.source_disagreement[
                                        query[query_position], module_position
                                    ],
                                    query_correction_size[
                                        query_position, module_position
                                    ],
                                    query_anchor_size[query_position, module_position],
                                    query_correction_size[
                                        query_position, module_position
                                    ]
                                    / (
                                        query_anchor_size[
                                            query_position, module_position
                                        ]
                                        + 1e-8
                                    ),
                                    distance[query_position],
                                    support_gain[:, module_position].mean(),
                                    support_gain[:, module_position].std(),
                                    support_base_mse[:, module_position].mean(),
                                    float(tau),
                                ]
                            ),
                            context.module_features[module_position],
                        ]
                    )
                )
        action_features.append(np.asarray(samples, dtype=np.float32))
        action_gains.append(query_gain.astype(np.float32))
        action_mse.append(query_candidate_mse.astype(np.float32))
    return {
        "features": np.stack(action_features, axis=1),
        "gain": np.stack(action_gains, axis=2),
        "base_mse": query_base_mse.astype(np.float32),
        "action_mse": np.stack(action_mse, axis=2),
    }


def make_episodes(context, context_position: int) -> list[dict[str, np.ndarray]]:
    outputs = []
    for episode_position in range(20):
        rng = np.random.default_rng(
            RANDOM_SEED + 1000 * context_position + episode_position
        )
        order = rng.permutation(len(context.values))
        support = np.sort(order[:17])
        query = np.sort(order[17:])
        outputs.append(action_episode(context, support, query))
    return outputs


def fit_model(episodes: list[dict[str, np.ndarray]], seed: int) -> ExtraTreesRegressor:
    features = np.concatenate(
        [item["features"].reshape(-1, item["features"].shape[-1]) for item in episodes]
    )
    gains = np.concatenate([item["gain"].reshape(-1) for item in episodes])
    model = ExtraTreesRegressor(
        n_estimators=192,
        min_samples_leaf=80,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(features, gains)
    return model


def evaluate(episodes, model, modules) -> dict[str, float | int]:
    deltas = []
    harms = []
    selections = []
    tau_values = []
    oracle_deltas = []
    for item in episodes:
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
        chosen = item["base_mse"].copy()
        oracle = item["base_mse"].copy()
        for action_position in range(len(TAUS)):
            use = selected & (best_action == action_position)
            chosen[use] = item["action_mse"][:, :, action_position][use]
        oracle_action = np.argmin(item["action_mse"], axis=2)
        oracle_selected = (
            np.min(item["action_mse"], axis=2) < item["base_mse"]
        )
        for action_position in range(len(TAUS)):
            use = oracle_selected & (oracle_action == action_position)
            oracle[use] = item["action_mse"][:, :, action_position][use]
        base_rmse = row_rmse_from_modules(item["base_mse"], modules)
        chosen_rmse = row_rmse_from_modules(chosen, modules)
        oracle_rmse = row_rmse_from_modules(oracle, modules)
        delta = base_rmse - chosen_rmse
        deltas.extend(delta.tolist())
        harms.extend((delta < 0.0).tolist())
        selections.extend(selected.mean(axis=1).tolist())
        oracle_deltas.extend((base_rmse - oracle_rmse).tolist())
        for row in range(len(selected)):
            for module in range(MODULE_COUNT):
                if selected[row, module]:
                    tau_values.append(float(TAUS[best_action[row, module]]))
    values = np.asarray(deltas)
    return {
        "delta": float(values.mean()),
        "wins": int((values > 0.0).sum()),
        "harm_rate": float(np.mean(harms)),
        "selected_fraction": float(np.mean(selections)),
        "selected_tau_mean": float(np.mean(tau_values)) if tau_values else 0.0,
        "oracle_delta": float(np.mean(oracle_deltas)),
    }


def main() -> None:
    drugs, genes, values = load_source_tensor()
    contexts = [prepare_context(values, position) for position in range(4)]
    all_episodes = [
        make_episodes(context, position) for position, context in enumerate(contexts)
    ]
    reports = []
    for held_position, context in enumerate(contexts):
        train = [
            item
            for position, episodes in enumerate(all_episodes)
            if position != held_position
            for item in episodes
        ]
        model = fit_model(train, RANDOM_SEED + held_position)
        metrics = evaluate(all_episodes[held_position], model, context.modules)
        report = {"held_context": context.name, **metrics}
        reports.append(report)
        print(json.dumps(report), flush=True)
    aggregate = {
        key: float(np.mean([report[key] for report in reports]))
        for key in reports[0]
        if key != "held_context"
    }
    aggregate["contexts_improved"] = int(
        sum(report["delta"] > 0.0 for report in reports)
    )
    passed = bool(
        aggregate["delta"] > 0.0
        and aggregate["contexts_improved"] >= 3
        and aggregate["harm_rate"] < 0.20
    )
    report = {
        "method": "MOSAIC normalized action selection",
        "protocol": "leave-one-source-context-out meta-validation",
        "data": {"drugs": len(drugs), "genes": len(genes)},
        "taus": TAUS.tolist(),
        "folds": reports,
        "aggregate": aggregate,
        "feasibility_gate_passed": passed,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
