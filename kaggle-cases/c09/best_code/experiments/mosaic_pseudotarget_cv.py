from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "de_train.parquet"
REPORT_PATH = ROOT / "experiments" / "mosaic_pseudotarget_cv_results.json"

SOURCE_CELLS = ["NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"]
SUPPORT_SIZE = 17
EPISODES_PER_CONTEXT = 20
FEATURE_COMPONENTS = 12
RESIDUAL_COMPONENTS = 8
MODULE_COUNT = 4
RIDGE_ALPHA = 10.0
LOWER_QUANTILE = 0.20
RANDOM_SEED = 20261014


@dataclass
class ContextData:
    name: str
    values: np.ndarray
    features: np.ndarray
    anchor: np.ndarray
    residual: np.ndarray
    modules: list[np.ndarray]
    module_features: np.ndarray
    source_disagreement: np.ndarray


def load_source_tensor() -> tuple[list[str], list[str], np.ndarray]:
    frame = pd.read_parquet(DATA_PATH)
    metadata = {"cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"}
    genes = [column for column in frame.columns if column not in metadata]
    indexed = {
        cell: frame[frame["cell_type"].eq(cell)].set_index("sm_name")[genes]
        for cell in SOURCE_CELLS
    }
    drugs = sorted(set.intersection(*[set(table.index) for table in indexed.values()]))
    values = np.stack(
        [indexed[cell].loc[drugs].to_numpy(dtype=np.float32) for cell in SOURCE_CELLS],
        axis=1,
    )
    return drugs, genes, values


def pairwise_gene_invariance(values: np.ndarray) -> np.ndarray:
    correlations = []
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            x = values[:, left].astype(np.float64)
            y = values[:, right].astype(np.float64)
            x -= x.mean(axis=0)
            y -= y.mean(axis=0)
            correlations.append(
                np.sum(x * y, axis=0)
                / (np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0)) + 1e-8)
            )
    return np.mean(correlations, axis=0)


def build_modules(invariance: np.ndarray) -> list[np.ndarray]:
    edges = np.quantile(invariance, np.linspace(0.0, 1.0, MODULE_COUNT + 1))
    modules = []
    for position in range(MODULE_COUNT):
        upper = invariance <= edges[position + 1] if position == MODULE_COUNT - 1 else invariance < edges[position + 1]
        modules.append((invariance >= edges[position]) & upper)
    return modules


def module_summary(values: np.ndarray, modules: list[np.ndarray]) -> np.ndarray:
    rows = []
    for mask in modules:
        block = values[..., mask]
        rows.append(np.sqrt(np.mean(block * block, axis=-1)))
    return np.stack(rows, axis=-1)


def prepare_context(values: np.ndarray, held_context: int) -> ContextData:
    source_positions = [position for position in range(values.shape[1]) if position != held_context]
    sources = values[:, source_positions]
    target = values[:, held_context]
    anchor = sources.mean(axis=1)
    residual = target - anchor

    reducer = PCA(
        n_components=FEATURE_COMPONENTS,
        svd_solver="randomized",
        random_state=RANDOM_SEED + held_context,
    )
    reducer.fit(sources.reshape(-1, sources.shape[-1]))
    scores = reducer.transform(sources.reshape(-1, sources.shape[-1])).reshape(
        len(values), len(source_positions), FEATURE_COMPONENTS
    )
    strengths = np.sqrt(np.mean(sources * sources, axis=-1))
    normalized_strengths = strengths / (strengths.mean(axis=1, keepdims=True) + 1e-6)
    disagreement = np.sqrt(np.mean((sources - anchor[:, None]) ** 2, axis=(1, 2)))
    anchor_strength = np.sqrt(np.mean(anchor * anchor, axis=1))
    features = np.concatenate(
        [
            scores.reshape(len(values), -1),
            normalized_strengths,
            anchor_strength[:, None],
            disagreement[:, None],
        ],
        axis=1,
    )
    feature_center = features.mean(axis=0)
    feature_scale = features.std(axis=0) + 1e-6
    features = (features - feature_center) / feature_scale

    invariance = pairwise_gene_invariance(sources)
    modules = build_modules(invariance)
    source_disagreement = module_summary(sources - anchor[:, None], modules).mean(axis=1)
    module_features = np.stack(
        [
            np.asarray(
                [
                    float(invariance[mask].mean()),
                    float(invariance[mask].std()),
                    float(mask.mean()),
                ]
            )
            for mask in modules
        ]
    )
    return ContextData(
        name=SOURCE_CELLS[held_context],
        values=target,
        features=features,
        anchor=anchor,
        residual=residual,
        modules=modules,
        module_features=module_features,
        source_disagreement=source_disagreement,
    )


def fit_transport(
    context: ContextData,
    support: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    component_count = min(RESIDUAL_COMPONENTS, len(support) - 1)
    reducer = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=RANDOM_SEED + int(support[0]),
    )
    support_scores = reducer.fit_transform(context.residual[support])
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(context.features[support], support_scores)
    support_correction = reducer.inverse_transform(model.predict(context.features[support]))
    query_correction = reducer.inverse_transform(model.predict(context.features[query]))
    return support_correction.astype(np.float32), query_correction.astype(np.float32)


def module_mse(error: np.ndarray, modules: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.mean(error[:, mask] ** 2, axis=1) for mask in modules], axis=1)


def nearest_distance(
    features: np.ndarray, support: np.ndarray, query: np.ndarray
) -> np.ndarray:
    difference = features[query, None] - features[support][None]
    return np.sqrt(np.min(np.mean(difference * difference, axis=-1), axis=1))


def episode(
    context: ContextData,
    support: np.ndarray,
    query: np.ndarray,
) -> dict[str, np.ndarray]:
    support_correction, query_correction = fit_transport(context, support, query)
    support_base_mse = module_mse(context.residual[support], context.modules)
    support_candidate_mse = module_mse(
        context.residual[support] - support_correction, context.modules
    )
    support_gain = support_base_mse - support_candidate_mse
    support_gate = support_gain.mean(axis=0) > 0.0

    query_base_mse = module_mse(context.residual[query], context.modules)
    query_candidate_mse = module_mse(
        context.residual[query] - query_correction, context.modules
    )
    gain = query_base_mse - query_candidate_mse
    correction_size = module_summary(query_correction, context.modules)
    anchor_size = module_summary(context.anchor[query], context.modules)
    distance = nearest_distance(context.features, support, query)

    samples = []
    for query_position in range(len(query)):
        for module_position in range(MODULE_COUNT):
            samples.append(
                np.concatenate(
                    [
                        np.asarray(
                            [
                                context.source_disagreement[query[query_position], module_position],
                                correction_size[query_position, module_position],
                                anchor_size[query_position, module_position],
                                distance[query_position],
                                support_gain[:, module_position].mean(),
                                support_gain[:, module_position].std(),
                                support_base_mse[:, module_position].mean(),
                            ]
                        ),
                        context.module_features[module_position],
                    ]
                )
            )
    return {
        "features": np.asarray(samples, dtype=np.float32),
        "gain": gain.reshape(-1).astype(np.float32),
        "base_mse": query_base_mse.astype(np.float32),
        "candidate_mse": query_candidate_mse.astype(np.float32),
        "support_gate": np.broadcast_to(support_gate, query_base_mse.shape),
    }


def context_episodes(context: ContextData, context_position: int) -> list[dict[str, np.ndarray]]:
    outputs = []
    for episode_position in range(EPISODES_PER_CONTEXT):
        rng = np.random.default_rng(
            RANDOM_SEED + 1000 * context_position + episode_position
        )
        order = rng.permutation(len(context.values))
        support = np.sort(order[:SUPPORT_SIZE])
        query = np.sort(order[SUPPORT_SIZE:])
        outputs.append(episode(context, support, query))
    return outputs


def row_rmse_from_modules(module_mse_values: np.ndarray, modules: list[np.ndarray]) -> np.ndarray:
    weights = np.asarray([mask.sum() for mask in modules], dtype=np.float64)
    weights /= weights.sum()
    return np.sqrt(np.sum(module_mse_values * weights[None], axis=1))


def evaluate_gate(
    episodes: list[dict[str, np.ndarray]],
    model: ExtraTreesRegressor,
    modules: list[np.ndarray],
) -> dict[str, float | int]:
    deltas = {name: [] for name in ("ungated", "support", "mean", "lower", "oracle")}
    harms = {name: [] for name in deltas}
    selected = {name: [] for name in ("support", "mean", "lower", "oracle")}
    for item in episodes:
        tree_predictions = np.stack(
            [tree.predict(item["features"]) for tree in model.estimators_], axis=1
        )
        predicted_mean = tree_predictions.mean(axis=1).reshape(-1, MODULE_COUNT)
        predicted_lower = np.quantile(
            tree_predictions, LOWER_QUANTILE, axis=1
        ).reshape(-1, MODULE_COUNT)
        gates = {
            "ungated": np.ones_like(item["base_mse"], dtype=bool),
            "support": item["support_gate"],
            "mean": predicted_mean > 0.0,
            "lower": predicted_lower > 0.0,
            "oracle": item["candidate_mse"] < item["base_mse"],
        }
        base_rmse = row_rmse_from_modules(item["base_mse"], modules)
        for name, gate in gates.items():
            chosen_mse = np.where(gate, item["candidate_mse"], item["base_mse"])
            model_rmse = row_rmse_from_modules(chosen_mse, modules)
            delta = base_rmse - model_rmse
            deltas[name].extend(delta.tolist())
            harms[name].extend((delta < 0.0).tolist())
            if name in selected:
                selected[name].extend(gate.mean(axis=1).tolist())
    result = {}
    for name in deltas:
        values = np.asarray(deltas[name])
        result[f"{name}_delta"] = float(values.mean())
        result[f"{name}_wins"] = int((values > 0.0).sum())
        result[f"{name}_harm_rate"] = float(np.mean(harms[name]))
        if name in selected:
            result[f"{name}_selected_fraction"] = float(np.mean(selected[name]))
    return result


def main() -> None:
    drugs, genes, values = load_source_tensor()
    contexts = [prepare_context(values, position) for position in range(len(SOURCE_CELLS))]
    all_episodes = [
        context_episodes(context, position) for position, context in enumerate(contexts)
    ]
    fold_reports = []
    for held_position, context in enumerate(contexts):
        train_features = np.concatenate(
            [item["features"] for position, episodes in enumerate(all_episodes) if position != held_position for item in episodes]
        )
        train_gain = np.concatenate(
            [item["gain"] for position, episodes in enumerate(all_episodes) if position != held_position for item in episodes]
        )
        model = ExtraTreesRegressor(
            n_estimators=192,
            min_samples_leaf=40,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_SEED + held_position,
        )
        model.fit(train_features, train_gain)
        metrics = evaluate_gate(all_episodes[held_position], model, context.modules)
        fold_reports.append({"held_context": context.name, **metrics})
        print(json.dumps(fold_reports[-1]), flush=True)

    aggregate = {}
    metric_names = [key for key in fold_reports[0] if key != "held_context"]
    for metric in metric_names:
        values_for_metric = [float(report[metric]) for report in fold_reports]
        aggregate[metric] = float(np.mean(values_for_metric))
    lower_deltas = [report["lower_delta"] for report in fold_reports]
    aggregate["contexts_improved_by_lower_gate"] = int(
        sum(delta > 0.0 for delta in lower_deltas)
    )
    passed = bool(
        aggregate["lower_delta"] > 0.0
        and aggregate["contexts_improved_by_lower_gate"] >= 3
        and aggregate["lower_harm_rate"] < aggregate["ungated_harm_rate"]
        and aggregate["lower_delta"] > aggregate["support_delta"]
    )
    report = {
        "method": "MOSAIC pseudo-target gain prediction feasibility",
        "data": {
            "drugs": len(drugs),
            "genes": len(genes),
            "contexts": SOURCE_CELLS,
            "support_size": SUPPORT_SIZE,
            "episodes_per_context": EPISODES_PER_CONTEXT,
        },
        "protocol": "leave-one-source-context-out meta-validation",
        "gain_target": "module-wise baseline-relative MSE improvement",
        "folds": fold_reports,
        "aggregate": aggregate,
        "feasibility_gate_passed": passed,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
