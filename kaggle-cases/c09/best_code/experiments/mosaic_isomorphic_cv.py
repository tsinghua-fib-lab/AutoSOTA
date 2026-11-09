from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from mosaic_action_cv import (
    LOWER_QUANTILE,
    TAUS,
    evaluate,
    fit_model,
    normalized_actions,
)
from mosaic_pseudotarget_cv import (
    ContextData,
    MODULE_COUNT,
    RANDOM_SEED,
    SOURCE_CELLS,
    build_modules,
    load_source_tensor,
    module_mse,
    module_summary,
    pairwise_gene_invariance,
)
from mosaic_target_oof import (
    OOF_PATH,
    TARGET_CELLS,
    evaluate_target,
    load_target_geometry,
    target_action_data,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tabpfn-v2-regressor.ckpt"
REPORT_PATH = ROOT / "experiments" / "mosaic_isomorphic_cv_results.json"
TARGET_COMPONENTS = 8
SOURCE_COMPONENTS = 16
SUPPORT_SIZE = 17
INNER_FOLDS = 4


def heuristic_anchor(sources: np.ndarray, source_names: list[str]) -> np.ndarray:
    mean = sources.mean(axis=1)
    median = np.median(sources, axis=1)
    strength = np.sqrt(np.mean(mean * mean, axis=1))
    if "NK cells" in source_names:
        nk = sources[:, source_names.index("NK cells")]
    else:
        nk = mean
    nk_strength = np.sqrt(np.mean(nk * nk, axis=1))
    anchor = mean.copy()
    weak = strength < 1.987
    nk_dominant = (~weak) & (nk_strength / np.maximum(strength, 1e-8) >= 1.5)
    anchor[weak] = median[weak]
    anchor[nk_dominant] = 0.75 * nk[nk_dominant] + 0.25 * mean[nk_dominant]
    return anchor.astype(np.float32)


def source_features(sources: np.ndarray, anchor: np.ndarray, seed: int) -> np.ndarray:
    reducer = PCA(
        n_components=SOURCE_COMPONENTS,
        svd_solver="randomized",
        random_state=seed,
    )
    reducer.fit(sources.reshape(-1, sources.shape[-1]))
    scores = reducer.transform(sources.reshape(-1, sources.shape[-1])).reshape(
        len(sources), sources.shape[1], SOURCE_COMPONENTS
    )
    strengths = np.sqrt(np.mean(sources * sources, axis=2))
    masks = np.ones_like(strengths)
    return np.concatenate(
        [scores.reshape(len(sources), -1), masks, strengths], axis=1
    ).astype(np.float32)


def prepare_isomorphic_context(values: np.ndarray, held_context: int) -> ContextData:
    source_positions = [position for position in range(4) if position != held_context]
    source_names = [SOURCE_CELLS[position] for position in source_positions]
    sources = values[:, source_positions]
    target = values[:, held_context]
    anchor = heuristic_anchor(sources, source_names)
    residual = target - anchor
    features = source_features(sources, anchor, RANDOM_SEED + held_context)
    invariance = pairwise_gene_invariance(sources)
    modules = build_modules(invariance)
    disagreement = module_summary(sources - anchor[:, None], modules).mean(axis=1)
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
    return ContextData(
        name=SOURCE_CELLS[held_context],
        values=target,
        features=features,
        anchor=anchor,
        residual=residual,
        modules=modules,
        module_features=module_features,
        source_disagreement=disagreement,
    )


def component_prediction(
    train_features: np.ndarray,
    train_residuals: np.ndarray,
    valid_features: np.ndarray,
    seed: int,
) -> np.ndarray:
    import torch
    from tabpfn import TabPFNRegressor

    component_count = min(TARGET_COMPONENTS, len(train_residuals) - 1)
    reducer = PCA(n_components=component_count, random_state=seed)
    scores = reducer.fit_transform(train_residuals)
    predicted = np.zeros((len(valid_features), component_count), dtype=np.float32)
    for component in range(component_count):
        model = TabPFNRegressor(
            n_estimators=2,
            device="cuda",
            model_path=MODEL_PATH,
            random_state=seed + component,
            n_jobs=1,
            fit_mode="fit_preprocessors",
        )
        model.fit(train_features, scores[:, component])
        predicted[:, component] = model.predict(valid_features)
        del model
    torch.cuda.empty_cache()
    return reducer.inverse_transform(predicted).astype(np.float32)


def nearest_distance(
    features: np.ndarray, support: np.ndarray, query: np.ndarray
) -> np.ndarray:
    center = features[support].mean(axis=0)
    scale = features[support].std(axis=0) + 1e-4
    support_scaled = (features[support] - center) / scale
    query_scaled = (features[query] - center) / scale
    difference = query_scaled[:, None] - support_scaled[None]
    return np.sqrt(np.min(np.mean(difference * difference, axis=2), axis=1))


def precomputed_episode(
    context: ContextData,
    support: np.ndarray,
    query: np.ndarray,
    support_correction: np.ndarray,
    query_correction: np.ndarray,
) -> dict[str, np.ndarray]:
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
        query_size = module_summary(query_action, context.modules)
        samples = []
        for row in range(len(query)):
            for module in range(MODULE_COUNT):
                samples.append(
                    np.concatenate(
                        [
                            np.asarray(
                                [
                                    context.source_disagreement[query[row], module],
                                    query_size[row, module],
                                    query_anchor_size[row, module],
                                    query_size[row, module]
                                    / (query_anchor_size[row, module] + 1e-8),
                                    distance[row],
                                    support_gain[:, module].mean(),
                                    support_gain[:, module].std(),
                                    support_base_mse[:, module].mean(),
                                    float(tau),
                                ],
                                dtype=np.float32,
                            ),
                            context.module_features[module],
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


def isomorphic_episode(
    context: ContextData,
    context_position: int,
    episode_position: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(
        RANDOM_SEED + 1000 * context_position + episode_position
    )
    order = rng.permutation(len(context.values))
    support = np.sort(order[:SUPPORT_SIZE])
    query = np.sort(order[SUPPORT_SIZE:])
    seed = RANDOM_SEED + 10000 * context_position + 100 * episode_position
    query_correction = component_prediction(
        context.features[support],
        context.residual[support],
        context.features[query],
        seed,
    )
    support_correction = np.zeros_like(context.residual[support])
    splitter = KFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    for fold_position, (train_position, valid_position) in enumerate(splitter.split(support)):
        support_correction[valid_position] = component_prediction(
            context.features[support[train_position]],
            context.residual[support[train_position]],
            context.features[support[valid_position]],
            seed + 10 * (fold_position + 1),
        )
    return precomputed_episode(
        context,
        support,
        query,
        support_correction,
        query_correction,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    started = time.perf_counter()
    _, genes, values = load_source_tensor()
    contexts = [prepare_isomorphic_context(values, position) for position in range(4)]
    all_episodes = []
    for context_position, context in enumerate(contexts):
        episodes = []
        for episode_position in range(args.episodes):
            episodes.append(
                isomorphic_episode(context, context_position, episode_position)
            )
            print(
                json.dumps(
                    {
                        "generated_context": context.name,
                        "episode": episode_position + 1,
                        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
                    }
                ),
                flush=True,
            )
        all_episodes.append(episodes)

    source_reports = []
    for held_position, context in enumerate(contexts):
        train = [
            item
            for position, episodes in enumerate(all_episodes)
            if position != held_position
            for item in episodes
        ]
        model = fit_model(train, RANDOM_SEED + held_position)
        source_reports.append(
            {
                "held_context": context.name,
                **evaluate(all_episodes[held_position], model, context.modules),
            }
        )
        print(json.dumps(source_reports[-1]), flush=True)
    source_aggregate = {
        key: float(np.mean([report[key] for report in source_reports]))
        for key in source_reports[0]
        if key != "held_context"
    }
    source_aggregate["contexts_improved"] = int(
        sum(report["delta"] > 0.0 for report in source_reports)
    )
    source_passed = bool(
        source_aggregate["delta"] > 0.0
        and source_aggregate["contexts_improved"] >= 3
        and source_aggregate["harm_rate"] < 0.20
    )

    target_reports = []
    combined_delta = []
    target_passed = False
    if source_passed:
        model = fit_model(
            [item for episodes in all_episodes for item in episodes],
            RANDOM_SEED + 100,
        )
        oof = np.load(OOF_PATH, allow_pickle=True)
        compounds = [str(value) for value in oof["compounds"]]
        base = oof["base"].astype(np.float32)
        truth = oof["truth"].astype(np.float32)
        correction = oof["corrections"].mean(axis=0).astype(np.float32)
        anchor, disagreement, modules, module_features, geometry = load_target_geometry(
            compounds, genes, values
        )
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
            _, report, delta, _, _, _ = evaluate_target(
                item,
                model,
                base[target_position],
                truth[target_position],
                correction[target_position],
                modules,
                RANDOM_SEED + 200 + target_position,
            )
            target_reports.append({"target": target_name, **report})
            combined_delta.append(delta)
            print(json.dumps(target_reports[-1]), flush=True)
        combined = np.concatenate(combined_delta)
        target_passed = bool(
            all(report["mosaic_delta"] > 0.0 for report in target_reports)
            and all(report["wins"] >= 10 for report in target_reports)
            and all(report["positive_gain_concentration"] <= 0.5 for report in target_reports)
        )
    result = {
        "method": "MOSAIC isomorphic TabPFN action selection",
        "protocol": "LOCO source meta-validation with nested support OOF",
        "episodes_per_context": args.episodes,
        "source_folds": source_reports,
        "source_aggregate": source_aggregate,
        "source_gate_passed": source_passed,
        "target_reports": target_reports,
        "target_preliminary_gate_passed": target_passed,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
