from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


ROOT = Path("/workspace/c09-single-cell")
DATA_PATH = ROOT / "data" / "de_train.parquet"
OUTPUT_PATH = ROOT / "experiments" / "dnpp_residual_cv_results.json"
SEED = 17
SOURCE_CELLS = [
    "NK cells",
    "T cells CD4+",
    "T cells CD8+",
    "T regulatory cells",
]
TARGET_CELLS = ["B cells", "Myeloid cells"]
CELLS = SOURCE_CELLS + TARGET_CELLS


@dataclass(frozen=True)
class Config:
    similarity: str
    neighbors: int
    power: float
    shrinkage: float
    prediction: str
    strength_scale: bool


def mr_rmse(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((prediction - truth) ** 2, axis=1))


def bootstrap_interval(values: np.ndarray, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(20000)]
    )
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def normalize_rows(array: np.ndarray, center: bool) -> np.ndarray:
    result = array.astype(np.float64, copy=True)
    if center:
        result -= result.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.sum(result * result, axis=1, keepdims=True)) + 1e-12
    return result / norm


def main() -> None:
    frame = pd.read_parquet(DATA_PATH)
    metadata = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]
    genes = [column for column in frame.columns if column not in metadata]
    drugs = sorted(frame["sm_name"].unique())
    drug_index = {drug: idx for idx, drug in enumerate(drugs)}
    cell_index = {cell: idx for idx, cell in enumerate(CELLS)}
    source_indices = [cell_index[cell] for cell in SOURCE_CELLS]
    target_indices = [cell_index[cell] for cell in TARGET_CELLS]

    values = np.full(
        (len(drugs), len(CELLS), len(genes)), np.nan, dtype=np.float32
    )
    for _, row in frame.iterrows():
        cell = row["cell_type"]
        if cell in cell_index:
            values[drug_index[row["sm_name"]], cell_index[cell]] = row[genes].to_numpy(
                dtype=np.float32
            )
    observed = np.isfinite(values[..., 0])

    anchors = []
    strengths = []
    for drug in range(len(drugs)):
        available = observed[drug, source_indices]
        source = values[drug, np.asarray(source_indices)[available]]
        mean = source.mean(axis=0)
        median = np.median(source, axis=0)
        strength = float(np.sqrt(np.mean(mean * mean)))
        nk_idx = cell_index["NK cells"]
        nk = values[drug, nk_idx] if observed[drug, nk_idx] else mean
        nk_strength = float(np.sqrt(np.mean(nk * nk)))
        if strength < 1.987:
            anchor = median
        elif nk_strength / max(strength, 1e-8) >= 1.5:
            anchor = 0.75 * nk + 0.25 * mean
        else:
            anchor = mean
        anchors.append(anchor)
        strengths.append(float(np.sqrt(np.mean(anchor * anchor))))
    anchors = np.stack(anchors)
    strengths = np.asarray(strengths)

    similarities: dict[str, np.ndarray] = {}
    for name, center in [("pearson", True), ("cosine", False)]:
        total = np.zeros((len(drugs), len(drugs)), dtype=np.float64)
        count = np.zeros_like(total)
        for cell in source_indices:
            idx = np.flatnonzero(observed[:, cell])
            normalized = normalize_rows(values[idx, cell], center=center)
            block = normalized @ normalized.T
            total[np.ix_(idx, idx)] += block
            count[np.ix_(idx, idx)] += 1.0
        similarities[name] = total / np.maximum(count, 1.0)
        np.fill_diagonal(similarities[name], -np.inf)

    complete = [
        drug
        for drug in drugs
        if all(observed[drug_index[drug], target] for target in target_indices)
    ]
    complete_array = np.asarray(complete)
    outer = KFold(n_splits=5, shuffle=True, random_state=SEED)
    configs = [
        Config(similarity, neighbors, power, shrinkage, prediction, strength_scale)
        for similarity, neighbors, power, shrinkage, prediction, strength_scale in product(
            ["pearson", "cosine"],
            [1, 3, 5, 10],
            [1.0, 2.0],
            [0.25, 0.5, 0.75, 1.0],
            ["absolute", "residual"],
            [False, True],
        )
    ]

    def predict_one(
        drug: int,
        target: int,
        masked: set[str],
        config: Config,
    ) -> np.ndarray:
        eligible = np.asarray(
            [
                idx
                for idx, name in enumerate(drugs)
                if observed[idx, target] and name not in masked and idx != drug
            ]
        )
        score = similarities[config.similarity][drug, eligible]
        positive = score > 0
        if positive.any():
            eligible = eligible[positive]
            score = score[positive]
        order = np.argsort(score)[::-1][: config.neighbors]
        eligible = eligible[order]
        score = score[order]
        shifted = np.maximum(score - min(float(score.min()), 0.0), 1e-8)
        weight = shifted**config.power
        weight /= weight.sum()

        if config.prediction == "absolute":
            neighbor_prediction = np.sum(
                weight[:, None] * values[eligible, target], axis=0
            )
            correction = neighbor_prediction - anchors[drug]
        else:
            residual = values[eligible, target] - anchors[eligible]
            if config.strength_scale:
                ratio = np.clip(
                    strengths[drug] / np.maximum(strengths[eligible], 1e-6),
                    0.5,
                    2.0,
                )
                residual = residual * ratio[:, None]
            correction = np.sum(weight[:, None] * residual, axis=0)
        return anchors[drug] + config.shrinkage * correction

    predictions: dict[tuple[str, str], np.ndarray] = {}
    fold_records = []
    for outer_fold, (train_pos, valid_pos) in enumerate(outer.split(complete_array)):
        outer_train = complete_array[train_pos]
        outer_valid = complete_array[valid_pos]
        inner = KFold(n_splits=3, shuffle=True, random_state=SEED + outer_fold)
        inner_splits = [
            set(outer_train[valid_inner].tolist())
            for _, valid_inner in inner.split(outer_train)
        ]
        selected: dict[str, Config] = {}
        selected_score: dict[str, float] = {}
        for target_name, target in zip(TARGET_CELLS, target_indices):
            base_cache: dict[tuple[object, ...], np.ndarray] = {}
            config_scores = []
            for config in configs:
                errors = []
                for inner_valid in inner_splits:
                    masked = set(outer_valid.tolist()) | inner_valid
                    rows = []
                    for drug in inner_valid:
                        key = (
                            tuple(sorted(inner_valid)),
                            drug,
                            config.similarity,
                            config.neighbors,
                            config.power,
                            config.prediction,
                            config.strength_scale,
                        )
                        if key not in base_cache:
                            endpoint_config = Config(
                                config.similarity,
                                config.neighbors,
                                config.power,
                                1.0,
                                config.prediction,
                                config.strength_scale,
                            )
                            base_cache[key] = predict_one(
                                drug_index[drug], target, masked, endpoint_config
                            )
                        d = drug_index[drug]
                        rows.append(
                            anchors[d]
                            + config.shrinkage * (base_cache[key] - anchors[d])
                        )
                    prediction = np.stack(rows)
                    truth = np.stack(
                        [values[drug_index[drug], target] for drug in inner_valid]
                    )
                    errors.extend(mr_rmse(prediction, truth).tolist())
                config_scores.append(float(np.mean(errors)))
            position = int(np.argmin(config_scores))
            selected[target_name] = configs[position]
            selected_score[target_name] = config_scores[position]

        masked_outer = set(outer_valid.tolist())
        for drug in outer_valid:
            d = drug_index[drug]
            for target_name, target in zip(TARGET_CELLS, target_indices):
                predictions[(drug, target_name)] = predict_one(
                    d, target, masked_outer, selected[target_name]
                ).astype(np.float32)
        fold_records.append(
            {
                "fold": outer_fold,
                "train": outer_train.tolist(),
                "valid": outer_valid.tolist(),
                "selected": {
                    target: config.__dict__ for target, config in selected.items()
                },
                "inner_score": selected_score,
            }
        )
        print(
            f"fold={outer_fold} selected={selected} inner={selected_score}",
            flush=True,
        )

    report: dict[str, object] = {
        "method": "strength-normalized residual DNPP",
        "complete_compounds": complete,
        "folds": fold_records,
        "targets": {},
    }
    combined_delta = []
    for target_name, target in zip(TARGET_CELLS, target_indices):
        truth = np.stack([values[drug_index[drug], target] for drug in complete])
        anchor_prediction = np.stack([anchors[drug_index[drug]] for drug in complete])
        model_prediction = np.stack(
            [predictions[(drug, target_name)] for drug in complete]
        )
        anchor_error = mr_rmse(anchor_prediction, truth)
        model_error = mr_rmse(model_prediction, truth)
        delta = anchor_error - model_error
        combined_delta.extend(delta.tolist())
        report["targets"][target_name] = {
            "anchor": float(anchor_error.mean()),
            "model": float(model_error.mean()),
            "delta": float(delta.mean()),
            "wins": int((delta > 0).sum()),
            "ci95": bootstrap_interval(delta),
            "per_compound": {
                drug: float(value) for drug, value in zip(complete, delta)
            },
        }
    combined = np.asarray(combined_delta)
    combined_ci = bootstrap_interval(combined)
    report["combined"] = {
        "delta": float(combined.mean()),
        "wins": int((combined > 0).sum()),
        "ci95": combined_ci,
        "accept": bool(combined.mean() > 0 and combined_ci[0] > 0),
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
