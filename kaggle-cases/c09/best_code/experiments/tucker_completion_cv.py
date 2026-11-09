from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


ROOT = Path("/workspace/c09-single-cell")
DATA_PATH = ROOT / "data" / "de_train.parquet"
OUTPUT_PATH = ROOT / "experiments" / "tucker_completion_cv_results.json"
SEED = 17
SOURCE_CELLS = [
    "NK cells",
    "T cells CD4+",
    "T cells CD8+",
    "T regulatory cells",
]
TARGET_CELLS = ["B cells", "Myeloid cells"]
CELLS = SOURCE_CELLS + TARGET_CELLS
PCA_COMPONENTS = 96


@dataclass(frozen=True)
class Config:
    rank: int
    ridge: float
    shrinkage: float


def mr_rmse(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((prediction - truth) ** 2, axis=1))


def bootstrap_interval(values: np.ndarray, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(20000)]
    )
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def stable_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def fit_tucker_als(
    latent: np.ndarray,
    observed: np.ndarray,
    source_indices: list[int],
    rank: int,
    ridge: float,
    iterations: int = 30,
) -> np.ndarray:
    n_drugs, n_cells, n_latent = latent.shape
    source_matrix = latent[:, source_indices, :].reshape(n_drugs, -1)
    source_center = source_matrix.mean(axis=0, keepdims=True)
    source_scale = source_matrix.std(axis=0, keepdims=True) + 1e-6
    normalized_source = (source_matrix - source_center) / source_scale
    u, s, _ = np.linalg.svd(normalized_source, full_matrices=False)
    factors = u[:, :rank] * s[:rank]
    factor_scale = factors.std(axis=0, keepdims=True) + 1e-6
    factors /= factor_scale

    bias = np.zeros((n_cells, n_latent), dtype=np.float64)
    weights = np.zeros((n_cells, rank, n_latent), dtype=np.float64)
    eye_rank = np.eye(rank, dtype=np.float64)

    for _ in range(iterations):
        for cell in range(n_cells):
            idx = np.flatnonzero(observed[:, cell])
            x = factors[idx]
            y = latent[idx, cell]
            x_mean = x.mean(axis=0, keepdims=True)
            y_mean = y.mean(axis=0, keepdims=True)
            xc = x - x_mean
            yc = y - y_mean
            weights[cell] = stable_solve(
                xc.T @ xc + ridge * eye_rank,
                xc.T @ yc,
            )
            bias[cell] = (y_mean - x_mean @ weights[cell]).ravel()

        for drug in range(n_drugs):
            cell_idx = np.flatnonzero(observed[drug])
            lhs = ridge * eye_rank
            rhs = np.zeros(rank, dtype=np.float64)
            for cell in cell_idx:
                w = weights[cell]
                residual = latent[drug, cell] - bias[cell]
                lhs += w @ w.T
                rhs += w @ residual
            factors[drug] = stable_solve(lhs, rhs)

    return bias[None, :, :] + np.einsum("dr,crk->dck", factors, weights)


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
    def anchor_for_drug(drug_idx: int) -> np.ndarray:
        available = observed[drug_idx, source_indices]
        source = values[drug_idx, np.asarray(source_indices)[available]]
        mean = source.mean(axis=0)
        median = np.median(source, axis=0)
        strength = float(np.sqrt(np.mean(mean * mean)))
        nk_idx = cell_index["NK cells"]
        nk = values[drug_idx, nk_idx] if observed[drug_idx, nk_idx] else mean
        nk_strength = float(np.sqrt(np.mean(nk * nk)))
        if strength < 1.987:
            return median
        if nk_strength / max(strength, 1e-8) >= 1.5:
            return 0.75 * nk + 0.25 * mean
        return mean

    anchors = np.stack([anchor_for_drug(idx) for idx in range(len(drugs))])
    source_residual = np.concatenate(
        [
            values[observed[:, cell], cell] - anchors[observed[:, cell]]
            for cell in source_indices
        ],
        axis=0,
    )
    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="randomized",
        random_state=SEED,
    ).fit(source_residual)
    latent_scale = np.sqrt(pca.explained_variance_).astype(np.float64) + 1e-6

    latent = np.zeros((len(drugs), len(CELLS), PCA_COMPONENTS), dtype=np.float64)
    for drug in range(len(drugs)):
        for cell in range(len(CELLS)):
            if observed[drug, cell]:
                residual = values[drug, cell] - anchors[drug]
                latent[drug, cell] = pca.transform(residual[None])[0] / latent_scale

    complete = [
        drug
        for drug in drugs
        if all(observed[drug_index[drug], target] for target in target_indices)
    ]
    outer = KFold(n_splits=5, shuffle=True, random_state=SEED)
    configs = [
        Config(rank, ridge, shrinkage)
        for rank, ridge, shrinkage in product(
            [1, 2, 3, 4, 6, 8, 12],
            [0.1, 1.0, 10.0, 100.0],
            [0.25, 0.5, 0.75, 1.0],
        )
    ]

    predictions: dict[tuple[str, str], np.ndarray] = {}
    fold_records = []
    complete_array = np.asarray(complete)

    def predict_latent(masked_drugs: set[str], config: Config) -> np.ndarray:
        training_mask = observed.copy()
        for drug in masked_drugs:
            training_mask[drug_index[drug], target_indices] = False
        return fit_tucker_als(
            latent,
            training_mask,
            source_indices,
            config.rank,
            config.ridge,
        )

    def decode_rows(
        latent_prediction: np.ndarray,
        row_keys: list[tuple[int, int]],
        shrinkage: float,
    ) -> np.ndarray:
        encoded = np.stack(
            [latent_prediction[drug, cell] for drug, cell in row_keys]
        )
        residual = pca.inverse_transform(encoded * latent_scale)
        row_anchors = np.stack([anchors[drug] for drug, _ in row_keys])
        return row_anchors + shrinkage * residual

    for outer_fold, (train_pos, valid_pos) in enumerate(outer.split(complete_array)):
        outer_train = complete_array[train_pos]
        outer_valid = complete_array[valid_pos]
        inner = KFold(n_splits=3, shuffle=True, random_state=SEED + outer_fold)
        inner_splits = [
            set(outer_train[inner_valid_pos].tolist())
            for _, inner_valid_pos in inner.split(outer_train)
        ]
        latent_cache: dict[tuple[tuple[str, ...], int, float], np.ndarray] = {}
        config_scores = []
        for config in configs:
            scores = []
            for inner_valid in inner_splits:
                masked = set(outer_valid.tolist()) | inner_valid
                cache_key = (tuple(sorted(inner_valid)), config.rank, config.ridge)
                if cache_key not in latent_cache:
                    latent_cache[cache_key] = predict_latent(masked, config)
                latent_prediction = latent_cache[cache_key]
                row_keys = [
                    (drug_index[drug], cell)
                    for drug in inner_valid
                    for cell in target_indices
                ]
                decoded = decode_rows(
                    latent_prediction, row_keys, config.shrinkage
                )
                truth = np.stack([values[drug, cell] for drug, cell in row_keys])
                scores.extend(mr_rmse(decoded, truth).tolist())
            config_scores.append(float(np.mean(scores)))

        best_position = int(np.argmin(config_scores))
        best_config = configs[best_position]
        outer_latent = predict_latent(set(outer_valid.tolist()), best_config)
        outer_keys = [
            (drug_index[drug], cell)
            for drug in outer_valid
            for cell in target_indices
        ]
        outer_decoded = decode_rows(
            outer_latent, outer_keys, best_config.shrinkage
        )
        for position, drug in enumerate(outer_valid):
            for target_position, cell_name in enumerate(TARGET_CELLS):
                predictions[(drug, cell_name)] = outer_decoded[
                    position * len(TARGET_CELLS) + target_position
                ].astype(np.float32)
        fold_records.append(
            {
                "fold": outer_fold,
                "train": outer_train.tolist(),
                "valid": outer_valid.tolist(),
                "selected": best_config.__dict__,
                "inner_score": config_scores[best_position],
            }
        )
        print(f"fold={outer_fold} selected={best_config} inner={config_scores[best_position]:.6f}", flush=True)

    report: dict[str, object] = {
        "method": "source-residual PCA plus masked Tucker ALS",
        "pca_components": PCA_COMPONENTS,
        "complete_compounds": complete,
        "folds": fold_records,
        "targets": {},
    }
    combined_delta = []
    for cell_name, cell in zip(TARGET_CELLS, target_indices):
        truth = np.stack([values[drug_index[drug], cell] for drug in complete])
        anchor_prediction = np.stack([anchors[drug_index[drug]] for drug in complete])
        model_prediction = np.stack([predictions[(drug, cell_name)] for drug in complete])
        anchor_error = mr_rmse(anchor_prediction, truth)
        model_error = mr_rmse(model_prediction, truth)
        delta = anchor_error - model_error
        combined_delta.extend(delta.tolist())
        report["targets"][cell_name] = {
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
