from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNRegressor
from torch import nn


ROOT = Path("/workspace/c09-single-cell")
DATA_PATH = ROOT / "data" / "de_train.parquet"
ID_MAP_PATH = ROOT / "data" / "id_map.csv"
BASE_PATH = ROOT / "final_submission" / "convex_diversity_optimal.zip"
MODEL_PATH = ROOT / "models" / "tabpfn-v2-regressor.ckpt"
REPORT_PATH = ROOT / "experiments" / "ramrt_nested_cv_results.json"
OOF_PATH = ROOT / "experiments" / "ramrt_oof_predictions.npz"
FULL_PATH = ROOT / "experiments" / "ramrt_full_corrections.npz"
OUTPUT_PATH = ROOT / "final_submission" / "ramrt_candidate.zip"
AUDIT_PATH = ROOT / "final_submission" / "ramrt_candidate_audit.json"

SOURCE_CELLS = ["NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"]
TARGET_CELLS = ["B cells", "Myeloid cells"]
CELLS = SOURCE_CELLS + TARGET_CELLS
SEEDS = (17, 29, 43)
OUTER_SEED = 17
SOURCE_COMPONENTS = 16
TARGET_COMPONENTS = 8
BOOTSTRAPS = 20000

GATE_FAMILIES = ("constant", "agreement", "distance", "strength", "blend")
LAMBDA0_GRID = (0.0, 0.025, 0.05)
SPAN_GRID = (0.0, 0.025, 0.05)
RHO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def mr_rmse(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((prediction - truth) ** 2, axis=1))


def bootstrap_interval(values: np.ndarray, seed: int = OUTER_SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(BOOTSTRAPS)]
    )
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def read_submission(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {path}, found {names}")
        with archive.open(names[0]) as handle:
            return pd.read_csv(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Model7(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(152, 48)
        widths = [96, 712, 400, 232, 216, output_dim]
        dropouts = [
            0.4903998136177629,
            0.032371643764537134,
            0.11138300987168903,
            0.019885384663655765,
        ]
        layers: list[nn.Module] = []
        for position, dropout in enumerate(dropouts):
            layers.extend(
                [
                    nn.Linear(widths[position], widths[position + 1]),
                    nn.BatchNorm1d(widths[position + 1], momentum=0.01),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(widths[-2], widths[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(self.embedding(features).flatten(1))


def fit_strong_base(
    x: np.ndarray,
    y: np.ndarray,
    train_rows: np.ndarray,
    valid_rows: np.ndarray,
    seed: int,
) -> np.ndarray:
    reducer = TruncatedSVD(n_components=100, random_state=seed)
    train_scores = reducer.fit_transform(y[train_rows])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model = Model7(train_scores.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005530331519967936)
    x_tensor = torch.as_tensor(x[train_rows], dtype=torch.long)
    y_tensor = torch.as_tensor(train_scores, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tensor, y_tensor),
        batch_size=128,
        shuffle=True,
        generator=generator,
    )
    model.train()
    for _ in range(141):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = torch.sqrt(torch.mean((prediction - batch_y) ** 2, dim=1) + 1e-12).mean()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        scores = model(torch.as_tensor(x[valid_rows], dtype=torch.long, device=device)).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return reducer.inverse_transform(scores).astype(np.float32)


@dataclass
class DataBundle:
    frame: pd.DataFrame
    genes: list[str]
    drugs: list[str]
    drug_index: dict[str, int]
    cell_index: dict[str, int]
    values: np.ndarray
    observed: np.ndarray
    anchors: np.ndarray
    features: np.ndarray
    strengths: np.ndarray
    label_x: np.ndarray
    row_y: np.ndarray
    complete: np.ndarray


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
    pca = PCA(n_components=SOURCE_COMPONENTS, random_state=OUTER_SEED)
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


def load_data() -> DataBundle:
    frame = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    metadata = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]
    genes = [column for column in frame.columns if column not in metadata]
    drugs = sorted(frame["sm_name"].unique())
    drug_index = {drug: idx for idx, drug in enumerate(drugs)}
    cell_index = {cell: idx for idx, cell in enumerate(CELLS)}
    values = np.full((len(drugs), len(CELLS), len(genes)), np.nan, dtype=np.float32)
    for _, row in frame.iterrows():
        cell = row["cell_type"]
        if cell in cell_index:
            values[drug_index[row["sm_name"]], cell_index[cell]] = row[genes].to_numpy(
                dtype=np.float32
            )
    observed = np.isfinite(values[..., 0])
    source_indices = [cell_index[cell] for cell in SOURCE_CELLS]
    target_indices = [cell_index[cell] for cell in TARGET_CELLS]
    anchors = build_anchor(values, observed, source_indices)
    features = build_source_features(values, observed, source_indices, anchors)
    strengths = np.sqrt(np.mean(anchors * anchors, axis=1)).astype(np.float32)
    labels = LabelEncoder()
    raw_x = frame[["cell_type", "sm_name"]].to_numpy()
    labels.fit(raw_x.flat)
    label_x = labels.transform(raw_x.flat).reshape(-1, 2)
    row_y = frame[genes].to_numpy(dtype=np.float32)
    complete = np.asarray(
        [
            drug
            for drug in drugs
            if all(observed[drug_index[drug], target] for target in target_indices)
        ]
    )
    return DataBundle(
        frame=frame,
        genes=genes,
        drugs=drugs,
        drug_index=drug_index,
        cell_index=cell_index,
        values=values,
        observed=observed,
        anchors=anchors,
        features=features,
        strengths=strengths,
        label_x=label_x,
        row_y=row_y,
        complete=complete,
    )


def component_prediction(
    train_features: np.ndarray,
    train_residuals: np.ndarray,
    valid_features: np.ndarray,
    random_state: int,
) -> np.ndarray:
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
        del model
    torch.cuda.empty_cache()
    return pca.inverse_transform(predicted_scores).astype(np.float32)


def predict_corrections(
    data: DataBundle,
    train_drugs: np.ndarray,
    valid_drugs: np.ndarray,
    stage_seed: int,
) -> np.ndarray:
    train_indices = np.asarray([data.drug_index[drug] for drug in train_drugs])
    valid_indices = np.asarray([data.drug_index[drug] for drug in valid_drugs])
    all_predictions = np.empty(
        (len(SEEDS), len(TARGET_CELLS), len(valid_drugs), len(data.genes)), dtype=np.float32
    )
    for seed_position, seed in enumerate(SEEDS):
        for target_position, target in enumerate(TARGET_CELLS):
            target_index = data.cell_index[target]
            train_residuals = (
                data.values[train_indices, target_index] - data.anchors[train_indices]
            )
            all_predictions[seed_position, target_position] = component_prediction(
                data.features[train_indices],
                train_residuals,
                data.features[valid_indices],
                stage_seed + 1000 * seed + 100 * target_position,
            )
    return all_predictions


def strong_base_prediction(
    data: DataBundle,
    train_drugs: np.ndarray,
    valid_drugs: np.ndarray,
    seed: int,
) -> np.ndarray:
    valid_mask = data.frame["sm_name"].isin(valid_drugs) & data.frame["cell_type"].isin(
        TARGET_CELLS
    )
    source_mask = data.frame["cell_type"].isin(SOURCE_CELLS)
    eligible_target_mask = data.frame["cell_type"].isin(TARGET_CELLS) & data.frame[
        "sm_name"
    ].isin(train_drugs)
    train_mask = source_mask | eligible_target_mask
    train_rows = np.flatnonzero(train_mask.to_numpy())
    valid_rows = np.flatnonzero(valid_mask.to_numpy())
    raw = fit_strong_base(data.label_x, data.row_y, train_rows, valid_rows, seed)
    mapped = {
        (data.frame.at[row, "sm_name"], data.frame.at[row, "cell_type"]): prediction
        for row, prediction in zip(valid_rows, raw)
    }
    return np.stack(
        [mapped[(drug, target)] for target in TARGET_CELLS for drug in valid_drugs]
    ).reshape(len(TARGET_CELLS), len(valid_drugs), len(data.genes))


def raw_reliability(
    data: DataBundle,
    train_drugs: np.ndarray,
    valid_drugs: np.ndarray,
    corrections: np.ndarray,
) -> dict[str, np.ndarray]:
    train_indices = np.asarray([data.drug_index[drug] for drug in train_drugs])
    valid_indices = np.asarray([data.drug_index[drug] for drug in valid_drugs])
    train_features = data.features[train_indices]
    valid_features = data.features[valid_indices]
    center = train_features.mean(axis=0)
    scale = train_features.std(axis=0) + 1e-4
    train_scaled = (train_features - center) / scale
    valid_scaled = (valid_features - center) / scale
    distance = np.sqrt(
        np.min(
            np.mean((valid_scaled[:, None, :] - train_scaled[None, :, :]) ** 2, axis=2),
            axis=1,
        )
    )
    mean_correction = corrections.mean(axis=0)
    seed_std = corrections.std(axis=0)
    disagreement = np.sqrt(np.mean(seed_std * seed_std, axis=-1)) / (
        np.sqrt(np.mean(mean_correction * mean_correction, axis=-1)) + 1e-6
    )
    strength = np.broadcast_to(data.strengths[valid_indices], disagreement.shape)
    distance_2d = np.broadcast_to(distance, disagreement.shape)
    return {
        "agreement": -np.log1p(disagreement),
        "distance": -np.log1p(distance_2d),
        "strength": np.log1p(strength),
    }


def percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    return np.searchsorted(reference, values, side="right") / max(len(reference), 1)


def quality(
    reliability: dict[str, np.ndarray],
    family: str,
    references: dict[str, np.ndarray],
) -> np.ndarray:
    if family == "constant":
        return np.zeros_like(reliability["agreement"], dtype=np.float64)
    ranks = {
        name: percentile(values, references[name]) for name, values in reliability.items()
    }
    if family == "blend":
        return (ranks["agreement"] + ranks["distance"] + ranks["strength"]) / 3.0
    return ranks[family]


def iter_gate_parameters():
    for family in GATE_FAMILIES:
        for lambda0 in LAMBDA0_GRID:
            for span in SPAN_GRID:
                if family == "constant" and span != 0.0:
                    continue
                for rho in RHO_GRID:
                    yield {
                        "family": family,
                        "lambda0": lambda0,
                        "span": span,
                        "rho": rho,
                    }


def gate_lambdas(q: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    base = np.clip(float(params["lambda0"]) + float(params["span"]) * q, 0.0, 0.1)
    result = base.copy()
    result[1] *= float(params["rho"])
    return result


def truths_and_anchors(data: DataBundle, drugs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray([data.drug_index[drug] for drug in drugs])
    truth = np.stack(
        [data.values[indices, data.cell_index[target]] for target in TARGET_CELLS]
    )
    anchor = np.broadcast_to(data.anchors[indices], truth.shape)
    return truth, anchor


def choose_gate(
    base: np.ndarray,
    truth: np.ndarray,
    corrections: np.ndarray,
    reliability: dict[str, np.ndarray],
) -> tuple[dict[str, float | str], dict[str, np.ndarray], list[dict[str, object]]]:
    mean_correction = corrections.mean(axis=0)
    references = {name: values.reshape(-1) for name, values in reliability.items()}
    base_errors = mr_rmse(base.reshape(-1, base.shape[-1]), truth.reshape(-1, truth.shape[-1]))
    base_target_errors = base_errors.reshape(len(TARGET_CELLS), -1).mean(axis=1)
    records = []
    for params in iter_gate_parameters():
        q = quality(reliability, str(params["family"]), references)
        lambdas = gate_lambdas(q, params)
        prediction = base + lambdas[..., None] * mean_correction
        errors = mr_rmse(
            prediction.reshape(-1, prediction.shape[-1]), truth.reshape(-1, truth.shape[-1])
        ).reshape(len(TARGET_CELLS), -1)
        target_errors = errors.mean(axis=1)
        regression = np.maximum(target_errors - base_target_errors, 0.0)
        score = float(errors.mean() + 4.0 * regression.sum())
        records.append(
            {
                **params,
                "selection_score": score,
                "combined_error": float(errors.mean()),
                "target_errors": dict(zip(TARGET_CELLS, map(float, target_errors))),
                "target_deltas": dict(
                    zip(TARGET_CELLS, map(float, base_target_errors - target_errors))
                ),
            }
        )
    records.sort(key=lambda record: (record["selection_score"], record["combined_error"]))
    best = {key: records[0][key] for key in ("family", "lambda0", "span", "rho")}
    return best, references, records[:10]


def evaluate(
    base: np.ndarray,
    truth: np.ndarray,
    corrections: np.ndarray,
    reliability: dict[str, np.ndarray],
    params: dict[str, float | str],
    references: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = quality(reliability, str(params["family"]), references)
    lambdas = gate_lambdas(q, params)
    prediction = base + lambdas[..., None] * corrections.mean(axis=0)
    return prediction, lambdas, q


def summarize_result(
    base: np.ndarray,
    prediction: np.ndarray,
    truth: np.ndarray,
    corrections: np.ndarray,
    lambdas: np.ndarray,
    compounds: np.ndarray,
) -> dict[str, object]:
    base_error = mr_rmse(base.reshape(-1, base.shape[-1]), truth.reshape(-1, truth.shape[-1])).reshape(
        len(TARGET_CELLS), -1
    )
    model_error = mr_rmse(
        prediction.reshape(-1, prediction.shape[-1]), truth.reshape(-1, truth.shape[-1])
    ).reshape(len(TARGET_CELLS), -1)
    delta = base_error - model_error
    per_compound = delta.mean(axis=0)
    positive_sum = float(np.maximum(per_compound, 0.0).sum())
    max_gain_share = (
        float(np.maximum(per_compound, 0.0).max() / positive_sum) if positive_sum > 0 else 1.0
    )
    seed_deltas = []
    for seed_position in range(len(SEEDS)):
        seed_prediction = base + lambdas[..., None] * corrections[seed_position]
        seed_error = mr_rmse(
            seed_prediction.reshape(-1, seed_prediction.shape[-1]),
            truth.reshape(-1, truth.shape[-1]),
        ).reshape(len(TARGET_CELLS), -1)
        seed_deltas.append(float((base_error - seed_error).mean()))
    targets = {}
    for target_position, target in enumerate(TARGET_CELLS):
        values = delta[target_position]
        targets[target] = {
            "base": float(base_error[target_position].mean()),
            "model": float(model_error[target_position].mean()),
            "delta": float(values.mean()),
            "wins": int((values > 0).sum()),
            "ci95": bootstrap_interval(values, OUTER_SEED + target_position),
            "mean_lambda": float(lambdas[target_position].mean()),
            "per_compound": dict(zip(compounds.tolist(), map(float, values))),
        }
    combined_ci = bootstrap_interval(per_compound)
    accept = bool(
        combined_ci[0] > 0.0
        and all(targets[target]["delta"] > 0.0 for target in TARGET_CELLS)
        and all(targets[target]["wins"] >= 10 for target in TARGET_CELLS)
        and all(value > 0.0 for value in seed_deltas)
        and max_gain_share <= 0.5
    )
    return {
        "combined_delta": float(per_compound.mean()),
        "combined_ci95_by_compound": combined_ci,
        "combined_wins": int((per_compound > 0).sum()),
        "seed_combined_deltas": dict(zip(map(str, SEEDS), seed_deltas)),
        "max_single_compound_positive_gain_share": max_gain_share,
        "targets": targets,
        "accept": accept,
    }


def main() -> None:
    for path in (REPORT_PATH.parent, OUTPUT_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)
    data = load_data()
    print(f"complete_compounds={len(data.complete)} genes={len(data.genes)}", flush=True)
    outer = KFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    outer_base: dict[str, np.ndarray] = {}
    outer_truth: dict[str, np.ndarray] = {}
    outer_corrections: dict[str, np.ndarray] = {}
    outer_reliability: dict[str, dict[str, np.ndarray]] = {}
    outer_lambdas: dict[str, np.ndarray] = {}
    outer_quality: dict[str, np.ndarray] = {}
    fold_records = []

    for outer_fold, (outer_train_pos, outer_valid_pos) in enumerate(outer.split(data.complete)):
        outer_train = data.complete[outer_train_pos]
        outer_valid = data.complete[outer_valid_pos]
        inner = KFold(n_splits=3, shuffle=True, random_state=OUTER_SEED + outer_fold)
        inner_base_parts = []
        inner_truth_parts = []
        inner_correction_parts = []
        inner_reliability_parts = {name: [] for name in ("agreement", "distance", "strength")}

        for inner_fold, (inner_train_pos, inner_valid_pos) in enumerate(inner.split(outer_train)):
            inner_train = outer_train[inner_train_pos]
            inner_valid = outer_train[inner_valid_pos]
            stage_seed = 100000 * outer_fold + 10000 * inner_fold
            corrections = predict_corrections(data, inner_train, inner_valid, stage_seed)
            base = strong_base_prediction(data, inner_train, inner_valid, OUTER_SEED + stage_seed)
            truth, _ = truths_and_anchors(data, inner_valid)
            reliability = raw_reliability(data, inner_train, inner_valid, corrections)
            inner_base_parts.append(base)
            inner_truth_parts.append(truth)
            inner_correction_parts.append(corrections)
            for name, values in reliability.items():
                inner_reliability_parts[name].append(values)
            print(
                f"outer={outer_fold} inner={inner_fold} train={len(inner_train)} valid={len(inner_valid)}",
                flush=True,
            )

        inner_base = np.concatenate(inner_base_parts, axis=1)
        inner_truth = np.concatenate(inner_truth_parts, axis=1)
        inner_corrections = np.concatenate(inner_correction_parts, axis=2)
        inner_reliability = {
            name: np.concatenate(parts, axis=1) for name, parts in inner_reliability_parts.items()
        }
        params, references, top_inner = choose_gate(
            inner_base, inner_truth, inner_corrections, inner_reliability
        )

        stage_seed = 900000 + 10000 * outer_fold
        corrections = predict_corrections(data, outer_train, outer_valid, stage_seed)
        base = strong_base_prediction(data, outer_train, outer_valid, OUTER_SEED + stage_seed)
        truth, _ = truths_and_anchors(data, outer_valid)
        reliability = raw_reliability(data, outer_train, outer_valid, corrections)
        prediction, lambdas, q = evaluate(
            base, truth, corrections, reliability, params, references
        )
        for position, drug in enumerate(outer_valid):
            outer_base[drug] = base[:, position]
            outer_truth[drug] = truth[:, position]
            outer_corrections[drug] = corrections[:, :, position]
            outer_reliability[drug] = {
                name: values[:, position] for name, values in reliability.items()
            }
            outer_lambdas[drug] = lambdas[:, position]
            outer_quality[drug] = q[:, position]
        fold_records.append(
            {
                "fold": outer_fold,
                "train": outer_train.tolist(),
                "valid": outer_valid.tolist(),
                "selected_gate": params,
                "top_inner_candidates": top_inner,
            }
        )
        fold_delta = summarize_result(
            base, prediction, truth, corrections, lambdas, outer_valid
        )["combined_delta"]
        print(
            f"outer={outer_fold} valid={outer_valid.tolist()} gate={params} delta={fold_delta:.6f}",
            flush=True,
        )

    base = np.stack([outer_base[drug] for drug in data.complete], axis=1)
    truth = np.stack([outer_truth[drug] for drug in data.complete], axis=1)
    corrections = np.stack([outer_corrections[drug] for drug in data.complete], axis=2)
    reliability = {
        name: np.stack([outer_reliability[drug][name] for drug in data.complete], axis=1)
        for name in ("agreement", "distance", "strength")
    }
    lambdas = np.stack([outer_lambdas[drug] for drug in data.complete], axis=1)
    quality_values = np.stack([outer_quality[drug] for drug in data.complete], axis=1)
    prediction = base + lambdas[..., None] * corrections.mean(axis=0)
    nested_summary = summarize_result(
        base, prediction, truth, corrections, lambdas, data.complete
    )

    final_params, final_references, final_top = choose_gate(
        base, truth, corrections, reliability
    )
    final_prediction, final_lambdas_oof, final_quality_oof = evaluate(
        base, truth, corrections, reliability, final_params, final_references
    )
    refit_summary = summarize_result(
        base, final_prediction, truth, corrections, final_lambdas_oof, data.complete
    )

    report = {
        "method": "Reliability-Adaptive Multi-Target Residual Transfer",
        "validation": "nested compound-held-out gate selection against strong neural OOF base",
        "complete_compounds": data.complete.tolist(),
        "seeds": list(SEEDS),
        "folds": fold_records,
        "nested_summary": nested_summary,
        "final_refit_gate": final_params,
        "final_refit_top_candidates": final_top,
        "final_refit_oof_summary": refit_summary,
        "submission_gate_passed": bool(nested_summary["accept"]),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        OOF_PATH,
        compounds=data.complete,
        base=base,
        truth=truth,
        corrections=corrections,
        lambdas=lambdas,
        quality=quality_values,
        final_lambdas=final_lambdas_oof,
        final_quality=final_quality_oof,
    )
    print(json.dumps(report, indent=2), flush=True)

    if not nested_summary["accept"]:
        print("SUBMISSION_GATE_FAILED", flush=True)
        return

    all_drugs = np.asarray(data.drugs)
    full_corrections = predict_corrections(data, data.complete, all_drugs, 1900000)
    full_reliability = raw_reliability(data, data.complete, all_drugs, full_corrections)
    q = quality(full_reliability, str(final_params["family"]), final_references)
    full_lambdas = gate_lambdas(q, final_params)
    np.savez_compressed(
        FULL_PATH,
        drugs=all_drugs,
        corrections=full_corrections,
        lambdas=full_lambdas,
        quality=q,
        seeds=np.asarray(SEEDS),
    )

    base_submission = read_submission(BASE_PATH)
    id_map = pd.read_csv(ID_MAP_PATH)
    if len(base_submission) != len(id_map):
        raise ValueError("Submission and id_map row counts differ")
    if not np.array_equal(base_submission.iloc[:, 0].to_numpy(), id_map["id"].to_numpy()):
        raise ValueError("Submission IDs do not align with id_map")
    drug_position = {drug: position for position, drug in enumerate(all_drugs)}
    target_position = {target: position for position, target in enumerate(TARGET_CELLS)}
    candidate = base_submission.copy()
    candidate_values = candidate.iloc[:, 1:].to_numpy(dtype=np.float64)
    base_values = candidate_values.copy()
    modified = []
    for row in range(len(id_map)):
        drug = id_map.at[row, "sm_name"]
        target = id_map.at[row, "cell_type"]
        if target not in target_position or drug not in drug_position:
            continue
        d = drug_position[drug]
        t = target_position[target]
        correction = full_corrections[:, t, d].mean(axis=0)
        candidate_values[row] += full_lambdas[t, d] * correction
        modified.append(row)
    if not np.isfinite(candidate_values).all():
        raise ValueError("Candidate contains non-finite values")
    candidate.iloc[:, 1:] = candidate_values
    with zipfile.ZipFile(OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.csv", candidate.to_csv(index=False).encode("utf-8"))
    with zipfile.ZipFile(OUTPUT_PATH) as archive:
        bad_member = archive.testzip()
    reread = read_submission(OUTPUT_PATH)
    reread_values = reread.iloc[:, 1:].to_numpy(dtype=np.float64)
    audit = {
        "method": report["method"],
        "base": BASE_PATH.name,
        "validation_report": str(REPORT_PATH),
        "submission_gate_passed": True,
        "final_gate": final_params,
        "modified_rows": len(modified),
        "shape": list(reread.shape),
        "unique_ids": int(reread.iloc[:, 0].nunique()),
        "finite": bool(np.isfinite(reread_values).all()),
        "zip_bad_member": bad_member,
        "delta_rms": float(np.sqrt(np.mean((reread_values - base_values) ** 2))),
        "delta_max_abs": float(np.max(np.abs(reread_values - base_values))),
        "sha256": sha256(OUTPUT_PATH),
        "output": str(OUTPUT_PATH),
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
