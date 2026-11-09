from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path("/workspace/c09-single-cell")
DATA_PATH = ROOT / "data" / "de_train.parquet"
TFM_PATH = ROOT / "experiments" / "tabpfn_component_cv_predictions.npz"
OUTPUT_PATH = ROOT / "experiments" / "third_place_bridge_cv_results.json"
PREDICTION_PATH = ROOT / "experiments" / "third_place_bridge_cv_predictions.npz"
SEED = 17
SOURCE_CELLS = ["NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"]
TARGET_CELLS = ["B cells", "Myeloid cells"]
ALPHAS = (0.0, 0.05, 0.1, 0.2)


def mr_rmse(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((prediction - truth) ** 2, axis=1))


def bootstrap_interval(values: np.ndarray, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(20000)]
    )
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


class Model7(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(152, 48)
        widths = [96, 712, 400, 232, 216, output_dim]
        dropouts = [0.4903998136177629, 0.032371643764537134, 0.11138300987168903, 0.019885384663655765]
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


def fit_predict_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model = Model7(train_y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005530331519967936)
    x_tensor = torch.as_tensor(train_x, dtype=torch.long)
    y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
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
        result = model(torch.as_tensor(valid_x, dtype=torch.long, device=device)).cpu().numpy()
    return result


def build_anchors(frame: pd.DataFrame, genes: list[str]) -> dict[str, np.ndarray]:
    anchors = {}
    for drug, group in frame.groupby("sm_name", sort=False):
        source = group[group["cell_type"].isin(SOURCE_CELLS)]
        values = source[genes].to_numpy(dtype=np.float32)
        mean = values.mean(axis=0)
        median = np.median(values, axis=0)
        strength = float(np.sqrt(np.mean(mean * mean)))
        nk_rows = source[source["cell_type"] == "NK cells"]
        nk = nk_rows[genes].to_numpy(dtype=np.float32)[0] if len(nk_rows) else mean
        nk_strength = float(np.sqrt(np.mean(nk * nk)))
        if strength < 1.987:
            anchor = median
        elif nk_strength / max(strength, 1e-8) >= 1.5:
            anchor = 0.75 * nk + 0.25 * mean
        else:
            anchor = mean
        anchors[drug] = anchor.astype(np.float32)
    return anchors


def main() -> None:
    frame = pd.read_parquet(DATA_PATH).reset_index(drop=True)
    metadata = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]
    genes = [column for column in frame.columns if column not in metadata]
    labels = LabelEncoder()
    raw_x = frame[["cell_type", "sm_name"]].to_numpy()
    labels.fit(raw_x.flat)
    x = labels.transform(raw_x.flat).reshape(-1, 2)
    y = frame[genes].to_numpy(dtype=np.float32)
    anchors = build_anchors(frame, genes)

    target_counts = frame[frame["cell_type"].isin(TARGET_CELLS)].groupby("sm_name")["cell_type"].nunique()
    complete = sorted(target_counts[target_counts == len(TARGET_CELLS)].index.tolist())
    outer = KFold(n_splits=5, shuffle=True, random_state=SEED)
    base_predictions: dict[tuple[str, str], np.ndarray] = {}
    fold_records = []

    for fold, (_, valid_pos) in enumerate(outer.split(np.asarray(complete))):
        valid_drugs = set(np.asarray(complete)[valid_pos].tolist())
        valid_mask = frame["sm_name"].isin(valid_drugs) & frame["cell_type"].isin(TARGET_CELLS)
        train_mask = ~valid_mask
        train_rows = np.flatnonzero(train_mask.to_numpy())
        valid_rows = np.flatnonzero(valid_mask.to_numpy())
        reducer = TruncatedSVD(n_components=100, random_state=SEED + fold)
        train_scores = reducer.fit_transform(y[train_rows])
        predicted_scores = fit_predict_model(x[train_rows], train_scores, x[valid_rows], SEED + fold)
        prediction = reducer.inverse_transform(predicted_scores).astype(np.float32)
        for row, pred in zip(valid_rows, prediction):
            base_predictions[(frame.at[row, "sm_name"], frame.at[row, "cell_type"])] = pred
        fold_records.append({"fold": fold, "valid": sorted(valid_drugs), "rows": int(len(valid_rows))})
        print(f"fold={fold} rows={len(valid_rows)}", flush=True)

    tfm = np.load(TFM_PATH)
    tfm_compounds = tfm["compounds"].astype(str).tolist()
    tfm_position = {drug: idx for idx, drug in enumerate(tfm_compounds)}
    truth_rows = []
    base_rows = []
    correction_rows = []
    row_keys = []
    for drug in complete:
        for target in TARGET_CELLS:
            row = frame.index[(frame["sm_name"] == drug) & (frame["cell_type"] == target)][0]
            truth_rows.append(y[row])
            base_rows.append(base_predictions[(drug, target)])
            correction_rows.append(tfm[target][tfm_position[drug]] - anchors[drug])
            row_keys.append((drug, target))
    truth = np.stack(truth_rows)
    base = np.stack(base_rows)
    correction = np.stack(correction_rows)
    base_error = mr_rmse(base, truth)

    candidates = {}
    for alpha in ALPHAS:
        prediction = base + alpha * correction
        error = mr_rmse(prediction, truth)
        delta = base_error - error
        target_stats = {}
        for target in TARGET_CELLS:
            mask = np.asarray([key[1] == target for key in row_keys])
            target_delta = delta[mask]
            target_stats[target] = {
                "base": float(base_error[mask].mean()),
                "model": float(error[mask].mean()),
                "delta": float(target_delta.mean()),
                "wins": int((target_delta > 0).sum()),
                "ci95": bootstrap_interval(target_delta),
            }
        candidates[str(alpha)] = {
            "score": float(error.mean()),
            "delta": float(delta.mean()),
            "wins": int((delta > 0).sum()),
            "ci95": bootstrap_interval(delta),
            "targets": target_stats,
        }

    report = {
        "method": "third-place architecture OOF plus frozen TabPFN residual correction",
        "folds": fold_records,
        "base_score": float(base_error.mean()),
        "alphas": candidates,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        PREDICTION_PATH,
        keys=np.asarray([f"{drug}|{target}" for drug, target in row_keys]),
        truth=truth,
        base=base,
        correction=correction,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
