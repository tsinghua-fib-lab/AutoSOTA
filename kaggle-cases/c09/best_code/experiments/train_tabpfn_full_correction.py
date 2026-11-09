from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/workspace/c09-single-cell")
sys.path.insert(0, str(ROOT / "experiments"))

from tabpfn_component_cv import (  # noqa: E402
    CELLS,
    SOURCE_CELLS,
    build_anchor,
    build_source_features,
    component_prediction,
)


DATA_PATH = ROOT / "data" / "de_train.parquet"
OUTPUT_PATH = ROOT / "experiments" / "tabpfn_full_b_correction.npz"
TARGET = "B cells"
SEEDS = (17, 29, 43)


def main() -> None:
    frame = pd.read_parquet(DATA_PATH)
    metadata = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]
    genes = [column for column in frame.columns if column not in metadata]
    drugs = sorted(frame["sm_name"].unique())
    drug_index = {drug: idx for idx, drug in enumerate(drugs)}
    cell_index = {cell: idx for idx, cell in enumerate(CELLS)}
    source_indices = [cell_index[cell] for cell in SOURCE_CELLS]
    target_index = cell_index[TARGET]
    values = np.full((len(drugs), len(CELLS), len(genes)), np.nan, dtype=np.float32)
    for _, row in frame.iterrows():
        cell = row["cell_type"]
        if cell in cell_index:
            values[drug_index[row["sm_name"]], cell_index[cell]] = row[genes].to_numpy(dtype=np.float32)
    observed = np.isfinite(values[..., 0])
    anchors = build_anchor(values, observed, source_indices)
    features = build_source_features(values, observed, source_indices, anchors)
    train_indices = np.flatnonzero(observed[:, target_index])
    train_residuals = values[train_indices, target_index] - anchors[train_indices]
    corrections = []
    for seed in SEEDS:
        correction, component_count = component_prediction(
            features[train_indices], train_residuals, features, seed + 20000
        )
        corrections.append(correction)
        print(f"seed={seed} components={component_count}", flush=True)
    correction = np.mean(np.stack(corrections), axis=0).astype(np.float32)
    np.savez_compressed(
        OUTPUT_PATH,
        drugs=np.asarray(drugs),
        correction=correction,
        seeds=np.asarray(SEEDS),
        target=np.asarray(TARGET),
    )
    print(
        {
            "rows": len(drugs),
            "genes": len(genes),
            "rms": float(np.sqrt(np.mean(correction * correction))),
            "max_abs": float(np.max(np.abs(correction))),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
