#!/usr/bin/env python3
"""Standalone generator for the extra ddG-transfer components.

Uses exactly the same feature pipeline as pipeline/run_pipeline.py, so the CSVs it
writes are identical to what the evaluator's run of `bash scripts/eval.sh`
regenerates inside the pipeline (which is the authoritative path).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/repo/pipeline")
import run_pipeline as rp  # noqa: E402
sys.path.insert(0, "/repo/sota")
from oof_models import build_train  # noqa: E402

SPECS = [
    {"name": "etx", "kind": "extratrees", "n_estimators": 500, "min_samples_leaf": 2,
     "max_features": 0.5, "random_state": 42, "n_jobs": 8},
    {"name": "rgx", "kind": "ridge", "alpha": 100.0},
    {"name": "knx", "kind": "knn", "n_neighbors": 150, "n_jobs": 8},
]


def main():
    paper = Path("/paper_data")
    out = Path("/repo/outputs/components")
    out.mkdir(parents=True, exist_ok=True)

    train = build_train()
    t0 = time.time()
    test = pd.read_csv(paper / "test.csv")
    test["wildtype"] = rp.WT_SEQUENCE
    test["len"] = test.protein_sequence.map(lambda x: len(x))
    test["pdb"] = "WT"
    test = test.apply(rp.get_info_mutation, axis=1)
    test = rp.get_features(test)
    rp.load_dict_dssp(test, paper / "jin" / "pdbs")
    test = rp.get_info_pdb(test)
    print(f"test frame built in {time.time() - t0:.1f}s")

    X = train[rp.FEATURES]
    y = train["target"]
    for spec in SPECS:
        name = spec["name"]
        m = rp.build_extra_model(spec)
        m.fit(X, y)
        preds = pd.DataFrame({"seq_id": test["seq_id"], "tm": m.predict(test[rp.FEATURES])})
        preds.loc[preds[preds.seq_id == 32559].index, "tm"] = 0
        path = out / f"submission_{name}.csv"
        preds.to_csv(path, index=False)
        print(f"wrote {path} n={len(preds)} distinct={preds.tm.nunique()}")


if __name__ == "__main__":
    main()
