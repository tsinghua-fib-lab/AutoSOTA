#!/usr/bin/env python3
"""Probe transfer-correctness fixes for the LGBM component.

`position`, `len` and `group_size` are all *out of distribution* for the test set:
the test rows are one 221-residue enzyme whose wild-type appears 2413 times, so the
`group_size` feature takes the value 2413 whereas the Jin v2 training groups are
tens of rows at most, and `position` is an absolute residue index whose usable
range differs per Jin protein. `zone` (= position / len) is the scale-free version
and stays. This script trains LGBM on different feature subsets and writes each
one as a candidate component so it can be scored by the official scorer.
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

DROPS = {
    "lgA": [],                                  # reference: the paper's feature set
    "lgB": ["position", "group_size"],          # absolute index + OOD group size
    "lgC": ["group_size"],                      # OOD group size only
    "lgD": ["position"],                        # absolute index only
    "lgE": ["position", "group_size", "len"],   # + length
}


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
    print(f"test frame built in {time.time() - t0:.1f}s", flush=True)

    for name, drop in DROPS.items():
        feats = [f for f in rp.FEATURES if f not in drop]
        m = rp.get_model(42)
        m.fit(train[feats], train["target"])
        preds = pd.DataFrame({"seq_id": test["seq_id"], "tm": m.predict(test[feats])})
        preds.loc[preds[preds.seq_id == 32559].index, "tm"] = 0
        path = out / f"submission_{name}.csv"
        preds.to_csv(path, index=False)
        print(f"{name}: dropped {drop or '(none)'} -> {len(feats)} features, wrote {path}",
              flush=True)


if __name__ == "__main__":
    main()
