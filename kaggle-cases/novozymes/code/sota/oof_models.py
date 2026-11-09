#!/usr/bin/env python3
"""Build the Jin training frame once, cache it, then report the 7-fold
GroupKFold OOF Spearman of candidate extra ddG-transfer models.

The OOF Spearman on Jin v2 is a test-label-free diagnostic: it uses only the
paper's own training data, never /paper_data/test_labels.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

sys.path.insert(0, "/repo/pipeline")
import run_pipeline as rp  # noqa: E402

CACHE = Path("/tmp/jin_train_feat.pkl")


def build_train():
    if CACHE.exists():
        print("using cached Jin training frame")
        return pd.read_pickle(CACHE)
    paper = Path("/paper_data")
    jin_dir = paper / "jin"
    jin_pdbs = jin_dir / "pdbs"
    t0 = time.time()
    df = pd.read_csv(jin_dir / "train_Jin_v2.txt")
    df.drop(["seq_index", "pdb_resseq", "wildtype", "mutation"], axis=1, inplace=True)
    df.rename(columns={"wt_seq": "wildtype", "mut_seq": "protein_sequence"}, inplace=True)
    df = df.apply(rp.get_info_mutation, axis=1)
    df["len"] = df.protein_sequence.map(lambda x: len(x))
    df["origen"] = "jin_ddG"
    df["measure"] = df["ddG"]
    df["UM"] = "ddG"
    df = df[["wildtype", "len", "position", "AA1", "AA2", "pdb", "origen", "UM", "measure"]]
    rp.check_wt_pdb(df, jin_pdbs)
    df.drop(df[(df.pdb == "1X0J") | (df.pdb == "3O39")].index, inplace=True)
    df = df.reset_index(drop=True)
    rp.load_dict_dssp(df, jin_pdbs)
    rp.check_alignment_pdb(df)
    df = rp.get_features(df)
    df = rp.get_info_pdb(df)
    df["target"] = df["measure"]
    df.to_pickle(CACHE)
    print(f"built Jin frame {df.shape} in {time.time() - t0:.1f}s")
    return df


def main():
    train = build_train()
    print(f"train {train.shape}, groups {train.group.nunique()}")
    skf = GroupKFold(n_splits=7)
    y = train["target"].to_numpy()

    models = {
        "lgbm_paper": ("paper", {}),
        "et500_leaf2_f0.5": ("et", dict(n_estimators=500, min_samples_leaf=2, max_features=0.5)),
        "et500_leaf5_f0.3": ("et", dict(n_estimators=500, min_samples_leaf=5, max_features=0.3)),
        "et1000_leaf1": ("et", dict(n_estimators=1000, min_samples_leaf=1, max_features=0.7)),
        "ridge_a10": ("ridge", dict(alpha=10.0)),
        "ridge_a100": ("ridge", dict(alpha=100.0)),
        "knn40": ("knn", dict(n_neighbors=40)),
        "knn150": ("knn", dict(n_neighbors=150)),
    }

    def make(kind, kw):
        if kind == "paper":
            return rp.get_model(42)
        if kind == "et":
            from sklearn.ensemble import ExtraTreesRegressor
            return ExtraTreesRegressor(random_state=42, n_jobs=8, **kw)
        if kind == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(StandardScaler(), Ridge(**kw))
        if kind == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(StandardScaler(), KNeighborsRegressor(weights="distance", n_jobs=8, **kw))
        raise ValueError(kind)

    for name, (kind, kw) in models.items():
        t0 = time.time()
        oof = np.zeros(len(train))
        for tr, va in skf.split(train, y, train.group):
            m = make(kind, kw)
            m.fit(train.loc[tr, rp.FEATURES], train.loc[tr, "target"])
            oof[va] = m.predict(train.loc[va, rp.FEATURES])
        rho = spearmanr(oof, y).correlation
        rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
        print(f"{name:20s} OOF Spearman={rho:.5f}  RMSE={rmse:.4f}   ({time.time() - t0:.1f}s)",
              flush=True)


if __name__ == "__main__":
    main()
