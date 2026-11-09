#!/usr/bin/env python3
"""Feature-family screening harness for the tabular half (exploration only).

Runs the exact LightGBM recipe of pipeline/run_pipeline.py (same CONFIG, same
frozen folds) over the same 5-fold CV, for a set of feature-block variants, and
reports both the tabular fold-mean and the resulting w=0.40 blend fold-mean.

Nothing here is written into the pipeline; it only ranks candidate blocks so
that a single block can be promoted into a recorded iteration.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

CLASS_NAMES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
LABEL_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = 5

LGB_PARAMS = dict(
    n_estimators=1000, random_state=42, verbose=-1, n_jobs=8,
    num_leaves=31, max_depth=-1, learning_rate=0.1, min_child_samples=20,
    feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0,
    reg_alpha=0.0, reg_lambda=0.0,
    class_weight={i: (2.0 if i == 0 else 1.0) for i in range(NUM_CLASSES)},
)
EPS = 1e-9


def wf1(y, pred):
    return float(f1_score(y, pred, average="weighted"))


def fold_stats(y, pred, folds):
    fv = [wf1(y[folds == f], pred[folds == f]) for f in sorted(set(folds.tolist()))]
    return float(np.mean(fv)), fv


def build_data(paper_data):
    tr = pd.read_csv(os.path.join(paper_data, "train.csv"))
    folds = pd.read_csv(os.path.join(paper_data, "features", "fold_indices.csv")) \
        .sort_values("point_index")["fold"].to_numpy().astype(int)
    y = np.array([LABEL_MAP[c] for c in tr["class"]])
    v7 = pd.read_csv(os.path.join(paper_data, "features", "train_features_v07.csv"))
    v9 = pd.read_csv(os.path.join(paper_data, "features", "train_features_v09.csv"))
    v7c = [c for c in v7.columns if c not in ("x", "y", "class")]
    cnn = np.load("outputs/cnn_oof.npy")
    return tr, y, folds, v7, v9, v7c, cnn


def variant_block(name, v7, v9):
    """Return (extra DataFrame of derived/native columns, categorical col names)."""
    d = pd.DataFrame(index=v7.index)
    cat = []
    if name == "base":
        pass

    if "acr" in name:
        d["acr_sapa_w3"] = v9["acr_sapa_w3"].to_numpy()
        d["acr_tin_w3"] = v9["acr_tin_w3"].to_numpy()

    if "acf" in name:
        for c in ["acf_decay_half_p51", "acf_decay_zero_p51", "acf_first_peak_dist_p51",
                  "acf_first_peak_val_p51", "acf_energy_p51", "acf_aniso_ratio_p51", "acf_aniso_angle_p51",
                  "acf_decay_half_p25", "acf_decay_zero_p25", "acf_first_peak_dist_p25",
                  "acf_first_peak_val_p25", "acf_energy_p25", "acf_aniso_ratio_p25", "acf_aniso_angle_p25"]:
            d[c] = v9[c].to_numpy()

    if "ratio" in name:
        for tag, pre in (("bathy", "bathy_focal"), ("bs", "bs_focal")):
            s41 = v7[f"{pre}_std_w41"].to_numpy()
            s21 = v7[f"{pre}_std_w21"].to_numpy()
            s11 = v7[f"{pre}_std_w11"].to_numpy()
            s7 = v7[f"{pre}_std_w7"].to_numpy()
            s3 = v7[f"{pre}_std_w3"].to_numpy()
            m41 = v7[f"{pre}_mean_w41"].to_numpy()
            m3 = v7[f"{pre}_mean_w3"].to_numpy()
            r41 = v7[f"{pre}_range_w41"].to_numpy()
            r21 = v7[f"{pre}_range_w21"].to_numpy()
            r11 = v7[f"{pre}_range_w11"].to_numpy()
            r3 = v7[f"{pre}_range_w3"].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                d[f"{tag}_logstd_41_3"] = np.log((s41 + EPS) / (s3 + EPS))
                d[f"{tag}_logstd_11_3"] = np.log((s11 + EPS) / (s3 + EPS))
                d[f"{tag}_logstd_21_7"] = np.log((s21 + EPS) / (s7 + EPS))
                d[f"{tag}_logrange_21_3"] = np.log((r21 + EPS) / (r3 + EPS))
                d[f"{tag}_logrange_41_21"] = np.log((r41 + EPS) / (r21 + EPS))
                d[f"{tag}_relief_41"] = m41 - m3
                d[f"{tag}_range_over_std_21"] = r21 / (s21 + EPS)
                d[f"{tag}_range_over_std_3"] = r3 / (s3 + EPS)
            d[f"{tag}_rough_ratio"] = (s41 + EPS) / (s11 + EPS)
            d[f"{tag}_rough_ratio2"] = (s11 + EPS) / (s21 + EPS)
        # cross-raster
        d["bs_over_bathy_std_21"] = (v7["bs_focal_std_w21"].to_numpy() + EPS) / (v7["bathy_focal_std_w21"].to_numpy() + EPS)
        d["bs_minus_bathy_std_3"] = v7["bs_focal_std_w3"].to_numpy() - v7["bathy_focal_std_w3"].to_numpy()

    if "cat" in name:
        cat = ["geomorphons", "pennock"]

    return d, cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-data", default="/paper_data")
    ap.add_argument("--variants", default="base,acr,acf,ratio,acr+ratio,acr+acf,acr+acf+ratio")
    ap.add_argument("--cnn-weight-used", type=float, default=0.60)
    ap.add_argument("--out", default="experiments/screen_results.json")
    args = ap.parse_args()

    tr, y, folds, v7, v9, v7c, cnn = build_data(args.paper_data)
    base_X = v7[v7c].to_numpy()
    base_names = list(v7c) + ["acr_sapa_w21"]
    base_X = np.column_stack([base_X, v9["acr_sapa_w21"].to_numpy()])
    print(f"[data] base X={base_X.shape}")

    results = {}
    for vname in args.variants.split(","):
        t0 = time.time()
        d, cat = variant_block(vname, v7, v9)
        # base already contains acr_sapa_w21; avoid duplicate column names
        d = d.loc[:, [c for c in d.columns if c not in base_names]]
        X = np.column_stack([base_X, d.to_numpy()]) if d.shape[1] else base_X
        names = base_names + list(d.columns)
        cat_idx = [names.index(c) for c in cat if c in names]

        oof = np.zeros((len(y), NUM_CLASSES))
        for f in sorted(set(folds.tolist())):
            tr_i = np.where(folds != f)[0]
            va_i = np.where(folds == f)[0]
            clf = lgb.LGBMClassifier(**LGB_PARAMS)
            kw = {"categorical_feature": cat_idx} if cat_idx else {}
            clf.fit(X[tr_i], y[tr_i], **kw)
            oof[va_i] = clf.predict_proba(X[va_i])

        tab_mean, tab_fv = fold_stats(y, oof.argmax(1), folds)
        blend = (1 - args.cnn_weight_used) * oof + args.cnn_weight_used * cnn
        bl_mean, bl_fv = fold_stats(y, blend.argmax(1), folds)
        results[vname] = {
            "n_features": int(X.shape[1]),
            "tab_fold_mean": round(tab_mean, 4), "tab_per_fold": [round(v, 4) for v in tab_fv],
            "blend_fold_mean": round(bl_mean, 4), "blend_per_fold": [round(v, 4) for v in bl_fv],
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[{vname:<16}] nfeat={X.shape[1]:3d}  tab={tab_mean:.4f} {[round(v,4) for v in tab_fv]}"
              f"  blend={bl_mean:.4f} {[round(v,4) for v in bl_fv]}  ({time.time()-t0:.0f}s)", flush=True)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[saved] {args.out}")
    print("\n=== ranked by blend fold-mean ===")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["blend_fold_mean"]):
        print(f"  {k:<18} blend={v['blend_fold_mean']:.4f}  tab={v['tab_fold_mean']:.4f}  nfeat={v['n_features']}")


if __name__ == "__main__":
    main()
