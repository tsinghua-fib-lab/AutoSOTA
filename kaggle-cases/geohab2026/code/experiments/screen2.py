#!/usr/bin/env python3
"""Screen 2 - at the corrected capacity (num_leaves=31, min_child_samples=100).

Reference for every delta is the iter-1 configuration actually recorded:
blend fold-mean 0.7785, per-fold [0.7857, 0.7788, 0.8246, 0.7602, 0.7430].

Axes probed: convergence at the new capacity, feature families re-tested under the
corrected capacity, and a capacity-averaged tabular model.
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

CLS = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
REF_BLEND = 0.7785
REF_FOLDS = [0.7857, 0.7788, 0.8246, 0.7602, 0.7430]
EPS = 1e-9


def wf1(y, pred):
    return float(f1_score(y, pred, average="weighted"))


def fold_mean(y, pred, folds):
    return float(np.mean([wf1(y[folds == f], pred[folds == f]) for f in sorted(set(folds.tolist()))]))


def fold_vec(y, pred, folds):
    return [round(wf1(y[folds == f], pred[folds == f]), 4) for f in sorted(set(folds.tolist()))]


def report(tag, oof, y, folds, cnn, t0):
    tm = fold_mean(y, oof.argmax(1), folds)
    b = 0.4 * oof + 0.6 * cnn
    bm = fold_mean(y, b.argmax(1), folds)
    fv = fold_vec(y, b.argmax(1), folds)
    dv = [round(a - c, 4) for a, c in zip(fv, REF_FOLDS)]
    npos = sum(1 for d in dv if d > 1e-9)
    star = " ***" if bm > REF_BLEND else ""
    print(f"  [{tag:<24}] tab={tm:.4f}  blend={bm:.4f}  delta={bm-REF_BLEND:+.4f}  {dv} pos={npos}/5"
          f"  ({time.time()-t0:.0f}s){star}", flush=True)
    return bm


def main():
    tr = pd.read_csv("/paper_data/train.csv")
    folds = pd.read_csv("/paper_data/features/fold_indices.csv").sort_values("point_index")["fold"].to_numpy()
    y = np.array([CLS.index(c) for c in tr["class"]])
    v7 = pd.read_csv("/paper_data/features/train_features_v07.csv")
    v9 = pd.read_csv("/paper_data/features/train_features_v09.csv")
    v7c = [c for c in v7.columns if c not in ("x", "y", "class")]
    X = np.column_stack([v7[v7c].to_numpy(), v9["acr_sapa_w21"].to_numpy()])
    cnn = np.load("outputs/cnn_oof.npy")

    def cv(Xm, nl=31, mcs=100, n_est=1000, ff=1.0, cat=(), alg_w=2.0, Xp_va=None):
        P = dict(n_estimators=n_est, random_state=42, verbose=-1, n_jobs=8, num_leaves=nl, max_depth=-1,
                 learning_rate=0.1, min_child_samples=mcs, feature_fraction=ff,
                 bagging_fraction=1.0, bagging_freq=0, reg_alpha=0.0, reg_lambda=0.0,
                 class_weight={i: (alg_w if i == 0 else 1.0) for i in range(5)})
        oof = np.zeros((len(y), 5))
        for fidx in sorted(set(folds.tolist())):
            tr_i, va_i = np.where(folds != fidx)[0], np.where(folds == fidx)[0]
            clf = lgb.LGBMClassifier(**P)
            clf.fit(Xm[tr_i], y[tr_i], **({"categorical_feature": list(cat)} if len(cat) else {}))
            oof[va_i] = clf.predict_proba(Xm[va_i])
        return oof

    print("=== A. convergence at the corrected capacity (num_leaves=31, mcs=100) ===")
    for n_est in (500, 1000, 2000):
        t0 = time.time()
        report(f"n_estimators={n_est}", cv(X, n_est=n_est), y, folds, cnn, t0)

    print("\n=== B. feature families re-tested under the corrected capacity ===")
    extra = {}
    for tag, pre in (("bathy", "bathy_focal"), ("bs", "bs_focal")):
        s41, s21, s11, s7, s3 = (v7[f"{pre}_std_w{w}"].to_numpy() for w in (41, 21, 11, 7, 3))
        m41, m3 = v7[f"{pre}_mean_w41"].to_numpy(), v7[f"{pre}_mean_w3"].to_numpy()
        r41, r21, r11, r3 = (v7[f"{pre}_range_w{w}"].to_numpy() for w in (41, 21, 11, 3))
        extra[f"{tag}_logstd_41_3"] = np.log((s41 + EPS) / (s3 + EPS))
        extra[f"{tag}_logstd_11_3"] = np.log((s11 + EPS) / (s3 + EPS))
        extra[f"{tag}_logstd_21_7"] = np.log((s21 + EPS) / (s7 + EPS))
        extra[f"{tag}_logrange_21_3"] = np.log((r21 + EPS) / (r3 + EPS))
        extra[f"{tag}_logrange_41_21"] = np.log((r41 + EPS) / (r21 + EPS))
        extra[f"{tag}_relief_41"] = m41 - m3
        extra[f"{tag}_range_over_std_21"] = r21 / (s21 + EPS)
        extra[f"{tag}_range_over_std_3"] = r3 / (s3 + EPS)
        extra[f"{tag}_rough_ratio"] = (s41 + EPS) / (s11 + EPS)
        extra[f"{tag}_rough_ratio2"] = (s11 + EPS) / (s21 + EPS)
    extra["bs_over_bathy_std_21"] = (v7["bs_focal_std_w21"].to_numpy() + EPS) / (v7["bathy_focal_std_w21"].to_numpy() + EPS)
    extra["bs_minus_bathy_std_3"] = v7["bs_focal_std_w3"].to_numpy() - v7["bathy_focal_std_w3"].to_numpy()
    Xr = np.column_stack([X, np.column_stack(list(extra.values()))])
    acf_cols = ["acf_decay_half_p51", "acf_decay_zero_p51", "acf_first_peak_dist_p51", "acf_first_peak_val_p51",
                "acf_energy_p51", "acf_aniso_ratio_p51", "acf_aniso_angle_p51", "acf_decay_half_p25",
                "acf_decay_zero_p25", "acf_first_peak_dist_p25", "acf_first_peak_val_p25", "acf_energy_p25",
                "acf_aniso_ratio_p25", "acf_aniso_angle_p25"]
    Xa = np.column_stack([X, v7[v7c].iloc[:, :0].to_numpy(), v9[acf_cols].to_numpy()])
    Xaa = np.column_stack([Xa, v9[["acr_sapa_w3", "acr_tin_w3"]].to_numpy()])
    Xar = np.column_stack([Xaa, np.column_stack(list(extra.values()))])
    t0 = time.time(); report("base (ref, mcs=100)", cv(X), y, folds, cnn, t0)
    t0 = time.time(); report("+ratio(22)", cv(Xr), y, folds, cnn, t0)
    t0 = time.time(); report("+acf(14)", cv(Xa), y, folds, cnn, t0)
    t0 = time.time(); report("+acr+acf(16)", cv(Xaa), y, folds, cnn, t0)
    t0 = time.time(); report("+acr+acf+ratio", cv(Xar), y, folds, cnn, t0)

    print("\n=== C. capacity-averaged tabular model (no argmax selection) ===")
    t0 = time.time()
    parts = [cv(X, nl=nl, mcs=mcs) for nl, mcs in ((31, 60), (31, 100), (31, 200))]
    report("mean(nl31,mcs{60,100,200})", np.mean(parts, 0), y, folds, cnn, t0)
    t0 = time.time()
    parts = [cv(X, nl=nl, mcs=100) for nl in (16, 31, 48)]
    report("mean(nl{16,31,48},mcs100)", np.mean(parts, 0), y, folds, cnn, t0)


if __name__ == "__main__":
    main()
