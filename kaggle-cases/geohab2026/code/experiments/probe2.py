#!/usr/bin/env python3
"""Probe 2: (a) seed sensitivity, (b) I3 class-weight output correction,
(c) blend-weight re-optimisation LOO protocol on the current matrices."""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

CLS = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
y_all = None


def wf1(y, pred):
    return float(f1_score(y, pred, average="weighted"))


def fold_mean(y, pred, folds):
    return float(np.mean([wf1(y[folds == f], pred[folds == f]) for f in sorted(set(folds.tolist()))]))


def main():
    tr = pd.read_csv("/paper_data/train.csv")
    folds = pd.read_csv("/paper_data/features/fold_indices.csv").sort_values("point_index")["fold"].to_numpy()
    y = np.array([CLS.index(c) for c in tr["class"]])
    tab = np.load("outputs/tab_oof.npy")
    cnn = np.load("outputs/cnn_oof.npy")
    v7 = pd.read_csv("/paper_data/features/train_features_v07.csv")
    v9 = pd.read_csv("/paper_data/features/train_features_v09.csv")
    v7c = [c for c in v7.columns if c not in ("x", "y", "class")]
    X = np.column_stack([v7[v7c].to_numpy(), v9["acr_sapa_w21"].to_numpy()])

    P = dict(n_estimators=1000, verbose=-1, n_jobs=8, num_leaves=31, max_depth=-1,
             learning_rate=0.1, min_child_samples=20, feature_fraction=1.0,
             bagging_fraction=1.0, bagging_freq=0, reg_alpha=0.0, reg_lambda=0.0,
             class_weight={i: (2.0 if i == 0 else 1.0) for i in range(5)})

    print("=== (a) seed sensitivity of the fold-0 LightGBM fit ===")
    tr_i = np.where(folds != 0)[0]
    va_i = np.where(folds == 0)[0]
    base_pred = None
    for seed in (42, 7, 123):
        clf = lgb.LGBMClassifier(**P, random_state=seed)
        clf.fit(X[tr_i], y[tr_i])
        p = clf.predict_proba(X[va_i])
        if base_pred is None:
            base_pred = p
            print(f"  seed={seed}: fitted")
        else:
            print(f"  seed={seed}: max|delta| vs seed 42 = {np.abs(p - base_pred).max():.3e}  "
                  f"identical={np.array_equal(p, base_pred)}")

    print("\n=== (b) I3: class-weight output correction on the tabular matrix ===")
    print("  (correction is applied post-hoc to predict_proba, so no retraining needed to test it)")
    w = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
    tab_corr = tab / w[None, :]
    tab_corr /= tab_corr.sum(1, keepdims=True)
    print(f"  corr argmax identical to raw tab argmax: {(tab_corr.argmax(1) == tab.argmax(1)).all()}")
    ws = np.arange(0.0, 1.01, 0.05)
    for tag, mat in (("raw tab", tab), ("corr tab", tab_corr)):
        rows = []
        for ww in ws:
            rows.append((ww, fold_mean(y, (ww * mat + (1 - ww) * cnn).argmax(1), folds)))
        bi = int(np.argmax([r[1] for r in rows]))
        print(f"  {tag:<9} best fixed w={rows[bi][0]:.2f} -> {rows[bi][1]:.4f}   "
              f"(w=0.40 -> {[r[1] for r in rows if abs(r[0]-0.4)<1e-9][0]:.4f})")

    print("\n  LOO-honest selection of w for each matrix (w on a 0.05 grid):")
    for tag, mat in (("raw tab", tab), ("corr tab", tab_corr)):
        pf, ch = [], []
        for f in sorted(set(folds.tolist())):
            m = folds != f
            bb, bs = None, -1
            for ww in ws:
                s = wf1(y[m], (ww * mat + (1 - ww) * cnn)[m].argmax(1))
                if s > bs:
                    bs, bb = s, ww
            pf.append(wf1(y[folds == f], (bb * mat + (1 - bb) * cnn)[folds == f].argmax(1)))
            ch.append(round(float(bb), 2))
        print(f"  {tag:<9} LOO fold-mean={np.mean(pf):.4f}  per-fold={[round(v,4) for v in pf]}  chosen={ch}")

    print("\n=== (c) mixture with the CNN prior-adjusted / power-scaled tab ===")
    for a in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5):
        t = tab ** a
        t /= t.sum(1, keepdims=True)
        rows = [(ww, fold_mean(y, (ww * t + (1 - ww) * cnn).argmax(1), folds)) for ww in ws]
        bi = int(np.argmax([r[1] for r in rows]))
        print(f"  tab^{a:<5} best w={rows[bi][0]:.2f} -> {rows[bi][1]:.4f}")


if __name__ == "__main__":
    main()
