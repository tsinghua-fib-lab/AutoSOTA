#!/usr/bin/env python3
"""Probe 6: capacity / regularisation diagnostic for the tabular LightGBM.

Motivation (from probe 5): the published recipe reaches in-sample F1 = 1.0000 on
the training folds while scoring 0.69 on the held-out fold, i.e. it memorises
near-duplicate neighbours.  The compact grid below asks whether a *less*
expressive model generalises better across the 150 m blocks.  Every cell is
reported (not only the winner) so the result is auditable.
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

CLS = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
GRID = [
    # (num_leaves, min_child_samples, feature_fraction, tag)
    (31, 20, 1.0, "published"),
    (31, 40, 1.0, ""), (31, 100, 1.0, ""),
    (15, 20, 1.0, ""), (7, 20, 1.0, ""), (4, 20, 1.0, ""),
    (31, 20, 0.5, ""), (15, 40, 0.5, ""), (15, 100, 0.5, ""),
    (7, 60, 0.5, ""), (15, 40, 0.3, ""),
]


def wf1(y, pred):
    return float(f1_score(y, pred, average="weighted"))


def fold_mean(y, pred, folds):
    return float(np.mean([wf1(y[folds == f], pred[folds == f]) for f in sorted(set(folds.tolist()))]))


def fold_vec(y, pred, folds):
    return [round(wf1(y[folds == f], pred[folds == f]), 4) for f in sorted(set(folds.tolist()))]


def main():
    tr = pd.read_csv("/paper_data/train.csv")
    folds = pd.read_csv("/paper_data/features/fold_indices.csv").sort_values("point_index")["fold"].to_numpy()
    y = np.array([CLS.index(c) for c in tr["class"]])
    v7 = pd.read_csv("/paper_data/features/train_features_v07.csv")
    v9 = pd.read_csv("/paper_data/features/train_features_v09.csv")
    v7c = [c for c in v7.columns if c not in ("x", "y", "class")]
    X = np.column_stack([v7[v7c].to_numpy(), v9["acr_sapa_w21"].to_numpy()])
    cnn = np.load("outputs/cnn_oof.npy")
    base = None

    print(f"{'leaves':<8}{'mcs':<6}{'ff':<6}{'tab':<9}{'delta':<9}{'blend':<9}{'delta':<9}per-fold blend delta")
    for nl, mcs, ff, tag in GRID:
        t0 = time.time()
        P = dict(n_estimators=1000, random_state=42, verbose=-1, n_jobs=8, num_leaves=nl, max_depth=-1,
                 learning_rate=0.1, min_child_samples=mcs, feature_fraction=ff,
                 bagging_fraction=1.0, bagging_freq=0, reg_alpha=0.0, reg_lambda=0.0,
                 class_weight={i: (2.0 if i == 0 else 1.0) for i in range(5)})
        oof = np.zeros((len(y), 5))
        for f in sorted(set(folds.tolist())):
            tr_i, va_i = np.where(folds != f)[0], np.where(folds == f)[0]
            clf = lgb.LGBMClassifier(**P)
            clf.fit(X[tr_i], y[tr_i])
            oof[va_i] = clf.predict_proba(X[va_i])
        tm = fold_mean(y, oof.argmax(1), folds)
        b = 0.4 * oof + 0.6 * cnn
        bm = fold_mean(y, b.argmax(1), folds)
        fv = fold_vec(y, b.argmax(1), folds)
        if base is None:
            base = (tm, bm, fv)
            dt, db, dv = 0.0, 0.0, [0.0] * 5
        else:
            dt, db = tm - base[0], bm - base[1]
            dv = [round(a - c, 4) for a, c in zip(fv, base[2])]
        star = "  <-- PUBLISHED" if tag else ("  ***" if bm > base[1] else "")
        print(f"{nl:<8}{mcs:<6}{ff:<6}{tm:<9.4f}{dt:<+9.4f}{bm:<9.4f}{db:<+9.4f}{dv}{star}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
