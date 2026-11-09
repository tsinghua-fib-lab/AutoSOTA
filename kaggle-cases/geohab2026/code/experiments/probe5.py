#!/usr/bin/env python3
"""Probe 5: convergence diagnostic for the tabular LightGBM.

Fits ONE model per fold with a generous iteration cap and reads predictions at
several iteration counts.  This is a diagnostic (is the published recipe
converged?), not a hyper-parameter search engine.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

CLS = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
K = [50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 2000, 3000]


def wf1(y, pred):
    return float(f1_score(y, pred, average="weighted"))


def fold_mean(y, pred, folds):
    return float(np.mean([wf1(y[folds == f], pred[folds == f]) for f in sorted(set(folds.tolist()))]))


def main():
    tr = pd.read_csv("/paper_data/train.csv")
    folds = pd.read_csv("/paper_data/features/fold_indices.csv").sort_values("point_index")["fold"].to_numpy()
    y = np.array([CLS.index(c) for c in tr["class"]])
    v7 = pd.read_csv("/paper_data/features/train_features_v07.csv")
    v9 = pd.read_csv("/paper_data/features/train_features_v09.csv")
    v7c = [c for c in v7.columns if c not in ("x", "y", "class")]
    X = np.column_stack([v7[v7c].to_numpy(), v9["acr_sapa_w21"].to_numpy()])
    cnn = np.load("outputs/cnn_oof.npy")

    P = dict(n_estimators=3000, random_state=42, verbose=-1, n_jobs=8, num_leaves=31, max_depth=-1,
             learning_rate=0.1, min_child_samples=20, feature_fraction=1.0,
             bagging_fraction=1.0, bagging_freq=0, reg_alpha=0.0, reg_lambda=0.0,
             class_weight={i: (2.0 if i == 0 else 1.0) for i in range(5)})

    oof_by_k = {k: np.zeros((len(y), 5)) for k in K}
    train_f1_at = {}
    for f in sorted(set(folds.tolist())):
        tr_i, va_i = np.where(folds != f)[0], np.where(folds == f)[0]
        clf = lgb.LGBMClassifier(**P)
        clf.fit(X[tr_i], y[tr_i])
        for k in K:
            oof_by_k[k][va_i] = clf.predict_proba(X[va_i], num_iteration=k)
        if f == 0:
            train_f1_at[1000] = wf1(y[tr_i], clf.predict(X[tr_i], num_iteration=1000))

    print(f"train-fold-0 in-sample F1 at 1000 trees = {train_f1_at[1000]:.4f}  (val fold-0 F1 = "
          f"{wf1(y[folds == 0], oof_by_k[1000][folds == 0].argmax(1)):.4f})")
    print(f"\n{'trees':<8}{'tab fold-mean':<16}{'blend(0.4) fold-mean':<22}per-fold blend")
    for k in K:
        o = oof_by_k[k]
        tm = fold_mean(y, o.argmax(1), folds)
        b = 0.4 * o + 0.6 * cnn
        bm = fold_mean(y, b.argmax(1), folds)
        fv = [round(wf1(y[folds == f], b[folds == f].argmax(1)), 4) for f in sorted(set(folds.tolist()))]
        mark = "  <-- published" if k == 1000 else ("  ***" if bm > 0.7688 else "")
        print(f"{k:<8}{tm:<16.4f}{bm:<22.4f}{fv}{mark}")


if __name__ == "__main__":
    main()
