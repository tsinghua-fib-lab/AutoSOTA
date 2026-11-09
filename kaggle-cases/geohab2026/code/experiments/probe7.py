#!/usr/bin/env python3
"""Probe 7: refine the regularisation response surface (is it a plateau or a spike?).

probe 6 showed a monotone improvement as min_child_samples grew 20 -> 40 -> 100 at
num_leaves=31, and smaller num_leaves also helped.  A real capacity effect should
appear as a smooth plateau; a noise spike would not.  This sweep maps the surface.
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

CLS = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]


def wf1(y, pred):
    return float(f1_score(y, pred, average="weighted"))


def fold_mean(y, pred, folds):
    return float(np.mean([wf1(y[folds == f], pred[folds == f]) for f in sorted(set(folds.tolist()))]))


def fold_vec(y, pred, folds):
    return [round(wf1(y[folds == f], pred[folds == f]), 4) for f in sorted(set(folds.tolist()))]


def run_cv(X, y, folds, nl, mcs, ff=1.0, n_est=1000):
    P = dict(n_estimators=n_est, random_state=42, verbose=-1, n_jobs=8, num_leaves=nl, max_depth=-1,
             learning_rate=0.1, min_child_samples=mcs, feature_fraction=ff,
             bagging_fraction=1.0, bagging_freq=0, reg_alpha=0.0, reg_lambda=0.0,
             class_weight={i: (2.0 if i == 0 else 1.0) for i in range(5)})
    oof = np.zeros((len(y), 5))
    for f in sorted(set(folds.tolist())):
        tr_i, va_i = np.where(folds != f)[0], np.where(folds == f)[0]
        clf = lgb.LGBMClassifier(**P)
        clf.fit(X[tr_i], y[tr_i])
        oof[va_i] = clf.predict_proba(X[va_i])
    return oof


def main():
    tr = pd.read_csv("/paper_data/train.csv")
    folds = pd.read_csv("/paper_data/features/fold_indices.csv").sort_values("point_index")["fold"].to_numpy()
    y = np.array([CLS.index(c) for c in tr["class"]])
    v7 = pd.read_csv("/paper_data/features/train_features_v07.csv")
    v9 = pd.read_csv("/paper_data/features/train_features_v09.csv")
    v7c = [c for c in v7.columns if c not in ("x", "y", "class")]
    X = np.column_stack([v7[v7c].to_numpy(), v9["acr_sapa_w21"].to_numpy()])
    cnn = np.load("outputs/cnn_oof.npy")

    print("=== min_child_samples sweep at num_leaves=31 ===")
    print(f"{'mcs':<7}{'tab':<9}{'blend':<9}per-fold blend delta vs base")
    for mcs in (20, 60, 80, 100, 130, 160, 200, 260):
        oof = run_cv(X, y, folds, 31, mcs)
        tm = fold_mean(y, oof.argmax(1), folds)
        b = 0.4 * oof + 0.6 * cnn
        bm = fold_mean(y, b.argmax(1), folds)
        fv = fold_vec(y, b.argmax(1), folds)
        dv = [round(a - c, 4) for a, c in zip(fv, [0.7841, 0.7642, 0.8078, 0.7606, 0.7274])]
        n_pos = sum(1 for d in dv if d > 0)
        print(f"{mcs:<7}{tm:<9.4f}{bm:<9.4f}{dv}  pos={n_pos}/5", flush=True)

    print("\n=== num_leaves sweep at min_child_samples=100 ===")
    print(f"{'nl':<7}{'tab':<9}{'blend':<9}per-fold blend delta")
    for nl in (4, 8, 12, 16, 20, 31, 48):
        oof = run_cv(X, y, folds, nl, 100)
        tm = fold_mean(y, oof.argmax(1), folds)
        b = 0.4 * oof + 0.6 * cnn
        bm = fold_mean(y, b.argmax(1), folds)
        fv = fold_vec(y, b.argmax(1), folds)
        dv = [round(a - c, 4) for a, c in zip(fv, [0.7841, 0.7642, 0.8078, 0.7606, 0.7274])]
        n_pos = sum(1 for d in dv if d > 0)
        print(f"{nl:<7}{tm:<9.4f}{bm:<9.4f}{dv}  pos={n_pos}/5", flush=True)


if __name__ == "__main__":
    main()
