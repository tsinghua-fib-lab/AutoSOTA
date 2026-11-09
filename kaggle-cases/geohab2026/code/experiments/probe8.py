#!/usr/bin/env python3
"""Probe 8: low-num_leaves region, probability calibration, cross-fold class mix."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

CLS = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
REF_BLEND = 0.7785
REF_TAB = 0.7623
REF_FOLDS = [0.7857, 0.7788, 0.8246, 0.7602, 0.7430]


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

    print("=== per-fold class distribution and per-component per-class F1 ===")
    from sklearn.metrics import precision_recall_fscore_support as prfs
    for f in sorted(set(folds.tolist())):
        m = folds == f
        cnt = np.bincount(y[m], minlength=5)
        print(f"  fold {f}: n={m.sum():<5} " + " ".join(f"{c}={cnt[i]}({cnt[i]/m.sum()*100:.1f}%)" for i, c in enumerate(CLS)))
    print("  blend per-fold per-class F1:")
    b = 0.4 * np.load("outputs/tab_oof.npy") + 0.6 * cnn
    for f in sorted(set(folds.tolist())):
        m = folds == f
        _, _, f1, _ = prfs(y[m], b[m].argmax(1), labels=range(5), zero_division=0)
        print(f"    fold {f}: " + " ".join(f"{c}={f1[i]:.2f}" for i, c in enumerate(CLS)))

    print("\n=== A. low-num_leaves region at min_child_samples=100 ===")
    def cv(nl, mcs=100, ff=1.0, bag=1.0, bagfreq=0, n_est=1000):
        P = dict(n_estimators=n_est, random_state=42, verbose=-1, n_jobs=8, num_leaves=nl, max_depth=-1,
                 learning_rate=0.1, min_child_samples=mcs, feature_fraction=ff,
                 bagging_fraction=bag, bagging_freq=bagfreq, reg_alpha=0.0, reg_lambda=0.0,
                 class_weight={i: (2.0 if i == 0 else 1.0) for i in range(5)})
        oof = np.zeros((len(y), 5))
        for fidx in sorted(set(folds.tolist())):
            tr_i, va_i = np.where(folds != fidx)[0], np.where(folds == fidx)[0]
            clf = lgb.LGBMClassifier(**P)
            clf.fit(X[tr_i], y[tr_i])
            oof[va_i] = clf.predict_proba(X[va_i])
        return oof

    tab31 = None
    for nl in (2, 3, 4, 6, 8, 12, 31):
        t0 = time.time()
        oof = cv(nl)
        if nl == 31:
            tab31 = oof
        tm = fold_mean(y, oof.argmax(1), folds)
        bm = fold_mean(y, (0.4 * oof + 0.6 * cnn).argmax(1), folds)
        fv = fold_vec(y, (0.4 * oof + 0.6 * cnn).argmax(1), folds)
        dv = [round(a - c, 4) for a, c in zip(fv, REF_FOLDS)]
        print(f"  nl={nl:<3} tab={tm:.4f}  blend={bm:.4f}  d={bm-REF_BLEND:+.4f}  {dv} pos="
              f"{sum(1 for d in dv if d > 1e-9)}/5  ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== B. probability calibration inside the fixed 0.40 blend (tab31) ===")
    assert tab31 is not None
    eps = 1e-12
    print("   tab^a (no cnn change):")
    for a in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0):
        t = tab31 ** a
        t /= t.sum(1, keepdims=True)
        print(f"     a={a:<4} blend={fold_mean(y, (0.4 * t + 0.6 * cnn).argmax(1), folds):.4f}")
    print("   cnn^b (no tab change):")
    for bb in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0):
        c = cnn ** bb
        c /= c.sum(1, keepdims=True)
        print(f"     b={bb:<4} blend={fold_mean(y, (0.4 * tab31 + 0.6 * c).argmax(1), folds):.4f}")

    print("\n=== C. bagging at nl=31, mcs=100 ===")
    for bag in (0.6, 0.8):
        t0 = time.time()
        oof = cv(31, bag=bag, bagfreq=1)
        tm = fold_mean(y, oof.argmax(1), folds)
        bm = fold_mean(y, (0.4 * oof + 0.6 * cnn).argmax(1), folds)
        print(f"  bagging_fraction={bag}  tab={tm:.4f}  blend={bm:.4f}  d={bm-REF_BLEND:+.4f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
