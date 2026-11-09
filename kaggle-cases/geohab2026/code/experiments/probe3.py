#!/usr/bin/env python3
"""Probe 3: tabular error structure + I4 (categorical/sentinel) + class-weight knob."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, confusion_matrix

CLS = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]


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
    cnn = np.load("outputs/cnn_oof.npy")
    tab = np.load("outputs/tab_oof.npy")
    v7 = pd.read_csv("/paper_data/features/train_features_v07.csv")
    v9 = pd.read_csv("/paper_data/features/train_features_v09.csv")
    v7c = [c for c in v7.columns if c not in ("x", "y", "class")]

    print("=== tabular confusion matrix (OOF) ===")
    cm = confusion_matrix(y, tab.argmax(1), labels=range(5))
    print("        " + "".join(f"pred{c:<7}" for c in CLS))
    for i, c in enumerate(CLS):
        print(f"true {c:<4}" + "".join(f"{cm[i, j]:<11d}" for j in range(5)))
    print("  row-normalised errors (true->pred, %):")
    for i, c in enumerate(CLS):
        errs = [(CLS[j], cm[i, j] / cm[i].sum() * 100) for j in range(5) if j != i and cm[i, j] > 0]
        errs.sort(key=lambda t: -t[1])
        print(f"    {c:<5} n={cm[i].sum():<5} " + ", ".join(f"{n} {p:.1f}%" for n, p in errs[:3]))

    print("\n=== categorical / sentinel survey ===")
    for c in ("geomorphons", "pennock"):
        tr_v = v7[c]
        te_v = pd.read_csv("/paper_data/features/test_features_v07.csv")[c]
        print(f"  {c}: train uniq={sorted(tr_v.unique().tolist())}")
        print(f"  {c}: test  uniq={sorted(te_v.unique().tolist())}")
    print(f"  pennock == -128 in train: {(v7['pennock'] == -128).sum()}  in test: "
          f"{(pd.read_csv('/paper_data/features/test_features_v07.csv')['pennock'] == -128).sum()}")
    nm_tr = v7[v7c].isna().sum(1)
    nm_te = pd.read_csv("/paper_data/features/test_features_v07.csv")[v7c].isna().sum(1)
    print(f"  rows with >=1 NaN: train={int((nm_tr>0).sum())} test={int((nm_te>0).sum())}")

    X = np.column_stack([v7[v7c].to_numpy(), v9["acr_sapa_w21"].to_numpy()])
    names = v7c + ["acr_sapa_w21"]

    def run(tag, Xm, names_m, cat=(), alg_w=2.0, extra=None):
        t0 = time.time()
        P = dict(n_estimators=1000, random_state=42, verbose=-1, n_jobs=8, num_leaves=31, max_depth=-1,
                 learning_rate=0.1, min_child_samples=20, feature_fraction=1.0,
                 bagging_fraction=1.0, bagging_freq=0, reg_alpha=0.0, reg_lambda=0.0,
                 class_weight={i: (alg_w if i == 0 else 1.0) for i in range(5)})
        ci = [names_m.index(c) for c in cat]
        oof = np.zeros((len(y), 5))
        for f in sorted(set(folds.tolist())):
            tr_i, va_i = np.where(folds != f)[0], np.where(folds == f)[0]
            clf = lgb.LGBMClassifier(**P)
            clf.fit(Xm[tr_i], y[tr_i], **({"categorical_feature": ci} if ci else {}))
            oof[va_i] = clf.predict_proba(Xm[va_i])
        tm = fold_mean(y, oof.argmax(1), folds)
        bl = 0.4 * oof + 0.6 * cnn
        bm = fold_mean(y, bl.argmax(1), folds)
        print(f"  [{tag:<22}] tab={tm:.4f} {fold_vec(y, oof.argmax(1), folds)}  "
              f"blend={bm:.4f} {fold_vec(y, bl.argmax(1), folds)}  ({time.time()-t0:.0f}s)", flush=True)
        if extra:
            extra["last"] = oof
        return oof

    print("\n=== I4 categorical + sentinel ===")
    run("base (ref)", X, names)
    Xs = X.copy()
    si = names.index("pennock")
    Xs[Xs[:, si] == -128] = np.nan
    run("pennock -128->NaN", Xs, names)
    run("categorical(geomorph,penn)", X, names, cat=("geomorphons", "pennock"))
    run("cat + sentinel", Xs, names, cat=("geomorphons", "pennock"))

    print("\n=== class_weight ALG knob (with base features) ===")
    for aw in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        run(f"alg_w={aw}", X, names, alg_w=aw)


if __name__ == "__main__":
    main()
