#!/usr/bin/env python3
"""Screen 9 — FUSE-STACK: a small meta-model over the saved component posteriors.

The repo's own write-up flags "a stacker with a tree-based meta-model rather than logistic
regression" as untested future work after three logistic-regression stackers failed. The
library rates this the highest-overfit-risk idea, so every configuration is measured with the
same fold-honest protocol the base models use:

    meta-model for fold f is trained ONLY on rows with folds != f, using those rows' own
    out-of-fold component probabilities, and applied only to fold f.

Two shapes are tried:
  full   : the meta-model predicts the class directly from the 10 probability columns
  gated  : the meta-model only decides the tie when tab and cnn disagree; FUSE-02 rules otherwise
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "experiments")
from probe_referee import fold_vec, load, selftest  # noqa: E402

D = load()
selftest(D)
Y, FOLDS = D["y"], D["folds"]
TAB, CNN = D["tab"], D["cnn"]
BASE_VEC = fold_vec(Y, D["blend"].argmax(1), FOLDS)
BASE_MEAN = float(BASE_VEC.mean())


def sharpen(p, e):
    q = np.power(p, e)
    return q / q.sum(1, keepdims=True)


TS6 = sharpen(TAB, 6.0)


def cls_prf(y, pred, c):
    tp = int(np.sum((y == c) & (pred == c)))
    fp = int(np.sum((pred == c) & (y != c)))
    fn = int(np.sum((y == c) & (pred != c)))
    s = tp + fn
    if s == 0:
        return None
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / s
    return {"F1": 2 * pr * rec / (pr + rec) if (pr + rec) else 0.0}


def shares_for(mask, eps=0.5):
    sh = np.full(5, 0.5)
    for c in range(5):
        rt = cls_prf(Y[mask], TAB.argmax(1)[mask], c)
        rc = cls_prf(Y[mask], CNN.argmax(1)[mask], c)
        if rt is None or rc is None:
            continue
        sh[c] = (rt["F1"] + eps * 0.5) / (rt["F1"] + rc["F1"] + eps)
    return sh


INCUMBENT = np.zeros(len(Y), dtype=int)
INC_SHARES = {}
for f in range(5):
    va = FOLDS == f
    sh = shares_for(FOLDS != f)
    INC_SHARES[f] = sh
    INCUMBENT[va] = (0.40 * TS6[va] * sh[None, :] + 0.60 * CNN[va] * (1 - sh)[None, :]).argmax(1)
inc_vec = fold_vec(Y, INCUMBENT, FOLDS)
inc_mean = float(inc_vec.mean())
print(f"[incumbent FUSE-02] fold-mean={inc_mean:.6f} per-fold={np.round(inc_vec,4).tolist()}")

FEATS = {
    "raw10": np.column_stack([TAB, CNN]),
    "sharp10": np.column_stack([TS6, CNN]),
    "diff15": np.column_stack([TAB, CNN, TAB - CNN, TS6 - CNN, (TS6 - CNN).sum(1)]),
    "logit20": np.column_stack([np.log(np.maximum(TAB, 1e-6)), np.log(np.maximum(CNN, 1e-6))]),
}


def show(tag, pred):
    fv = fold_vec(Y, pred, FOLDS)
    pf = np.round(fv - inc_vec, 4)
    print(f"  {tag:44s} mean={fv.mean():.4f} d_inc={fv.mean()-inc_mean:+.4f} "
          f"d_base={fv.mean()-BASE_MEAN:+.4f} pos={int((fv-inc_vec >= -1e-12).sum())}/5 {pf.tolist()}")
    return {"tag": tag, "mean": float(fv.mean()), "delta_inc": float(fv.mean() - inc_mean),
            "n_pos": int((fv - inc_vec >= -1e-12).sum()), "per_fold_delta": pf.tolist()}


out = []
print("\n[Y] full meta-model (predicts the class directly)")
from lightgbm import LGBMClassifier  # noqa: E402

for fname in ["sharp10", "logit20", "diff15"]:
    X = FEATS[fname]
    for nl, mcs, ne in [(3, 200, 150), (7, 400, 100)]:
        pred = np.zeros(len(Y), dtype=int)
        for f in range(5):
            tr, va = FOLDS != f, FOLDS == f
            m = LGBMClassifier(n_estimators=ne, num_leaves=nl, min_child_samples=mcs,
                               learning_rate=0.05, verbose=-1, n_jobs=4, random_state=42)
            m.fit(X[tr], Y[tr])
            pred[va] = m.predict(X[va])
        out.append(show(f"Y/full {fname} nl={nl} mcs={mcs} ne={ne}", pred))

print("\n[Z] gated meta-model (only decides when tab and cnn disagree)")
DIS = TAB.argmax(1) != CNN.argmax(1)
print(f"    disagreement rate: {DIS.mean():.3f} ({int(DIS.sum())} of {len(Y)})")
for fname in ["raw10", "sharp10"]:
    X = FEATS[fname]
    for nl, mcs in [(3, 200), (7, 400)]:
        pred = INCUMBENT.copy()
        for f in range(5):
            tr, va = FOLDS != f, FOLDS == f
            m = LGBMClassifier(n_estimators=150, num_leaves=nl, min_child_samples=mcs,
                               learning_rate=0.05, verbose=-1, n_jobs=4, random_state=42)
            m.fit(X[tr & DIS], Y[tr & DIS])
            sub = va & DIS
            pred[sub] = m.predict(X[sub])
        out.append(show(f"Z/gated {fname} nl={nl} mcs={mcs}", pred))

print("\n[Z2] gated, but only override when the meta confidence is high")
for fname in ["sharp10"]:
    X = FEATS[fname]
    for thr in [0.5, 0.7]:
        pred = INCUMBENT.copy()
        for f in range(5):
            tr, va = FOLDS != f, FOLDS == f
            m = LGBMClassifier(n_estimators=150, num_leaves=3, min_child_samples=200,
                               learning_rate=0.05, verbose=-1, n_jobs=4, random_state=42)
            m.fit(X[tr & DIS], Y[tr & DIS])
            sub = va & DIS
            pp = m.predict_proba(X[sub])
            keep = pp.max(1) >= thr
            idx = np.where(sub)[0][keep]
            pred[idx] = pp[keep].argmax(1)
        out.append(show(f"Z2/gated-confd {fname} thr={thr}", pred))

print("\n[AA] multinomial logistic regression on the same 10 columns (round-1 failure recheck)")
from sklearn.linear_model import LogisticRegression  # noqa: E402

for C in [0.01, 0.1, 1.0]:
    X = FEATS["logit20"]
    pred = np.zeros(len(Y), dtype=int)
    for f in range(5):
        tr, va = FOLDS != f, FOLDS == f
        m = LogisticRegression(max_iter=2000, C=C, multi_class="multinomial")
        m.fit(X[tr], Y[tr])
        pred[va] = m.predict(X[va])
    out.append(show(f"AA/logreg C={C}", pred))

with open("experiments/screen_iter3_stack.json", "w") as fh:
    json.dump({"incumbent": inc_mean, "baseline": BASE_MEAN, "results": out}, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter3_stack.json")
