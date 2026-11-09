#!/usr/bin/env python3
"""Screen 20 — the decisive LOO gate for iteration 4 (component-wise smoothing).

Iteration 4 scored 0.79804, above the incumbent 0.79567, but was positive on only 3 of 5 folds.
The objective's success policy says a change is accepted only if it is positive on >= 4/5 folds
AND, when a parameter is tuned on OOF, survives leave-one-fold-out selection. This screen runs
that second gate over the union of the incumbent family and the iteration-4 family, so the
adoption decision is made on a number rather than on the in-sample maximum.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "experiments")
from probe_referee import fold_vec, load, referee, selftest  # noqa: E402

D = load()
selftest(D)
Y, FOLDS = D["y"], D["folds"]
TAB, CNN = D["tab"], D["cnn"]
XY = np.column_stack([D["df"]["x"].to_numpy(), D["df"]["y"].to_numpy()])
TREE = cKDTree(XY)
DIST, IDX = TREE.query(XY, k=250)
BASE_VEC = fold_vec(Y, D["blend"].argmax(1), FOLDS)
BASE_MEAN = float(BASE_VEC.mean())
A, EPS = 12.0, 0.5


def sharpen(p, e):
    q = np.power(p, e)
    return q / q.sum(1, keepdims=True)


def f1c(y, pred, c):
    tp = int(np.sum((y == c) & (pred == c)))
    fp = int(np.sum((pred == c) & (y != c)))
    fn = int(np.sum((y == c) & (pred != c)))
    s = tp + fn
    if s == 0:
        return None
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / s
    return 2 * pr * rec / (pr + rec) if (pr + rec) else 0.0


SHARES = {}
for f in range(5):
    sh = np.full(5, 0.5)
    for c in range(5):
        rt = f1c(Y[FOLDS != f], TAB.argmax(1)[FOLDS != f], c)
        rc = f1c(Y[FOLDS != f], CNN.argmax(1)[FOLDS != f], c)
        if rt is None or rc is None:
            continue
        sh[c] = (rt + EPS * 0.5) / (rt + rc + EPS)
    SHARES[f] = sh

TS = sharpen(TAB, A)
NBM = {}


def nb_mean(M, k):
    key = (id(M), k)
    if key not in NBM:
        idx = IDX[:, :k]
        w = (FOLDS[idx] == FOLDS[:, None]).astype(float)
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        NBM[key] = np.einsum("ij,ijc->ic", w, M[idx])
    return NBM[key]


def pool_flat(TSm, CSm):
    P = np.zeros((len(Y), 5))
    for f in range(5):
        va = FOLDS == f
        sh = SHARES[f]
        p = 0.40 * TSm[va] * sh[None, :] + 0.60 * CSm[va] * (1 - sh)[None, :]
        P[va] = p / p.sum(1, keepdims=True)
    return P


def pool_then_smooth(k, alpha):
    P = pool_flat(TS, CNN)
    return (alpha * P + (1.0 - alpha) * nb_mean(P, k))


def smooth_then_pool(k, alpha):
    TSs = alpha * TS + (1.0 - alpha) * nb_mean(TS, k)
    CSs = alpha * CNN + (1.0 - alpha) * nb_mean(CNN, k)
    return pool_flat(TSs / TSs.sum(1, keepdims=True), CSs / CSs.sum(1, keepdims=True))


CAND = {}
for k in [80, 120, 160, 200]:
    for alpha in [0.6, 0.65, 0.7, 0.75]:
        CAND[("pool", k, alpha)] = pool_then_smooth(k, alpha).argmax(1)
        CAND[("comp", k, alpha)] = smooth_then_pool(k, alpha).argmax(1)

INC = CAND[("pool", 120, 0.7)]
INC_VEC = fold_vec(Y, INC, FOLDS)
INC_MEAN = float(INC_VEC.mean())
print(f"[incumbent iter3 = pool-then-smooth k=120,a=0.7] mean={INC_MEAN:.6f}")
print(f"[iter4 = smooth-components k=160,a=0.65]        "
      f"mean={fold_vec(Y, CAND[('comp', 160, 0.65)], FOLDS).mean():.6f}")

print("\n[ZZ] referee over the union grid (which mode/params would an honest selector pick?)")
r = referee(Y, FOLDS, [(f"{m}/k={k},a={a}", CAND[(m, k, a)]) for (m, k, a) in CAND])
print(f"  in_sample best = {r['in_sample_best_score']:.4f} ({r['in_sample_best_label']})")
print(f"  LOO honest     = {r['loo_honest']:.4f}  d_inc={r['loo_honest']-INC_MEAN:+.4f}  "
      f"d_base={r['loo_honest']-BASE_MEAN:+.4f}")
print(f"  chosen         = {r['loo_chosen']}")

print("\n[ZZ2] per-family referee")
for mode in ["pool", "comp"]:
    rr = referee(Y, FOLDS, [(f"{mode}/k={k},a={a}", CAND[(mode, k, a)])
                            for (m, k, a) in CAND if m == mode])
    print(f"  {mode:5s}: in_sample={rr['in_sample_best_score']:.4f} ({rr['in_sample_best_label']}) "
          f"LOO={rr['loo_honest']:.4f} d_inc={rr['loo_honest']-INC_MEAN:+.4f} chose={rr['loo_chosen']}")

with open("experiments/screen_iter4_gate.json", "w") as fh:
    json.dump({"incumbent": INC_MEAN, "union_loo": float(r["loo_honest"]),
               "union_chosen": r["loo_chosen"]}, fh, indent=2)
print("\nwrote experiments/screen_iter4_gate.json")
