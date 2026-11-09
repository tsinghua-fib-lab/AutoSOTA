#!/usr/bin/env python3
"""Screen 17 — does the ORDER of pool and smooth matter, and does a second pool pass help?

The current operator is pool-then-smooth. Two structurally different, zero-free-parameter
alternatives:
  T1 smooth-then-pool : smooth each component's posterior on its own, then pool. Each component
                        has its own spatial coherence, so averaging them before smoothing mixes
                        two different neighbourhood statistics.
  T2 re-pool          : pool -> smooth -> recompute the per-class reliability shares from the
                        SMOOTHED posteriors on the other folds -> pool again.
Also re-verifies the k=200 / alpha=0.65 point under the current temperature a=12.
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
A = 12.0
EPS = 0.5


def sharpen(p, e):
    q = np.power(p, e)
    return q / q.sum(1, keepdims=True)


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


def shares_from(tab_p, cnn_p, mask):
    sh = np.full(5, 0.5)
    for c in range(5):
        rt = cls_prf(Y[mask], tab_p.argmax(1)[mask], c)
        rc = cls_prf(Y[mask], cnn_p.argmax(1)[mask], c)
        if rt is None or rc is None:
            continue
        sh[c] = (rt["F1"] + EPS * 0.5) / (rt["F1"] + rc["F1"] + EPS)
    return sh


def nb_mean(M, k):
    idx = IDX[:, :k]
    w = (FOLDS[idx] == FOLDS[:, None]).astype(np.float64)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return np.einsum("ij,ijc->ic", w, M[idx])


def smooth_M(M, k, alpha):
    Q = alpha * M + (1.0 - alpha) * nb_mean(M, k)
    return Q / Q.sum(1, keepdims=True)


def pool_from(TS, CS, eps=EPS):
    P = np.zeros((len(Y), 5))
    for f in range(5):
        va = FOLDS == f
        sh = shares_from(TS, CS, FOLDS != f)
        p = 0.40 * TS[va] * sh[None, :] + 0.60 * CS[va] * (1 - sh)[None, :]
        P[va] = p / p.sum(1, keepdims=True)
    return P


TS = sharpen(TAB, A)
CS = CNN
P0 = pool_from(TS, CS)
INC_VEC = fold_vec(Y, smooth_M(P0, 120, 0.7).argmax(1), FOLDS)
INC_MEAN = float(INC_VEC.mean())
print(f"[incumbent iter3] mean={INC_MEAN:.6f} per-fold={np.round(INC_VEC,4).tolist()}")
out = []


def show(tag, pred):
    fv = fold_vec(Y, pred, FOLDS)
    pf = np.round(fv - INC_VEC, 4)
    print(f"  {tag:40s} mean={fv.mean():.4f} d_inc={fv.mean()-INC_MEAN:+.4f} "
          f"d_base={fv.mean()-BASE_MEAN:+.4f} pos={int((fv-INC_VEC >= -1e-12).sum())}/5 {pf.tolist()}")
    return {"tag": tag, "mean": float(fv.mean()), "delta_inc": float(fv.mean() - INC_MEAN),
            "n_pos": int((fv - INC_VEC >= -1e-12).sum()), "per_fold_delta": pf.tolist()}


print("\n[TT] T1 smooth-then-pool (smooth each component separately, then pool)")
for k in [120, 160, 200]:
    for alpha in [0.6, 0.7, 0.8]:
        TSs = smooth_M(TS, k, alpha)
        CSs = smooth_M(CS, k, alpha)
        out.append(show(f"TT/k={k},a={alpha}", pool_from(TSs, CSs).argmax(1)))

print("\n[UU] T2 re-pool after smoothing (shares recomputed from smoothed posteriors)")
for k in [120, 160]:
    for alpha in [0.6, 0.7]:
        Ps = smooth_M(P0, k, alpha)
        # smoothed component surrogates: scale the sharpened components by the smoothed pool
        # (the smoothed pool is what we have; reuse it as the tab surrogate is not valid, so
        # instead recompute shares from the smoothed POOL vs the raw CNN argmax)
        P2 = np.zeros((len(Y), 5))
        for f in range(5):
            va = FOLDS == f
            sh = shares_from(Ps, CS, FOLDS != f)
            p = 0.40 * TS[va] * sh[None, :] + 0.60 * CS[va] * (1 - sh)[None, :]
            P2[va] = p / p.sum(1, keepdims=True)
        out.append(show(f"UU/re-pool then smooth k={k},a={alpha}", smooth_M(P2, k, alpha).argmax(1)))

print("\n[VV] k/alpha re-verified at a=12 (narrow grid)")
CAND = {}
for k in [100, 120, 160, 200]:
    for alpha in [0.65, 0.70, 0.75]:
        CAND[(k, alpha)] = smooth_M(P0, k, alpha).argmax(1)
        fv = fold_vec(Y, CAND[(k, alpha)], FOLDS)
        print(f"  k={k:<4} a={alpha} mean={fv.mean():.4f} d_inc={fv.mean()-INC_MEAN:+.4f} "
              f"pos={int((fv-INC_VEC >= -1e-12).sum())}/5 {np.round(fv-INC_VEC,4).tolist()}")
r = referee(Y, FOLDS, [(f"k={k},a={a}", CAND[(k, a)]) for (k, a) in CAND])
print(f"  LOO={r['loo_honest']:.4f} d_base={r['loo_honest']-BASE_MEAN:+.4f} chose={r['loo_chosen']}")
out.append({"tag": "VV/loo-narrow", "loo": float(r["loo_honest"]), "chosen": r["loo_chosen"]})

with open("experiments/screen_iter4_order.json", "w") as fh:
    json.dump({"incumbent": INC_MEAN, "baseline": BASE_MEAN, "results": out}, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter4_order.json")
