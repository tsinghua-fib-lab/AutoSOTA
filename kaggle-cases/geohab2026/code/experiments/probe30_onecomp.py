#!/usr/bin/env python3
"""Screen 18 — smooth only ONE component, then pool.

The tabular branch is built from multi-scale focal statistics and may be the spatially coherent
one; the CNN is a patch classifier whose errors may be less patch-structured. Smoothing only one
side is a one-choice change, and it is the last structural variant of the FUSE-03 axis.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "experiments")
from probe_referee import fold_vec, load, selftest  # noqa: E402

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


def nb_mean(M, k):
    idx = IDX[:, :k]
    w = (FOLDS[idx] == FOLDS[:, None]).astype(np.float64)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return np.einsum("ij,ijc->ic", w, M[idx])


def smooth_M(M, k, alpha):
    Q = alpha * M + (1.0 - alpha) * nb_mean(M, k)
    return Q / Q.sum(1, keepdims=True)


def pool_from(TS, CS):
    P = np.zeros((len(Y), 5))
    for f in range(5):
        va = FOLDS == f
        sh = np.full(5, 0.5)
        for c in range(5):
            rt = cls_prf(Y[FOLDS != f], TAB.argmax(1)[FOLDS != f], c)
            rc = cls_prf(Y[FOLDS != f], CNN.argmax(1)[FOLDS != f], c)
            if rt is None or rc is None:
                continue
            sh[c] = (rt["F1"] + EPS * 0.5) / (rt["F1"] + rc["F1"] + EPS)
        p = 0.40 * TS[va] * sh[None, :] + 0.60 * CS[va] * (1 - sh)[None, :]
        P[va] = p / p.sum(1, keepdims=True)
    return P


TS = sharpen(TAB, A)
P0 = pool_from(TS, CNN)
INC_VEC = fold_vec(Y, smooth_M(P0, 120, 0.7).argmax(1), FOLDS)
INC_MEAN = float(INC_VEC.mean())
print(f"[incumbent iter3] mean={INC_MEAN:.6f}")
out = []


def show(tag, pred):
    fv = fold_vec(Y, pred, FOLDS)
    pf = np.round(fv - INC_VEC, 4)
    print(f"  {tag:40s} mean={fv.mean():.4f} d_inc={fv.mean()-INC_MEAN:+.4f} "
          f"d_base={fv.mean()-BASE_MEAN:+.4f} pos={int((fv-INC_VEC >= -1e-12).sum())}/5 {pf.tolist()}")
    return {"tag": tag, "mean": float(fv.mean()), "delta_inc": float(fv.mean() - INC_MEAN),
            "n_pos": int((fv - INC_VEC >= -1e-12).sum()), "per_fold_delta": pf.tolist()}


print("\n[WW] pool-then-smooth variants (current operator family, for reference)")
out.append(show("WW/pool-then-smooth k=120,a=0.7 (incumbent)", INC_VEC if False else smooth_M(P0, 120, 0.7).argmax(1)))

print("\n[XX] smooth ONE component then pool")
for comp in ["tab", "cnn"]:
    for k in [120, 200]:
        for alpha in [0.6, 0.7]:
            if comp == "tab":
                P = pool_from(smooth_M(TS, k, alpha), CNN)
            else:
                P = pool_from(TS, smooth_M(CNN, k, alpha))
            out.append(show(f"XX/smooth-{comp} k={k},a={alpha}", P.argmax(1)))

print("\n[YY] smooth both components then pool (T1), for a final head-to-head")
for k in [160, 200]:
    for alpha in [0.55, 0.6, 0.65]:
        out.append(show(f"YY/smooth-both k={k},a={alpha}",
                        pool_from(smooth_M(TS, k, alpha), smooth_M(CNN, k, alpha)).argmax(1)))

with open("experiments/screen_iter4_onecomp.json", "w") as fh:
    json.dump({"incumbent": INC_MEAN, "baseline": BASE_MEAN, "results": out}, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter4_onecomp.json")
