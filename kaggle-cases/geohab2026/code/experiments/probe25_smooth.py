#!/usr/bin/env python3
"""Screen 13 — FUSE-SMOOTH re-screened on top of the FUSE-02 operator.

Smoothing each point's pooled posterior with the mean posterior of its same-fold neighbours is
the standard post-classification filter in benthic mapping. On the two earlier operators it
showed a consistent *in-sample* gain near +0.006 that did not survive joint (k, alpha) LOO
selection, i.e. the effect is real but the grid selection is noisy. Here the grid is mapped on
the current operator, and the honest question is whether a *pre-declared* (k, alpha) helps.

Neighbour set is restricted to the same fold, so no held-out label can enter through a model
trained on it. Uses predicted probabilities only, never labels.
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


# FUSE-02 posterior + prediction, per fold
P = np.zeros((len(Y), 5))
for f in range(5):
    va = FOLDS == f
    sh = shares_for(FOLDS != f)
    p = 0.40 * TS6[va] * sh[None, :] + 0.60 * CNN[va] * (1 - sh)[None, :]
    P[va] = p / p.sum(1, keepdims=True)
INC = P.argmax(1)
INC_VEC = fold_vec(Y, INC, FOLDS)
INC_MEAN = float(INC_VEC.mean())
print(f"[incumbent FUSE-02] mean={INC_MEAN:.6f} per-fold={np.round(INC_VEC,4).tolist()}")


def smooth_pred(k, alpha):
    _, idx = TREE.query(XY, k=k)
    same = FOLDS[idx] == FOLDS[:, None]
    w = same.astype(np.float64)
    w /= w.sum(1, keepdims=True)
    nb = np.einsum("ij,ijc->ic", w, P[idx])
    return (alpha * P + (1.0 - alpha) * nb).argmax(1)


out = []
print("\n[FF] fixed (k, alpha) on top of FUSE-02")
CAND = {}
for k in [20, 40, 80, 120, 160]:
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = smooth_pred(k, alpha)
        CAND[(k, alpha)] = pred
        fv = fold_vec(Y, pred, FOLDS)
        pf = np.round(fv - INC_VEC, 4)
        print(f"  k={k:<4} alpha={alpha}  mean={fv.mean():.4f} d_inc={fv.mean()-INC_MEAN:+.4f} "
              f"d_base={fv.mean()-BASE_MEAN:+.4f} pos={int((fv-INC_VEC >= -1e-12).sum())}/5 {pf.tolist()}")
        out.append({"tag": f"FF/k={k},a={alpha}", "mean": float(fv.mean()),
                    "delta_inc": float(fv.mean() - INC_MEAN),
                    "delta_base": float(fv.mean() - BASE_MEAN),
                    "n_pos": int((fv - INC_VEC >= -1e-12).sum()), "per_fold_delta": pf.tolist()})

print("\n[GG] referee over the (k, alpha) grid -> honest LOO for the smoothing idea")
r = referee(Y, FOLDS, [(f"k={k},a={a}", CAND[(k, a)]) for (k, a) in CAND])
print(f"  in_sample best = {r['in_sample_best_score']:.4f} ({r['in_sample_best_label']})")
print(f"  LOO honest     = {r['loo_honest']:.4f}  d_inc={r['loo_honest']-INC_MEAN:+.4f}  "
      f"d_base={r['loo_honest']-BASE_MEAN:+.4f}  chose={r['loo_chosen']}")
out.append({"tag": "GG/loo-grid", "loo": float(r["loo_honest"]),
            "delta_inc": float(r["loo_honest"] - INC_MEAN), "chosen": r["loo_chosen"]})

with open("experiments/screen_iter5_smooth.json", "w") as fh:
    json.dump({"incumbent": INC_MEAN, "baseline": BASE_MEAN, "results": out}, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter5_smooth.json")
