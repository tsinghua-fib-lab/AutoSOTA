#!/usr/bin/env python3
"""Screen 14 — refine FUSE-03 and re-check the FUSE-02 knobs underneath it.

Smoothing changed the operator, so the interacting parameters are re-screened:
  HH  wider k / lower alpha (does more patch pooling keep helping?)
  II  distance-weighted neighbours and two-pass smoothing (same information, better estimator)
  JJ  the FUSE-02 reliability eps and measure, re-screened under smoothing
  KK  the fixed per-class multipliers that failed before, re-tested under smoothing

Every candidate is fold-honest: shares come from the other four folds, and the neighbour set is
restricted to the same fold (a cross-fold neighbour's posterior was produced by a model trained
on the target point's own fold, which would leak that point's label).
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
NMAX = 500
DIST, IDX = TREE.query(XY, k=NMAX)
BASE_VEC = fold_vec(Y, D["blend"].argmax(1), FOLDS)
BASE_MEAN = float(BASE_VEC.mean())


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


def shares_for(mask, eps=0.5, metric="F1", tab=None, cnn=None):
    tab = TAB if tab is None else tab
    cnn = CNN if cnn is None else cnn
    sh = np.full(5, 0.5)
    for c in range(5):
        rt = cls_prf(Y[mask], tab.argmax(1)[mask], c)
        rc = cls_prf(Y[mask], cnn.argmax(1)[mask], c)
        if rt is None or rc is None:
            continue
        sh[c] = (rt[metric] + eps * 0.5) / (rt[metric] + rc[metric] + eps)
    return sh


def pool_oof(tab=None, cnn=None, eps=0.5, atab=6.0, cnmult=None):
    tab = TAB if tab is None else tab
    cnn = CNN if cnn is None else cnn
    ts = sharpen(tab, atab)
    P = np.zeros((len(Y), 5))
    for f in range(5):
        va = FOLDS == f
        sh = shares_for(FOLDS != f, eps, tab=tab, cnn=cnn)
        if cnmult is not None:
            sh = sh  # tab share unchanged; the multiplier applies below
            cs = cnn[va] * cnmult[None, :]
        else:
            cs = cnn[va]
        p = 0.40 * ts[va] * sh[None, :] + 0.60 * cs * (1 - sh)[None, :]
        P[va] = p / p.sum(1, keepdims=True)
    return P


def smooth(P, k, alpha, weighted=False, passes=1):
    Q = P
    for _ in range(passes):
        idx = IDX[:, :k]
        d = DIST[:, :k]
        same = FOLDS[idx] == FOLDS[:, None]
        if weighted:
            # Gaussian kernel on distance, scaled by the k-th neighbour distance of the point
            sigma = np.maximum(d[:, -1:], 1e-6) / 2.0
            w = np.exp(-0.5 * (d / sigma) ** 2) * same
        else:
            w = same.astype(np.float64)
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
        Q = alpha * Q + (1.0 - alpha) * np.einsum("ij,ijc->ic", w, Q[idx])
        Q = Q / Q.sum(1, keepdims=True)
    return Q


P0 = pool_oof()
INC_VEC = fold_vec(Y, smooth(P0, 120, 0.7).argmax(1), FOLDS)
INC_MEAN = float(INC_VEC.mean())
print(f"[incumbent iter2 = FUSE-02 + smooth k=120,alpha=0.7] mean={INC_MEAN:.6f} "
      f"per-fold={np.round(INC_VEC,4).tolist()}")

out = []


def show(tag, pred, ref_vec=None, ref_mean=None):
    ref_vec = INC_VEC if ref_vec is None else ref_vec
    ref_mean = INC_MEAN if ref_mean is None else ref_mean
    fv = fold_vec(Y, pred, FOLDS)
    pf = np.round(fv - ref_vec, 4)
    print(f"  {tag:34s} mean={fv.mean():.4f} d_inc={fv.mean()-ref_mean:+.4f} "
          f"d_base={fv.mean()-BASE_MEAN:+.4f} pos={int((fv-ref_vec >= -1e-12).sum())}/5 {pf.tolist()}")
    return {"tag": tag, "mean": float(fv.mean()), "delta_inc": float(fv.mean() - ref_mean),
            "n_pos": int((fv - ref_vec >= -1e-12).sum()), "per_fold_delta": pf.tolist()}


print("\n[HH] wider k / alpha grid (uniform neighbour mean)")
CAND = {}
for k in [120, 160, 200, 250, 300, 400]:
    for alpha in [0.4, 0.5, 0.6, 0.7, 0.8]:
        pr = smooth(P0, k, alpha)
        CAND[(k, alpha)] = pr.argmax(1)
for (k, alpha), pr in CAND.items():
    fv = fold_vec(Y, pr, FOLDS)
    out.append({"tag": f"HH/k={k},a={alpha}", "mean": float(fv.mean()),
                "delta_inc": float(fv.mean() - INC_MEAN),
                "n_pos": int((fv - INC_VEC >= -1e-12).sum()),
                "per_fold_delta": np.round(fv - INC_VEC, 4).tolist()})
    print(f"  k={k:<4} a={alpha}  mean={fv.mean():.4f} d_inc={fv.mean()-INC_MEAN:+.4f} "
          f"pos={int((fv-INC_VEC >= -1e-12).sum())}/5 {np.round(fv-INC_VEC,4).tolist()}")

print("\n[HH2] referee over the (k, alpha) grid")
r = referee(Y, FOLDS, [(f"k={k},a={a}", CAND[(k, a)]) for (k, a) in CAND])
print(f"  in_sample best={r['in_sample_best_score']:.4f} ({r['in_sample_best_label']}) "
      f"LOO={r['loo_honest']:.4f} d_base={r['loo_honest']-BASE_MEAN:+.4f} chose={r['loo_chosen']}")
out.append({"tag": "HH2/loo-grid", "loo": float(r["loo_honest"]), "chosen": r["loo_chosen"]})

print("\n[II] distance-weighted neighbours and multi-pass smoothing")
for k in [120, 200]:
    for alpha in [0.5, 0.6, 0.7]:
        out.append(show(f"II/gauss k={k},a={alpha}", smooth(P0, k, alpha, weighted=True).argmax(1)))
for k in [120]:
    for alpha in [0.7, 0.8]:
        out.append(show(f"II/2pass k={k},a={alpha}", smooth(P0, k, alpha, passes=2).argmax(1)))

print("\n[JJ] FUSE-02 reliability eps re-screened UNDER smoothing (k=120, alpha=0.7)")
for eps in [0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0]:
    P = pool_oof(eps=eps)
    out.append(show(f"JJ/eps={eps}", smooth(P, 120, 0.7).argmax(1)))

print("\n[KK] fixed per-class CNN multipliers re-tested UNDER smoothing")
for tag, cm in [("SGAM=1.2", [1, 1, 1, 1.2, 1]), ("FMAT=0.65", [1, 0.65, 1, 1, 1]),
                ("FMAT=.65,SGAM=1.2", [1, 0.65, 1, 1.2, 1])]:
    P = pool_oof(cnmult=np.array(cm, float))
    out.append(show(f"KK/{tag}", smooth(P, 120, 0.7).argmax(1)))

with open("experiments/screen_iter2_smooth2.json", "w") as fh:
    json.dump({"incumbent": INC_MEAN, "baseline": BASE_MEAN, "results": out}, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter2_smooth2.json")
