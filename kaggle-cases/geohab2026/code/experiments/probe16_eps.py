#!/usr/bin/env python3
"""Screen 5 — pin down the reliability operator's shrinkage and get its honest LOO gate.

The aggressive end of the share formula (eps -> 0) reweights per-class using fold-specific
class mixes and costs fold 0; the mild end (eps -> infinity) degenerates to the published
scalar pool. This screen maps the whole sweep, checks a support-aware variant, and then runs
the referee over the eps grid so that the reported number accounts for eps being chosen.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "experiments")
from probe_referee import fold_vec, load, referee, selftest  # noqa: E402

D = load()
selftest(D)
Y, FOLDS = D["y"], D["folds"]
TAB, CNN = D["tab"], D["cnn"]
BASE_VEC = fold_vec(Y, D["blend"].argmax(1), FOLDS)
BASE_MEAN = float(BASE_VEC.mean())
print(f"[frozen] baseline = {BASE_MEAN:.6f}")


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
    return {"P": pr, "R": rec, "F1": 2 * pr * rec / (pr + rec) if (pr + rec) else 0.0, "s": s}


def shares_for(tr, eps, metric="F1", support_k=0.0):
    sh = np.full(5, 0.5)
    for c in range(5):
        rt = cls_prf(Y[tr], TAB.argmax(1)[tr], c)
        rc = cls_prf(Y[tr], CNN.argmax(1)[tr], c)
        if rt is None or rc is None:
            continue
        s = (rt[metric] + eps * 0.5) / (rt[metric] + rc[metric] + eps)
        if support_k > 0.0:
            n = min(rt["s"], rc["s"])
            s = 0.5 + (s - 0.5) * (n / (n + support_k))
        sh[c] = s
    return sh


def evaluate(shares_fn):
    pred = np.zeros(len(Y), dtype=int)
    for f in range(5):
        va = FOLDS == f
        sh = shares_fn(~va)
        p = 0.40 * (TS6 * sh[np.newaxis, :]) + 0.60 * (CNN * (1.0 - sh)[np.newaxis, :])
        pred[va] = p[va].argmax(1)
    return pred


def show(tag, pred):
    fv = fold_vec(Y, pred, FOLDS)
    d = float(fv.mean() - BASE_MEAN)
    pf = np.round(fv - BASE_VEC, 4)
    npos = int((fv - BASE_VEC >= -1e-12).sum())
    print(f"  {tag:40s} mean={fv.mean():.4f} d={d:+.4f} pos={npos}/5 per-fold={pf.tolist()}")
    return {"tag": tag, "mean": float(fv.mean()), "delta": d, "n_pos": npos,
            "per_fold_delta": pf.tolist()}


out = []
print("\n[T] eps sweep, metric=F1 (mild -> aggressive)")
cands_by_eps = {}
for eps in [0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
    pred = evaluate(lambda tr, e=eps: shares_for(tr, e, "F1"))
    cands_by_eps[eps] = pred
    out.append(show(f"T/F1 eps={eps}", pred))

print("\n[T2] eps sweep, metric=P")
for eps in [0.5, 1.0, 2.0, 3.0, 5.0]:
    out.append(show(f"T2/P eps={eps}", evaluate(lambda tr, e=eps: shares_for(tr, e, "P"))))

print("\n[T3] support-aware shrinkage, metric=F1 (support_k)")
for k in [50, 150, 400]:
    out.append(show(f"T3/F1 eps=0.05 k={k}",
                    evaluate(lambda tr, k=k: shares_for(tr, 0.05, "F1", support_k=k))))

print("\n[T4] shares at the two candidate operating points")
for eps in [1.0, 2.0]:
    for f in range(5):
        sh = shares_for(FOLDS != f, eps, "F1")
        print(f"      eps={eps} fold {f} share={np.round(sh, 3).tolist()}")

print("\n[U] referee over the eps grid (treats eps as a selected knob -> honest gate)")
cands = [(f"eps={e}", cands_by_eps[e]) for e in cands_by_eps]
r = referee(Y, FOLDS, cands)
print(f"  in_sample={r['in_sample_best_score']:.4f} ({r['in_sample_best_label']}) "
      f"LOO={r['loo_honest']:.4f} d={r['loo_honest']-BASE_MEAN:+.4f} chose={r['loo_chosen']} "
      f"per-fold={np.round(np.array(r['loo_per_fold'])-BASE_VEC,4).tolist()}")
out.append({"tag": "U/loo-over-eps", "loo": float(r["loo_honest"]),
            "delta": float(r["loo_honest"] - BASE_MEAN), "chosen": r["loo_chosen"]})

with open("experiments/screen_iter1f.json", "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter1f.json")
