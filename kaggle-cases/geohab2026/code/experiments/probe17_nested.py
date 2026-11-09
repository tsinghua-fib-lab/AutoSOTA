#!/usr/bin/env python3
"""Screen 6 — honest nested gate for the fold-adaptive reliability operator.

Why the plain referee is not enough here: `probe_referee.referee` chooses the parameter for
held-out fold f by scoring each candidate on folds != f. For a *fold-adaptive* operator those
rows' predictions were themselves produced with shares derived from a mask that includes fold f,
so the selection score is mildly contaminated. The honest protocol nests one level deeper:

    for held-out fold f:
        for each candidate eps e:
            score(e) = mean over g != f of  F1( fold g | shares derived from folds not in {f, g} )
        e* = argmax score(e)                       # never sees fold f, directly or indirectly
        prediction for fold f = operator with shares derived from folds != f, using e*

Also reports the a-priori-fixed variants, which need no selection at all.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "experiments")
from probe_referee import fold_vec, load, selftest, wf1_fast  # noqa: E402

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
    return {"F1": 2 * pr * rec / (pr + rec) if (pr + rec) else 0.0, "s": s}


def shares_for(mask, eps, metric="F1"):
    sh = np.full(5, 0.5)
    for c in range(5):
        rt = cls_prf(Y[mask], TAB.argmax(1)[mask], c)
        rc = cls_prf(Y[mask], CNN.argmax(1)[mask], c)
        if rt is None or rc is None:
            continue
        sh[c] = (rt[metric] + eps * 0.5) / (rt[metric] + rc[metric] + eps)
    return sh


def pred_for(shares, rows):
    """operator applied to a subset of rows given a share vector."""
    sh = shares
    p = 0.40 * (TS6[rows] * sh[np.newaxis, :]) + 0.60 * (CNN[rows] * (1.0 - sh)[np.newaxis, :])
    return p.argmax(1)


EPS_GRID = [0.05, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
rows_all = np.arange(len(Y))

print("=== honest nested LOO for eps selection ===")
nested_per_fold, chosen = [], []
for f in range(5):
    inner_scores = []
    for e in EPS_GRID:
        sc = []
        for g in range(5):
            if g == f:
                continue
            mask = (FOLDS != f) & (FOLDS != g)
            rows = FOLDS == g
            sc.append(wf1_fast(Y[rows], pred_for(shares_for(mask, e), rows)))
        inner_scores.append(float(np.mean(sc)))
    best = int(np.argmax(inner_scores))
    eps_star = EPS_GRID[best]
    rows = FOLDS == f
    p = pred_for(shares_for(FOLDS != f, eps_star), rows)
    nested_per_fold.append(wf1_fast(Y[rows], p))
    chosen.append(eps_star)
    print(f"  fold {f}: inner scores {[round(s,4) for s in inner_scores]} -> eps*={eps_star} "
          f"F1_fold={nested_per_fold[-1]:.4f} (baseline {BASE_VEC[f]:.4f})")

nested = float(np.mean(nested_per_fold))
print(f"\n  NESTED-LOO fold-mean = {nested:.4f}  d={nested-BASE_MEAN:+.4f}  chosen={chosen}")
print(f"  per-fold deltas      = {np.round(np.array(nested_per_fold)-BASE_VEC,4).tolist()}")

print("\n=== fixed eps (no selection at all) ===")
out = []
for e in EPS_GRID:
    per = [wf1_fast(Y[FOLDS == f], pred_for(shares_for(FOLDS != f, e), FOLDS == f)) for f in range(5)]
    per = np.array(per)
    d = float(per.mean() - BASE_MEAN)
    npos = int((per - BASE_VEC >= -1e-12).sum())
    print(f"  eps={e:<5} fold-mean={per.mean():.6f} d={d:+.4f} pos={npos}/5 "
          f"per-fold={np.round(per-BASE_VEC,4).tolist()}")
    out.append({"eps": e, "mean": float(per.mean()), "delta": d, "n_pos": npos,
                "per_fold_delta": np.round(per - BASE_VEC, 4).tolist()})

print("\n=== chosen operating point: full-precision shares and result ===")
for e in [0.5]:
    for f in range(5):
        print(f"  eps={e} fold {f} share={np.round(shares_for(FOLDS != f, e), 4).tolist()}")
    per = np.array([wf1_fast(Y[FOLDS == f], pred_for(shares_for(FOLDS != f, e), FOLDS == f)) for f in range(5)])
    print(f"  eps={e} fold-mean={per.mean():.16f} per-fold={np.round(per,6).tolist()}")

with open("experiments/screen_iter1g.json", "w") as fh:
    json.dump({"nested_loo": nested, "nested_chosen": chosen,
               "nested_per_fold": nested_per_fold, "fixed": out}, fh, indent=2)
print("\nwrote experiments/screen_iter1g.json")
