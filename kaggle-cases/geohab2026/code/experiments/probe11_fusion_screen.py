#!/usr/bin/env python3
"""Iteration-1 screen: full fusion-operator search on the frozen round-2 matrices (REF-01).

Every candidate is scored twice through experiments/probe_referee.py:
  in_sample : best parameter on all 6256 OOF points (optimistic bound, never adopted)
  loo       : honest leave-one-fold-out selection (the only number that may be reported)
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "experiments")
from probe_referee import CLASSES, fold_mean, fold_vec, load, referee, selftest  # noqa: E402

BASELINE = load()
selftest(BASELINE)
Y, FOLDS = BASELINE["y"], BASELINE["folds"]
TAB, CNN = BASELINE["tab"], BASELINE["cnn"]
BASE_PRED = BASELINE["blend"].argmax(1)
BASE_MEAN = fold_mean(Y, BASE_PRED, FOLDS)
BASE_VEC = fold_vec(Y, BASE_PRED, FOLDS)
print(f"\n[frozen] baseline fold-mean = {BASE_MEAN:.6f}  per-fold = {np.round(BASE_VEC,4).tolist()}")

XY = np.column_stack([BASELINE["df"]["x"].to_numpy(), BASELINE["df"]["y"].to_numpy()])
TREE = cKDTree(XY)


def report(tag, res, extra=""):
    d = res["loo_honest"] - BASE_MEAN
    pf = np.array(res["loo_per_fold"]) - BASE_VEC
    n_pos = int((pf >= -1e-12).sum())
    print(f"  {tag:34s} in_sample={res['in_sample_best_score']:.4f} "
          f"({str(res['in_sample_best_label']):22s}) LOO={res['loo_honest']:.4f} d={d:+.4f} "
          f"pos={n_pos}/5 chose={res['loo_chosen']} {extra}")
    return {"tag": tag, "in_sample": res["in_sample_best_score"], "loo": res["loo_honest"],
            "delta": d, "n_pos": n_pos, "chosen": res["loo_chosen"], "extra": extra}


def sharpen(p, e):
    if e == 1.0:
        return p
    q = np.power(p, e)
    return q / q.sum(1, keepdims=True)


def pool(tab_m=None, cnn_m=None, w=0.40, a=6.0, b=1.0, tab=None, cnn=None):
    tab = TAB if tab is None else tab
    cnn = CNN if cnn is None else cnn
    ts, cs = sharpen(tab, a), sharpen(cnn, b)
    if tab_m is not None:
        ts = ts * np.asarray(tab_m)[None, :]
    if cnn_m is not None:
        cs = cs * np.asarray(cnn_m)[None, :]
    p = w * ts + (1.0 - w) * cs
    return p / p.sum(1, keepdims=True)


def smooth(p, k, alpha):
    _, idx = TREE.query(XY, k=k)
    same = FOLDS[idx] == FOLDS[:, None]
    wt = same.astype(np.float64)
    wt /= wt.sum(1, keepdims=True)
    nb = np.einsum("ij,ijc->ic", wt, p[idx])
    return alpha * p + (1.0 - alpha) * nb


results = []

print("\n[A] per-class CNN multiplier, one class at a time (grid 0.6..2.0 step 0.05)")
GRID = [round(v, 2) for v in np.arange(0.6, 2.01, 0.05)]
for ci, cname in enumerate(CLASSES):
    cands = []
    for v in GRID:
        m = np.ones(5)
        m[ci] = v
        cands.append((f"{cname}={v}", pool(cnn_m=m).argmax(1)))
    results.append(report(f"A/cnnmult/{cname}", referee(Y, FOLDS, cands)))

print("\n[A2] per-class TAB multiplier, one class at a time")
for ci, cname in enumerate(CLASSES):
    cands = []
    for v in GRID:
        m = np.ones(5)
        m[ci] = v
        cands.append((f"{cname}={v}", pool(tab_m=m).argmax(1)))
    results.append(report(f"A2/tabmult/{cname}", referee(Y, FOLDS, cands)))

print("\n[B] joint CNN/SGAM multiplier sweep at fixed 1.2/1.4/1.6 (the library hypothesis)")
for v in [1.1, 1.2, 1.3, 1.4, 1.6]:
    m = np.ones(5)
    m[3] = v
    p = pool(cnn_m=m).argmax(1)
    pf = fold_vec(Y, p, FOLDS)
    print(f"  B/SGAM-cnn={v:<21} fold-mean={pf.mean():.4f} d={pf.mean()-BASE_MEAN:+.4f} "
          f"pos={int(((pf-BASE_VEC)>=-1e-12).sum())}/5 per-fold={np.round(pf-BASE_VEC,4).tolist()}")

print("\n[C] pairs of CNN per-class multipliers (FMAT x SGAM)")
cands = []
for vf in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]:
    for vs in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        m = np.ones(5)
        m[1], m[3] = vf, vs
        cands.append((f"FMAT={vf},SGAM={vs}", pool(cnn_m=m).argmax(1)))
results.append(report("C/cnnpair(FMAT,SGAM)", referee(Y, FOLDS, cands)))

print("\n[D] triple: CNN FMAT mult x TAB SGAM mult")
cands = []
for vf in [0.55, 0.6, 0.65, 0.7, 0.8, 1.0]:
    for vs in [0.5, 0.6, 0.7, 0.8, 1.0]:
        mt = np.ones(5)
        mt[3] = vs
        mc = np.ones(5)
        mc[1] = vf
        cands.append((f"cnnFMAT={vf},tabSGAM={vs}", pool(tab_m=mt, cnn_m=mc).argmax(1)))
results.append(report("D/cnnFMAT+tabSGAM", referee(Y, FOLDS, cands)))

print("\n[E] same-fold spatial smoothing (k, alpha)")
SM = {}
cands = []
for k in [10, 20, 40, 80, 120, 160]:
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        SM[(k, alpha)] = smooth(BASELINE["blend"], k, alpha)
        cands.append((f"k={k},a={alpha}", SM[(k, alpha)].argmax(1)))
results.append(report("E/smooth(k,alpha)", referee(Y, FOLDS, cands)))

print("\n[F] smoothing on top of the best class-multiplier pool")
cands = []
for vf in [0.6, 0.65, 0.7, 1.0]:
    for vs in [1.0, 1.2, 1.4]:
        mc = np.ones(5)
        mc[1], mc[3] = vf, vs
        base = pool(cnn_m=mc)
        for k in [40, 80, 160]:
            for alpha in [0.7, 0.8, 0.9]:
                cands.append((f"FMAT={vf},SGAM={vs},k={k},a={alpha}", smooth(base, k, alpha).argmax(1)))
results.append(report("F/mult+smooth", referee(Y, FOLDS, cands)))

with open("experiments/screen_iter1.json", "w") as fh:
    json.dump({"baseline": BASE_MEAN, "results": results}, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter1.json")
