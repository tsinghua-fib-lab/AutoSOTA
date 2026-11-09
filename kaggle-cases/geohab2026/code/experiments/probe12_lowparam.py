#!/usr/bin/env python3
"""Screen 2 — low-/zero-degree-of-freedom fusion operators (the honest-selection escape
from the LOO-selection-noise problem measured in screen 1, where every 2-D grid lost).

Families:
  G. global CNN temperature b (1 param) and joint (a,b) grid
  H. per-class CNN multiplier, fine sweep on FMAT only, with per-fold deltas
  I. ZERO-free-parameter per-class reliability weighting: for each fold f the per-class
     component weights are computed from the OTHER four folds only (classic stacking with a
     fixed formula, no tuned parameter at all -> immune to LOO selection noise).
  J. same, but reliability = per-class precision.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "experiments")
from probe_referee import CLASSES, fold_mean, fold_vec, load, referee, selftest, wf1_fast  # noqa: E402

D = load()
selftest(D)
Y, FOLDS = D["y"], D["folds"]
TAB, CNN = D["tab"], D["cnn"]
BASE_MEAN = fold_mean(Y, D["blend"].argmax(1), FOLDS)
BASE_VEC = fold_vec(Y, D["blend"].argmax(1), FOLDS)
print(f"[frozen] baseline = {BASE_MEAN:.6f} per-fold {np.round(BASE_VEC,4).tolist()}")


def sharpen(p, e):
    if e == 1.0:
        return p
    q = np.power(p, e)
    return q / q.sum(1, keepdims=True)


def pool(tab_m=None, cnn_m=None, w=0.40, a=6.0, b=1.0):
    ts, cs = sharpen(TAB, a), sharpen(CNN, b)
    if tab_m is not None:
        ts = ts * np.asarray(tab_m)[None, :]
    if cnn_m is not None:
        cs = cs * np.asarray(cnn_m)[None, :]
    p = w * ts + (1.0 - w) * cs
    return p / p.sum(1, keepdims=True)


def show(tag, pred):
    pf = fold_vec(Y, pred, FOLDS)
    d = pf.mean() - BASE_MEAN
    print(f"  {tag:30s} mean={pf.mean():.4f} d={d:+.4f} pos={int(((pf-BASE_VEC)>=-1e-12).sum())}/5 "
          f"per-fold={np.round(pf-BASE_VEC,4).tolist()}")
    return {"tag": tag, "mean": float(pf.mean()), "delta": float(d), "per_fold_delta": np.round(pf - BASE_VEC, 4).tolist()}


out = []

print("\n[G] global CNN temperature b (fixed values, no selection)")
for b in [0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]:
    out.append(show(f"G/cnn_temp b={b}", pool(b=b).argmax(1)))

print("\n[G2] LOO-selected global CNN temperature")
cands = [(f"b={b}", pool(b=b).argmax(1)) for b in [0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]]
r = referee(Y, FOLDS, cands)
print(f"  in_sample={r['in_sample_best_score']:.4f} ({r['in_sample_best_label']}) LOO={r['loo_honest']:.4f} "
      f"d={r['loo_honest']-BASE_MEAN:+.4f} chose={r['loo_chosen']}")
out.append({"tag": "G2/loo-cnn-temp", "loo": r["loo_honest"], "delta": r["loo_honest"] - BASE_MEAN})

print("\n[H] FMAT CNN multiplier, fine sweep (fixed values)")
for vf in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]:
    m = np.ones(5)
    m[1] = vf
    out.append(show(f"H/cnnFMAT={vf}", pool(cnn_m=m).argmax(1)))

print("\n[H2] FMAT CNN mult coarse LOO (grid .5,.6,.7,.8,.9,1.0 -> tiny selection space)")
cands = []
for vf in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    m = np.ones(5)
    m[1] = vf
    cands.append((f"FMAT={vf}", pool(cnn_m=m).argmax(1)))
r = referee(Y, FOLDS, cands)
print(f"  in_sample={r['in_sample_best_score']:.4f} ({r['in_sample_best_label']}) LOO={r['loo_honest']:.4f} "
      f"d={r['loo_honest']-BASE_MEAN:+.4f} chose={r['loo_chosen']} per-fold={np.round(np.array(r['loo_per_fold'])-BASE_VEC,4).tolist()}")
out.append({"tag": "H2/loo-cnnFMAT", "loo": r["loo_honest"], "delta": r["loo_honest"] - BASE_MEAN})


def class_stats(y, pred, m):
    """per-class F1 and precision of a component on mask m"""
    f1s, precs = {}, {}
    for c in range(5):
        tp = int(np.sum((y[m] == c) & (pred[m] == c)))
        fp = int(np.sum((pred[m] == c) & (y[m] != c)))
        fn = int(np.sum((y[m] == c) & (pred[m] != c)))
        sup = tp + fn
        if sup == 0:
            f1s[c] = np.nan
            precs[c] = np.nan
            continue
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / sup
        f1s[c] = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        precs[c] = pr
    return f1s, precs


print("\n[I] ZERO-parameter per-class reliability weighting (weights from other 4 folds)")
for stat_name, stat_fn in (("f1", 0), ("prec", 1)):
    pred = np.zeros(len(Y), dtype=int)
    Ws = {}
    for f in range(5):
        m = FOLDS != f
        tf, tp_ = class_stats(Y, TAB.argmax(1), m)
        cf, cp_ = class_stats(Y, CNN.argmax(1), m)
        ts_, cs_ = (tf, cf) if stat_fn == 0 else (tp_, cp_)
        wt = np.ones(5)
        wc = np.ones(5)
        for c in range(5):
            a_, b_ = ts_[c], cs_[c]
            if np.isnan(a_) or np.isnan(b_) or (a_ + b_) <= 0:
                continue
            # tab gets 0.40 share, cnn 0.60; redistribute by relative reliability
            share = a_ / (a_ + b_)
            wt[c] = 0.40 * share / 0.40
            wc[c] = 0.60 * (1 - share) / 0.60
        Ws[f] = (wt, wc)
        ts = sharpen(TAB[m], 6.0) * wt[None, :]
        cs = sharpen(CNN[m], 1.0) * wc[None, :]
        p = 0.40 * ts + 0.60 * cs
        pred[m] = p.argmax(1)
    print(f"  I/reliability_{stat_name}: weights per fold")
    for f in range(5):
        print(f"      fold {f}: tab={np.round(Ws[f][0],3).tolist()} cnn={np.round(Ws[f][1],3).tolist()}")
    out.append(show(f"I/reliability_{stat_name}", pred))

print("\n[J] per-class CNN reliability weighting ONLY on the CNN side (tab untouched)")
pred = np.zeros(len(Y), dtype=int)
for f in range(5):
    m = FOLDS != f
    tf, _ = class_stats(Y, TAB.argmax(1), m)
    cf, _ = class_stats(Y, CNN.argmax(1), m)
    wc = np.ones(5)
    for c in range(5):
        if np.isnan(tf[c]) or np.isnan(cf[c]) or (tf[c] + cf[c]) <= 0:
            continue
        share = cf[c] / (tf[c] + cf[c])
        wc[c] = share / 0.5      # 1.0 when both components equally reliable
    ts = sharpen(TAB[m], 6.0)
    cs = sharpen(CNN[m], 1.0) * wc[None, :]
    p = 0.40 * ts + 0.60 * cs
    pred[m] = p.argmax(1)
    if f == 0:
        print(f"      e.g. fold0 cnn weights (from folds1-4) = {np.round(wc,3).tolist()}")
out.append(show("J/cnn-reliability", pred))

with open("experiments/screen_iter1b.json", "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter1b.json")
