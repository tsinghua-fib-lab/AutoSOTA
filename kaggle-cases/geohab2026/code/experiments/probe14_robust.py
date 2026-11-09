#!/usr/bin/env python3
"""Screen 4 — robustness sweep of the reliability-scaled fusion operator (K5 family).

Operator:  p_i,c = 0.40 * share_c * tab^a_i,c  +  0.60 * (1 - share_c) * cnn_i,c
           share_c = (F1_tab,c + eps/2) / (F1_tab,c + F1_cnn,c + eps)
           with both F1's measured ONLY on the other four folds (zero tuned parameters).

This screen varies the *measurement* of reliability and the shrinkage, to check that the gain
is a plateau rather than a knife-edge, and to look for a strictly better measurement.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "experiments")
from probe_referee import fold_mean, fold_vec, load, selftest  # noqa: E402

D = load()
selftest(D)
Y, FOLDS = D["y"], D["folds"]
TAB, CNN = D["tab"], D["cnn"]
BASE_MEAN = fold_mean(Y, D["blend"].argmax(1), FOLDS)
BASE_VEC = fold_vec(Y, D["blend"].argmax(1), FOLDS)
print(f"[frozen] baseline = {BASE_MEAN:.6f}")


def sharpen(p, e):
    if e == 1.0:
        return p
    q = np.power(p, e)
    return q / q.sum(1, keepdims=True)


def cls_prf(y, pred, c):
    tp = int(np.sum((y == c) & (pred == c)))
    fp = int(np.sum((pred == c) & (y != c)))
    fn = int(np.sum((y == c) & (pred != c)))
    sup = tp + fn
    if sup == 0:
        return None
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / sup
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
    return {"P": pr, "R": rc, "F1": f1, "acc": (tp + 1.0 * 0) / sup, "support": sup}


def reliability(y, pred, m, metric, per_fold):
    fs = sorted(set(FOLDS[m].tolist()))
    out = {}
    for c in range(5):
        if per_fold:
            vals = [cls_prf(y[m & (FOLDS == f)], pred[m & (FOLDS == f)], c) for f in fs]
            vals = [v[metric] for v in vals if v is not None]
            out[c] = float(np.mean(vals)) if vals else np.nan
        else:
            r = cls_prf(y[m], pred[m], c)
            out[c] = np.nan if r is None else r[metric]
    return out


def run(metric="F1", eps=0.05, a_tab=6.0, a_cnn=1.0, per_fold=False, w_tab=0.40, w_cnn=0.60,
        share_mode="scaled"):
    pred = np.zeros(len(Y), dtype=int)
    folds_shares = {}
    for f in range(5):
        m = FOLDS != f
        rt = reliability(Y, TAB.argmax(1), m, metric, per_fold)
        rc = reliability(Y, CNN.argmax(1), m, metric, per_fold)
        sh = np.full(5, 0.5)
        for c in range(5):
            if np.isnan(rt[c]) or np.isnan(rc[c]):
                continue
            sh[c] = (rt[c] + eps * 0.5) / (rt[c] + rc[c] + eps)
        folds_shares[f] = sh
        ts, cs = sharpen(TAB[m], a_tab), sharpen(CNN[m], a_cnn)
        if share_mode == "scaled":
            p = (w_tab * sh)[None, :] * ts + (w_cnn * (1.0 - sh))[None, :] * cs
        else:  # 'convex'
            p = sh[None, :] * ts + (1.0 - sh)[None, :] * cs
        pred[m] = p.argmax(1)
    return pred, folds_shares


def show(tag, pred, extra=""):
    pf = fold_vec(Y, pred, FOLDS)
    d = pf.mean() - BASE_MEAN
    npos = int(((pf - BASE_VEC) >= -1e-12).sum())
    print(f"  {tag:40s} mean={pf.mean():.4f} d={d:+.4f} pos={npos}/5 per-fold={np.round(pf-BASE_VEC,4).tolist()} {extra}")
    return {"tag": tag, "mean": float(pf.mean()), "delta": float(d),
            "per_fold_delta": np.round(pf - BASE_VEC, 4).tolist(), "n_pos": npos}


out = []
print("\n[M] eps sensitivity (metric=F1, scaled, a_tab=6)")
for eps in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0, 3.0]:
    p, _ = run(eps=eps)
    out.append(show(f"M/eps={eps}", p))

print("\n[M2] reliability metric (scaled, a_tab=6, eps=0.05)")
for met in ["F1", "P", "R", "acc"]:
    p, _ = run(metric=met)
    out.append(show(f"M2/metric={met}", p))

print("\n[M3] per-fold-averaged reliability vs pooled")
for pf in [False, True]:
    p, _ = run(per_fold=pf)
    out.append(show(f"M3/per_fold={pf}", p))

print("\n[M4] a_tab sensitivity (scaled, F1, eps=0.05)")
for a in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 1e9]:
    p, _ = run(a_tab=a)
    out.append(show(f"M4/a_tab={a}", p))

print("\n[M5] global weight sensitivity at eps=0.05 (does 0.40/0.60 still matter?)")
for wt in [0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
    p, _ = run(w_tab=wt, w_cnn=1.0 - wt)
    out.append(show(f"M5/w_tab={wt}", p))

print("\n[M6] convex mode (drop the global 0.40/0.60)")
p, _ = run(share_mode="convex")
out.append(show("M6/convex", p))

print("\n[M7] leave-one-fold-out selection over eps (the honest gate for this knob)")
cands = []
for eps in [0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]:
    pp, _ = run(eps=eps)
    cands.append((f"eps={eps}", pp))
from probe_referee import referee  # noqa: E402
r = referee(Y, FOLDS, cands)
print(f"  in_sample={r['in_sample_best_score']:.4f} ({r['in_sample_best_label']}) LOO={r['loo_honest']:.4f} "
      f"d={r['loo_honest']-BASE_MEAN:+.4f} chose={r['loo_chosen']} per-fold={np.round(np.array(r['loo_per_fold'])-BASE_VEC,4).tolist()}")
out.append({"tag": "M7/loo-eps", "loo": r["loo_honest"], "delta": r["loo_honest"] - BASE_MEAN,
            "chosen": r["loo_chosen"]})

p, sh = run()
print("\n[M8] per-fold share vectors at the chosen setting:")
for f in range(5):
    print(f"      fold {f}: {np.round(sh[f],4).tolist()}")

with open("experiments/screen_iter1d.json", "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter1d.json")
