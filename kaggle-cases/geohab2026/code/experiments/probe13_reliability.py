#!/usr/bin/env python3
"""Screen 3 — variants of the ZERO-free-parameter per-class reliability-weighted fusion.

The operator replaces the scalar 0.40/0.60 pool by a per-class tab:cnn ratio derived from each
component's own out-of-fold reliability on the OTHER four folds (classic cross-validated
stacking with a fixed, non-fitted formula). Nothing is tuned on the OOF that is scored: for
every candidate the number printed is already leave-one-fold-out honest.

Variants differ only in how "reliability" is measured and how the two component shares are
formed.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "experiments")
from probe_referee import CLASSES, fold_mean, fold_vec, load, selftest  # noqa: E402

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
    return {"P": pr, "R": rc, "F1": f1, "support": sup}


def reliability(y, pred, m, metric="F1", per_fold=False):
    """per-class reliability of one component, measured only on rows where mask m is True."""
    fs = sorted(set(FOLDS[m].tolist()))
    out = {}
    for c in range(5):
        if per_fold:
            vals = []
            for f in fs:
                mm = m & (FOLDS == f)
                r = cls_prf(y[mm], pred[mm], c)
                if r is not None:
                    vals.append(r[metric])
            out[c] = float(np.mean(vals)) if vals else np.nan
        else:
            r = cls_prf(y[m], pred[m], c)
            out[c] = np.nan if r is None else r[metric]
    return out


def build(variant, eps=0.05, a_tab=6.0, a_cnn=1.0, metric="F1", per_fold=False,
          mode="share", clip=None):
    """mode: 'share'  -> w_tab_c = share_c ; w_cnn_c = 1-share_c  (renormalised pool)
             'scaled' -> p = 0.40*share_c*ts_c + 0.60*(1-share_c)*cs_c
             'ratio'  -> p = 0.40*ts_c + 0.60*(share_c/(1-share_c))*cs_c"""
    pred = np.zeros(len(Y), dtype=int)
    tabs, cnns = {}, {}
    for f in range(5):
        m = FOLDS != f
        rt = reliability(Y, TAB.argmax(1), m, metric, per_fold)
        rc = reliability(Y, CNN.argmax(1), m, metric, per_fold)
        sh = np.full(5, 0.5)
        for c in range(5):
            t, k = rt[c], rc[c]
            if np.isnan(t) or np.isnan(k):
                continue
            sh[c] = (t + eps * 0.5) / (t + k + eps)
        if clip is not None:
            sh = np.clip(sh, clip[0], clip[1])
        tabs[f], cnns[f] = sh, 1.0 - sh
        ts = sharpen(TAB[m], a_tab)
        cs = sharpen(CNN[m], a_cnn)
        if mode == "share":
            w_t, w_c = sh, 1.0 - sh
        elif mode == "scaled":
            w_t, w_c = 0.40 * sh, 0.60 * (1.0 - sh)
        else:
            w_t, w_c = np.full(5, 0.40), 0.60 * np.where(sh >= 1 - 1e-12, 1e6, sh / np.maximum(1e-12, 1 - sh))
        p = w_t[None, :] * ts + w_c[None, :] * cs
        pred[m] = p.argmax(1)
    return pred, tabs, cnns


def show(tag, pred, extra=""):
    pf = fold_vec(Y, pred, FOLDS)
    d = pf.mean() - BASE_MEAN
    print(f"  {tag:44s} mean={pf.mean():.4f} d={d:+.4f} pos={int(((pf-BASE_VEC)>=-1e-12).sum())}/5 "
          f"per-fold={np.round(pf-BASE_VEC,4).tolist()} {extra}")
    return {"tag": tag, "mean": float(pf.mean()), "delta": float(d),
            "per_fold_delta": np.round(pf - BASE_VEC, 4).tolist(),
            "n_pos": int(((pf - BASE_VEC) >= -1e-12).sum())}


out = []

print("\n[K] variant sweep")
for tag, kw in [
    ("K1 share/F1/pooled      ", dict(mode="share", metric="F1", eps=0.05)),
    ("K2 share/F1/perfoldavg  ", dict(mode="share", metric="F1", per_fold=True, eps=0.05)),
    ("K3 share/acc/pooled     ", dict(mode="share", metric="R", eps=0.05)),
    ("K4 share/prec/pooled    ", dict(mode="share", metric="P", eps=0.05)),
    ("K5 scaled/F1/pooled     ", dict(mode="scaled", metric="F1", eps=0.05)),
    ("K6 ratio/F1/pooled      ", dict(mode="ratio", metric="F1", eps=0.05)),
    ("K7 share/F1/eps=0.01    ", dict(mode="share", metric="F1", eps=0.01)),
    ("K8 share/F1/eps=0.20    ", dict(mode="share", metric="F1", eps=0.20)),
    ("K9 share/F1/eps=0.50    ", dict(mode="share", metric="F1", eps=0.50)),
    ("K10 share/F1/clip[.2.8] ", dict(mode="share", metric="F1", eps=0.05, clip=(0.2, 0.8))),
    ("K11 share/F1/clip[.3.7] ", dict(mode="share", metric="F1", eps=0.05, clip=(0.3, 0.7))),
    ("K12 share/F1/atab=1     ", dict(mode="share", metric="F1", eps=0.05, a_tab=1.0)),
    ("K13 share/F1/atab=3     ", dict(mode="share", metric="F1", eps=0.05, a_tab=3.0)),
    ("K14 share/F1/atab=12    ", dict(mode="share", metric="F1", eps=0.05, a_tab=12.0)),
]:
    p, tabs, cnns = build(None, **kw)
    out.append(show(tag, p))

print("\n[K15] chosen-variant fold weights (share/F1/pooled, eps=0.05)")
p, tabs, cnns = build(None, mode="share", metric="F1", eps=0.05)
for f in range(5):
    print(f"      fold {f}: w_tab={np.round(tabs[f],3).tolist()} w_cnn={np.round(cnns[f],3).tolist()}")

print("\n[L] what would the full-train (all-5-fold) weights be, for the test matrices?")
rt = reliability(Y, TAB.argmax(1), np.ones(len(Y), bool), "F1")
rc = reliability(Y, CNN.argmax(1), np.ones(len(Y), bool), "F1")
sh_full = np.array([(rt[c] + 0.025) / (rt[c] + rc[c] + 0.05) for c in range(5)])
print("      tab per-class F1:", np.round([rt[c] for c in range(5)], 4).tolist())
print("      cnn per-class F1:", np.round([rc[c] for c in range(5)], 4).tolist())
print("      share (w_tab_c) :", np.round(sh_full, 4).tolist())

with open("experiments/screen_iter1c.json", "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter1c.json")
