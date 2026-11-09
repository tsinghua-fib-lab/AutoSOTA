#!/usr/bin/env python3
"""Corrected operator screen (REF-02 harness bug fix).

The earlier reliability screens applied each fold's parameters to the rows they were *derived*
from (mask `folds != f` used for both derivation and application). That is in-sample parameter
selection and it inflated the deltas. This harness applies every fold-adaptive operator the only
legitimate way:

    for held-out fold f: derive everything from rows with folds != f, apply ONLY to folds == f.

Global (non-fold-adaptive) candidates keep the pure-referee protocol (LOO selection over their
parameter grid), which probe11 already implemented correctly.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "experiments")
from probe_referee import CLASSES, fold_vec, load, referee, selftest  # noqa: E402

D = load()
selftest(D)
Y, FOLDS = D["y"], D["folds"]
TAB, CNN = D["tab"], D["cnn"]
NF = 5
BASE_VEC = fold_vec(Y, D["blend"].argmax(1), FOLDS)
BASE_MEAN = float(BASE_VEC.mean())
print(f"[frozen] baseline = {BASE_MEAN:.6f} per-fold {np.round(BASE_VEC,4).tolist()}")


def sharpen(p, e):
    if e == 1.0:
        return p
    q = np.power(p, e)
    return q / q.sum(1, keepdims=True)


TS = {a: sharpen(TAB, a) for a in (1.0, 3.0, 6.0, 12.0)}
CS = {b: sharpen(CNN, b) for b in (1.0,)}


def pool2(tab_m=None, cnn_m=None, a=6.0, b=1.0, w_tab=0.40):
    """p = w_tab*tab_m*tab^a + (1-w_tab)*cnn_m*cnn^b, row-normalised."""
    ts = TS[a] if tab_m is None else TS[a] * tab_m[np.newaxis, :]
    cs = CS[b] if cnn_m is None else CS[b] * cnn_m[np.newaxis, :]
    p = w_tab * ts + (1.0 - w_tab) * cs
    return p / p.sum(1, keepdims=True)


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


def make_fold_adaptive(param_fn, a=6.0, b=1.0, w_tab=0.40):
    """param_fn(train_mask) -> (tab_mult|None, cnn_mult|None). Applied only to the held-out fold."""
    pred = np.zeros(len(Y), dtype=int)
    detail = {}
    for f in range(NF):
        va = FOLDS == f
        tm, cm = param_fn(~va)
        detail[f] = (tm, cm)
        pred[va] = pool2(tm, cm, a, b, w_tab)[va].argmax(1)
    return pred, detail


def show(tag, pred, extra=""):
    fv = fold_vec(Y, pred, FOLDS)
    d = float(fv.mean() - BASE_MEAN)
    pf = np.round(fv - BASE_VEC, 4)
    npos = int((fv - BASE_VEC >= -1e-12).sum())
    print(f"  {tag:44s} mean={fv.mean():.4f} d={d:+.4f} pos={npos}/5 per-fold={pf.tolist()} {extra}")
    return {"tag": tag, "mean": float(fv.mean()), "delta": d, "n_pos": npos,
            "per_fold_delta": pf.tolist()}


results = []

# ── sanity: the operator with no per-class term must reproduce the baseline exactly ──
p0, _ = make_fold_adaptive(lambda tr: (None, None))
assert np.array_equal(p0, D["blend"].argmax(1)), "pool2 is not equivalent to the frozen blend"
print("[sanity] pool2(None,None) == frozen blend argmax OK")

print("\n[N] reliability-scaled share, CORRECT protocol")
for metric in ["F1", "P"]:
    for eps in [0.0, 0.05, 0.2, 1.0]:
        def pf(tr, metric=metric, eps=eps):
            sh = np.full(5, 0.5)
            for c in range(5):
                rt = cls_prf(Y[tr], TAB.argmax(1)[tr], c)
                rc = cls_prf(Y[tr], CNN.argmax(1)[tr], c)
                if rt is None or rc is None:
                    continue
                sh[c] = (rt[metric] + eps * 0.5) / (rt[metric] + rc[metric] + eps)
            return sh, 1.0 - sh
        p, det = make_fold_adaptive(pf)
        results.append(show(f"N/reliability/{metric}/eps={eps}", p))
        if metric == "F1" and eps == 0.05:
            for f in range(NF):
                print(f"        fold {f}: share={np.round(det[f][0],3).tolist()}")

print("\n[O] reliability share raised to a power gamma (gamma=1 is the plain share)")
for gamma in [0.5, 1.0, 2.0, 4.0]:
    def pf(tr, gamma=gamma):
        sh = np.full(5, 0.5)
        for c in range(5):
            rt = cls_prf(Y[tr], TAB.argmax(1)[tr], c)
            rc = cls_prf(Y[tr], CNN.argmax(1)[tr], c)
            if rt is None or rc is None:
                continue
            s = (rt["F1"] + 0.025) / (rt["F1"] + rc["F1"] + 0.05)
            sh[c] = s ** gamma
        return sh, 1.0 - sh
    p, _ = make_fold_adaptive(pf)
    results.append(show(f"O/reliability^gamma={gamma}", p))

print("\n[P] reliability share, but keeping the global 0.40/0.60 and scaling only the tab side")
for gamma in [1.0, 2.0]:
    def pf(tr, gamma=gamma):
        sh = np.full(5, 0.5)
        for c in range(5):
            rt = cls_prf(Y[tr], TAB.argmax(1)[tr], c)
            rc = cls_prf(Y[tr], CNN.argmax(1)[tr], c)
            if rt is None or rc is None:
                continue
            s = (rt["F1"] + 0.025) / (rt["F1"] + rc["F1"] + 0.05)
            sh[c] = s ** gamma
        return sh, None      # only the tabular side is re-scaled
    p, _ = make_fold_adaptive(pf)
    results.append(show(f"P/tab-only-share^gamma={gamma}", p))

print("\n[Q] simple per-class CNN multiplier, but DERIVED (zero parameters) and fold-honest")
for gamma in [1.0, 2.0, 4.0]:
    def pf(tr, gamma=gamma):
        cm = np.ones(5)
        for c in range(5):
            rt = cls_prf(Y[tr], TAB.argmax(1)[tr], c)
            rc = cls_prf(Y[tr], CNN.argmax(1)[tr], c)
            if rt is None or rc is None or rt["F1"] <= 1e-9:
                continue
            cm[c] = min(3.0, (rc["F1"] / rt["F1"]) ** gamma)
        return None, cm
    p, _ = make_fold_adaptive(pf)
    results.append(show(f"Q/cnn-mult-from-ratio^gamma={gamma}", p))

print("\n[R] FIXED global per-class multipliers (no fold adaptation; the probe11 winners)")
for tag, tm, cm in [
    ("R/SGAM-cnn=1.2", None, [1, 1, 1, 1.2, 1]),
    ("R/FMAT-cnn=0.65", None, [1, 0.65, 1, 1, 1]),
    ("R/FMAT=.65,SGAM=1.2", None, [1, 0.65, 1, 1.2, 1]),
    ("R/SGAM-tab=0.6", [1, 1, 1, 0.6, 1], None),
    ("R/FMAT=.65,SGAMtab=.6", [1, 1, 1, 0.6, 1], [1, 0.65, 1, 1, 1]),
]:
    p = pool2(np.array(tm, float) if tm else None, np.array(cm, float) if cm else None).argmax(1)
    results.append(show(tag, p))

print("\n[S] reliability share COMBINED with the fixed FMAT multiplier")
for vf in [0.65, 0.7, 0.8, 1.0]:
    def pf(tr, vf=vf):
        sh = np.full(5, 0.5)
        for c in range(5):
            rt = cls_prf(Y[tr], TAB.argmax(1)[tr], c)
            rc = cls_prf(Y[tr], CNN.argmax(1)[tr], c)
            if rt is None or rc is None:
                continue
            sh[c] = (rt["F1"] + 0.025) / (rt["F1"] + rc["F1"] + 0.05)
        mn = np.ones(5)
        mn[1] = vf
        return sh * mn, (1.0 - sh) * mn
    p, _ = make_fold_adaptive(pf)
    results.append(show(f"S/reliability x FMATmult={vf}", p))

print("\n[S2] reliability share with a FACTORISED form: share_c * mult_c on the CNN side only")
for vf in [0.65, 0.75, 1.0]:
    def pf(tr, vf=vf):
        sh = np.full(5, 0.5)
        for c in range(5):
            rt = cls_prf(Y[tr], TAB.argmax(1)[tr], c)
            rc = cls_prf(Y[tr], CNN.argmax(1)[tr], c)
            if rt is None or rc is None:
                continue
            sh[c] = (rt["F1"] + 0.025) / (rt["F1"] + rc["F1"] + 0.05)
        mn = np.ones(5)
        mn[1] = vf
        return sh, (1.0 - sh) * mn
    p, _ = make_fold_adaptive(pf)
    results.append(show(f"S2/reliability, cnn FMAT mult={vf}", p))

with open("experiments/screen_iter1e.json", "w") as fh:
    json.dump(results, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter1e.json")
