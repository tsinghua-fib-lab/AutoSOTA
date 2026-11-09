#!/usr/bin/env python3
"""Screen 10 — FMAT-SPEC: a gated FMAT/NVB/SGAM specialist.

Diagnosis driving this (experiments/probe21_reftable.py): FUSE-02's two largest weighted losses
are both FMAT, and they have opposite causes —
    fold 0 FMAT n=286: cnn F1 0.783, tab 0.484 -> the *tabular* is destroying it
    fold 4 FMAT n=236: tab F1 0.522, cnn 0.035 -> the *CNN* is destroying it
so no global per-class weight can fix both (which is exactly why the single FMAT multiplier
gained on one fold and lost on another). A second model with capacity dedicated to the
{FMAT, NVB, SGAM} boundary is the structurally different attempt.

Protocol: the specialist for fold f is trained only on rows with folds != f whose label lies in
{FMAT, NVB, SGAM}; the override fires only where the incumbent's top-1 and top-2 are both in
that set and the specialist's margin exceeds tau. tau is selected leave-one-fold-out.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
from probe_referee import fold_vec, load, selftest, wf1_fast  # noqa: E402

PAPER_DATA = "/paper_data"
CLASSES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
SUB = [1, 2, 3]          # FMAT, NVB, SGAM
SUBNAME = [CLASSES[i] for i in SUB]

D = load()
selftest(D)
Y, FOLDS = D["y"], D["folds"]
TAB, CNN = D["tab"], D["cnn"]

# ── feature assembly identical to the pipeline §3 ────────────────────────────────
v07_tr = pd.read_csv(f"{PAPER_DATA}/features/train_features_v07.csv")
v09_tr = pd.read_csv(f"{PAPER_DATA}/features/train_features_v09.csv", usecols=["x", "y", "acr_sapa_w21"])
cols = [c for c in v07_tr.columns if c not in ("x", "y", "class")]
X = np.column_stack([v07_tr[cols].to_numpy(), v09_tr["acr_sapa_w21"].to_numpy()])
assert X.shape == (6256, 87), X.shape


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


PROB = np.zeros((len(Y), 5))
for f in range(5):
    va = FOLDS == f
    sh = shares_for(FOLDS != f)
    p = 0.40 * TS6[va] * sh[None, :] + 0.60 * CNN[va] * (1 - sh)[None, :]
    PROB[va] = p / p.sum(1, keepdims=True)
INC = PROB.argmax(1)
inc_vec = fold_vec(Y, INC, FOLDS)
inc_mean = float(inc_vec.mean())
print(f"[incumbent FUSE-02] mean={inc_mean:.6f} per-fold={np.round(inc_vec,4).tolist()}")


def show(tag, pred):
    fv = fold_vec(Y, pred, FOLDS)
    pf = np.round(fv - inc_vec, 4)
    print(f"  {tag:40s} mean={fv.mean():.4f} d_inc={fv.mean()-inc_mean:+.4f} "
          f"pos={int((fv-inc_vec >= -1e-12).sum())}/5 {pf.tolist()}")
    return {"tag": tag, "mean": float(fv.mean()), "delta_inc": float(fv.mean() - inc_mean),
            "n_pos": int((fv - inc_vec >= -1e-12).sum()), "per_fold_delta": pf.tolist()}


# ── train the specialist per fold ────────────────────────────────────────────────
print("\n[specialist] training 5 fold specialists on {FMAT,NVB,SGAM} rows ...")
t0 = time.time()
from lightgbm import LGBMClassifier  # noqa: E402

SPEC = np.zeros((len(Y), 5))
for f in range(5):
    tr = (FOLDS != f) & np.isin(Y, SUB)
    va = (FOLDS == f) & np.isin(Y, SUB)
    m = LGBMClassifier(n_estimators=1000, random_state=42, verbose=-1, n_jobs=8, num_leaves=31,
                       min_child_samples=100, learning_rate=0.1, feature_fraction=1.0)
    m.fit(X[tr], Y[tr])
    pr = m.predict_proba(X[va])
    for k, cl in enumerate(m.classes_):
        SPEC[va, int(cl)] = pr[:, k]
print(f"  done in {time.time()-t0:.0f}s; specialist coverage on subset rows: "
      f"{int((SPEC.sum(1) > 0).sum())}/{int(np.isin(Y, SUB).sum())}")

SPC = SPEC.argmax(1)
SPC_MARGIN = np.sort(SPEC, 1)[:, -1] - np.sort(SPEC, 1)[:, -2]
TOP2 = np.argsort(-PROB, 1)[:, :2]
GATE_TOP2 = np.isin(TOP2, SUB).all(1)          # both top-1 and top-2 inside {FMAT,NVB,SGAM}
print(f"  gate fires on {int((GATE_TOP2 & (SPEC.sum(1) > 0)).sum())} of "
      f"{int(GATE_TOP2.sum())} eligible rows ({GATE_TOP2.mean()*100:.1f}% of all rows)")

out = []
print("\n[BB] override with margin threshold tau (fixed values, in-sample view)")
CAND = {}
for tau in [0.0, 0.1, 0.2, 0.3, 0.5]:
    for cap in [1.0, 0.10]:
        pred = INC.copy()
        fire = GATE_TOP2 & (SPEC.sum(1) > 0) & (SPC_MARGIN > tau)
        if cap < 1.0:
            # cap the override rate per fold: keep only the most confident `cap` fraction
            for f in range(5):
                m = fire & (FOLDS == f)
                idx = np.where(m)[0]
                if len(idx) > cap * int((FOLDS == f).sum()):
                    keep = idx[np.argsort(-SPC_MARGIN[idx])[:int(cap * int((FOLDS == f).sum()))]]
                    drop = np.setdiff1d(idx, keep)
                    fire[drop] = False
        pred[fire] = SPC[fire]
        CAND[(tau, cap)] = pred
        out.append(show(f"BB/tau={tau} cap={cap} (fire {int(fire.sum())})", pred))

print("\n[BC] leave-one-fold-out selection over tau (cap=1.0)")
gates = [0.0, 0.1, 0.2, 0.3, 0.5]
loo_per_fold, chosen = [], []
for f in range(5):
    tr = FOLDS != f
    best, best_s = None, -1.0
    for tau in gates:
        fire = GATE_TOP2 & (SPEC.sum(1) > 0) & (SPC_MARGIN > tau)
        pred = np.where(fire, SPC, INC)
        s = wf1_fast(Y[tr], pred[tr])
        if s > best_s:
            best_s, best = s, tau
    fire = GATE_TOP2 & (SPEC.sum(1) > 0) & (SPC_MARGIN > best)
    pred = np.where(fire, SPC, INC)
    va = FOLDS == f
    loo_per_fold.append(wf1_fast(Y[va], pred[va]))
    chosen.append(best)
loo = float(np.mean(loo_per_fold))
print(f"  LOO fold-mean={loo:.4f} d_inc={loo-inc_mean:+.4f} chose={chosen} "
      f"per-fold={np.round(np.array(loo_per_fold)-inc_vec,4).tolist()}")
out.append({"tag": "BC/loo-over-tau", "loo": loo, "delta_inc": loo - inc_mean, "chosen": chosen})

print("\n[BD] FMAT per-class F1 comparison, fold by fold")
for f in range(5):
    m = FOLDS == f
    r_inc = cls_prf(Y[m], INC[m], 1)
    pred = np.where(GATE_TOP2 & (SPEC.sum(1) > 0) & (SPC_MARGIN > 0.2), SPC, INC)
    r_new = cls_prf(Y[m], pred[m], 1)
    print(f"  fold {f}: inc FMAT F1={r_inc['F1']:.3f}   with specialist(tau=0.2) F1={r_new['F1']:.3f}")

with open("experiments/screen_iter4_specialist.json", "w") as fh:
    json.dump({"incumbent": inc_mean, "results": out}, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter4_specialist.json")
