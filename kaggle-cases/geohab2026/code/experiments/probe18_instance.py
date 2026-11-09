#!/usr/bin/env python3
"""Screen 7 — instance-adaptive fusion operators (zero free parameters).

The class-adaptive reliability operator (FUSE-02, now the baseline) still applies ONE pooled
posterior per point. But the folds show the two components trading places completely: fold 0
CNN 0.854 vs tab 0.687, fold 4 CNN 0.677 vs tab 0.790. A per-instance rule that reacts to how
confident/agreeing the two components are is the natural next family, and the library records
that only *scalar* confidence gating was tried before.

Every candidate here uses NO fitted parameter, so the reported fold-mean is already honest.
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


def shares_for(mask, eps=0.5):
    sh = np.full(5, 0.5)
    for c in range(5):
        rt = cls_prf(Y[mask], TAB.argmax(1)[mask], c)
        rc = cls_prf(Y[mask], CNN.argmax(1)[mask], c)
        if rt is None or rc is None:
            continue
        sh[c] = (rt["F1"] + eps * 0.5) / (rt["F1"] + rc["F1"] + eps)
    return sh


# ── the FUSE-02 incumbent, recomputed here so every delta is against it ───────────
INCUMBENT = np.zeros(len(Y), dtype=int)
for f in range(5):
    va = FOLDS == f
    sh = shares_for(FOLDS != f)
    p = 0.40 * TS6[va] * sh[None, :] + 0.60 * CNN[va] * (1 - sh)[None, :]
    INCUMBENT[va] = p.argmax(1)
inc_vec = fold_vec(Y, INCUMBENT, FOLDS)
inc_mean = float(inc_vec.mean())
print(f"[incumbent FUSE-02 eps=0.5] fold-mean={inc_mean:.6f} per-fold={np.round(inc_vec,4).tolist()}")


def show(tag, pred):
    fv = fold_vec(Y, pred, FOLDS)
    d_inc = float(fv.mean() - inc_mean)
    d_base = float(fv.mean() - BASE_MEAN)
    pf = np.round(fv - inc_vec, 4)
    npos = int((fv - inc_vec >= -1e-12).sum())
    print(f"  {tag:46s} mean={fv.mean():.4f} d_inc={d_inc:+.4f} d_base={d_base:+.4f} "
          f"pos={npos}/5 {pf.tolist()}")
    return {"tag": tag, "mean": float(fv.mean()), "delta_incumbent": d_inc,
            "delta_baseline": d_base, "n_pos_vs_incumbent": npos, "per_fold_delta_inc": pf.tolist()}


out = []
CT = TAB.max(1)
CC = CNN.max(1)
# entropy of each component posterior (natural log, in units of log 5)
HT = -(np.where(TAB > 0, TAB * np.log(TAB + 1e-300), 0.0)).sum(1) / np.log(5)
HC = -(np.where(CNN > 0, CNN * np.log(CNN + 1e-300), 0.0)).sum(1) / np.log(5)
# normalised margin (top1 - top2)/top1
S6T = np.sort(TAB, 1)
S6C = np.sort(CNN, 1)
MT = (S6T[:, -1] - S6T[:, -2]) / np.maximum(S6T[:, -1], 1e-12)
MC = (S6C[:, -1] - S6C[:, -2]) / np.maximum(S6C[:, -1], 1e-12)
AGREE = TAB.argmax(1) == CNN.argmax(1)


def fold_adaptive(pool_fn):
    pred = np.zeros(len(Y), dtype=int)
    for f in range(5):
        va = FOLDS == f
        sh = shares_for(FOLDS != f)
        pred[va] = pool_fn(va, sh)
    return pred


print("\n[V] instance-adaptive reweighting of the FUSE-02 pool (zero fitted parameters)")
for g in [1.0, 2.0, 4.0]:
    p = fold_adaptive(lambda va, sh, g=g: (
        0.40 * TS6[va] * sh[None, :] * (CT[va] ** g)[:, None]
        + 0.60 * CNN[va] * (1 - sh)[None, :] * (CC[va] ** g)[:, None]
    ).argmax(1))
    out.append(show(f"V1/confidence^{g} weighting", p))

for g in [1.0, 2.0]:
    p = fold_adaptive(lambda va, sh, g=g: (
        0.40 * TS6[va] * sh[None, :] * np.exp(-g * HT[va])[:, None]
        + 0.60 * CNN[va] * (1 - sh)[None, :] * np.exp(-g * HC[va])[:, None]
    ).argmax(1))
    out.append(show(f"V2/entropy weighting g={g}", p))

for g in [1.0, 2.0, 4.0]:
    p = fold_adaptive(lambda va, sh, g=g: (
        0.40 * TS6[va] * sh[None, :] * (MT[va] ** g)[:, None]
        + 0.60 * CNN[va] * (1 - sh)[None, :] * (MC[va] ** g)[:, None]
    ).argmax(1))
    out.append(show(f"V3/margin weighting g={g}", p))

print("\n[W] agreement gating (where the two components agree, does forcing their choice help/hurt?)")
p = fold_adaptive(lambda va, sh: np.where(AGREE[va], TAB[va].argmax(1),
                                           (0.40 * TS6[va] * sh[None, :]
                                            + 0.60 * CNN[va] * (1 - sh)[None, :]).argmax(1)))
out.append(show("W1/agree -> tab argmax (they coincide)", p))

p = fold_adaptive(lambda va, sh: np.where(
    AGREE[va],
    (0.40 * TS6[va] * sh[None, :] + 0.60 * CNN[va] * (1 - sh)[None, :]).argmax(1),
    np.where(CT[va] >= CC[va], TAB[va].argmax(1), CNN[va].argmax(1))))
out.append(show("W2/disagree -> more confident component", p))

p = fold_adaptive(lambda va, sh: np.where(
    AGREE[va],
    (0.40 * TS6[va] * sh[None, :] + 0.60 * CNN[va] * (1 - sh)[None, :]).argmax(1),
    (0.40 * TS6[va] * sh[None, :] / np.maximum(CT[va], 1e-9)[:, None]
     + 0.60 * CNN[va] * (1 - sh)[None, :] / np.maximum(CC[va], 1e-9)[:, None]).argmax(1)))
out.append(show("W3/disagree -> self-normalised confidence pool", p))

p = fold_adaptive(lambda va, sh: np.where(
    AGREE[va],
    (0.40 * TS6[va] * sh[None, :] + 0.60 * CNN[va] * (1 - sh)[None, :]).argmax(1),
    TAB[va].argmax(1) if False else np.where(MC[va] >= 0.5, CNN[va].argmax(1), TAB[va].argmax(1))))
out.append(show("W4/disagree -> cnn if margin_cnn>=0.5", p))

print("\n[X] geometric pooling with the reliability shares")
for w in [0.40]:
    def gp(va, sh, w=w):
        lt = np.log(np.maximum(TS6[va], 1e-300)) * (w * sh)[None, :]
        lc = np.log(np.maximum(CNN[va], 1e-300)) * ((1 - w) * (1 - sh))[None, :]
        return (lt + lc).argmax(1)
    out.append(show("X/geometric pooling, FUSE-02 shares", fold_adaptive(gp)))

with open("experiments/screen_iter2a.json", "w") as fh:
    json.dump({"incumbent": inc_mean, "baseline": BASE_MEAN, "results": out}, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter2a.json")
