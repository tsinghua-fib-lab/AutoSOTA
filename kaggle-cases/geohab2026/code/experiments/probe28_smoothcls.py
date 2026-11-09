#!/usr/bin/env python3
"""Screen 16 — class-aware and confidence-gated variants of FUSE-03.

FUSE-03's one consistent loser is fold 3 (-0.011), whose blocks mix ALG/NVB patches: a majority
filter erases minority patches. Two principled repairs, each with at most one parameter:
  A. smooth only the classes that are not rare (rare classes are the ones a majority filter kills)
  B. only smooth a point whose own posterior is not already confident, and/or scale the
     smoothing weight by the within-fold class support (zero free parameters)
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
CLASSES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
XY = np.column_stack([D["df"]["x"].to_numpy(), D["df"]["y"].to_numpy()])
TREE = cKDTree(XY)
DIST, IDX = TREE.query(XY, k=250)
BASE_VEC = fold_vec(Y, D["blend"].argmax(1), FOLDS)
BASE_MEAN = float(BASE_VEC.mean())
CLASS_FREQ = np.array([float((Y == c).mean()) for c in range(5)])
print("pooled class frequencies:", np.round(CLASS_FREQ, 4).tolist())


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


def pool_oof(eps=0.5):
    ts = sharpen(TAB, 6.0)
    P = np.zeros((len(Y), 5))
    for f in range(5):
        va = FOLDS == f
        sh = np.full(5, 0.5)
        for c in range(5):
            rt = cls_prf(Y[FOLDS != f], TAB.argmax(1)[FOLDS != f], c)
            rc = cls_prf(Y[FOLDS != f], CNN.argmax(1)[FOLDS != f], c)
            if rt is None or rc is None:
                continue
            sh[c] = (rt["F1"] + eps * 0.5) / (rt["F1"] + rc["F1"] + eps)
        p = 0.40 * ts[va] * sh[None, :] + 0.60 * CNN[va] * (1 - sh)[None, :]
        P[va] = p / p.sum(1, keepdims=True)
    return P


P0 = pool_oof()


def nb_mean(P, k):
    idx = IDX[:, :k]
    w = (FOLDS[idx] == FOLDS[:, None]).astype(np.float64)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return np.einsum("ij,ijc->ic", w, P[idx])


NB = nb_mean(P0, 120)


def smooth_vec(P, alpha_vec, k=120):
    nb = NB if k == 120 else nb_mean(P, k)
    Q = alpha_vec[np.newaxis, :] * P + (1.0 - alpha_vec)[np.newaxis, :] * nb
    return Q / Q.sum(1, keepdims=True)


def smooth_gated(P, alpha, tau, k=120):
    nb = NB if k == 120 else nb_mean(P, k)
    conf = P.max(1)
    a_i = np.where(conf >= tau, 1.0, alpha)
    Q = a_i[:, None] * P + (1.0 - a_i)[:, None] * nb
    return Q / Q.sum(1, keepdims=True)


INC_VEC = fold_vec(Y, smooth_vec(P0, np.full(5, 0.7)).argmax(1), FOLDS)
INC_MEAN = float(INC_VEC.mean())
print(f"[incumbent iter2] mean={INC_MEAN:.6f}")
out = []


def show(tag, pred):
    fv = fold_vec(Y, pred, FOLDS)
    pf = np.round(fv - INC_VEC, 4)
    print(f"  {tag:36s} mean={fv.mean():.4f} d_inc={fv.mean()-INC_MEAN:+.4f} "
          f"d_base={fv.mean()-BASE_MEAN:+.4f} pos={int((fv-INC_VEC >= -1e-12).sum())}/5 {pf.tolist()}")
    return {"tag": tag, "mean": float(fv.mean()), "delta_inc": float(fv.mean() - INC_MEAN),
            "n_pos": int((fv - INC_VEC >= -1e-12).sum()), "per_fold_delta": pf.tolist()}


print("\n[PP] per-class alpha vectors (1.0 = leave that class unsmoothed)")
VECS = {
    "all 0.7 (incumbent)": [0.7] * 5,
    "no smooth ALG": [1.0, 0.7, 0.7, 0.7, 0.7],
    "no smooth SGZ": [0.7, 0.7, 0.7, 0.7, 1.0],
    "no smooth ALG+SGZ": [1.0, 0.7, 0.7, 0.7, 1.0],
    "no smooth ALG+SGZ+SGAM": [1.0, 0.7, 0.7, 1.0, 1.0],
    "0.85 for rare, 0.7 else": [0.85, 0.7, 0.7, 0.85, 0.85],
}
for tag, v in VECS.items():
    out.append(show(f"PP/{tag}", smooth_vec(P0, np.array(v)).argmax(1)))

print("\n[QQ] zero-parameter support-scaled alpha: alpha_c = 1 - (1-a0)*freq_c^gamma")
for a0 in [0.7]:
    for gamma in [0.25, 0.5, 1.0]:
        av = 1.0 - (1.0 - a0) * (CLASS_FREQ ** gamma)
        av = np.clip(av, 0.0, 1.0)
        print(f"    a0={a0} gamma={gamma} -> alpha={np.round(av,4).tolist()}")
        out.append(show(f"QQ/support-scaled g={gamma}", smooth_vec(P0, av).argmax(1)))

print("\n[RR] confidence-gated smoothing (only smooth points with max-prob < tau)")
for tau in [0.4, 0.5, 0.6, 0.7, 0.8]:
    out.append(show(f"RR/tau={tau}", smooth_gated(P0, 0.7, tau).argmax(1)))

print("\n[SS] referee over the union of the class-aware variants (honest gate)")
cands = []
for tag, v in VECS.items():
    cands.append((tag, smooth_vec(P0, np.array(v)).argmax(1)))
for gamma in [0.25, 0.5, 1.0]:
    av = np.clip(1.0 - 0.3 * (CLASS_FREQ ** gamma), 0, 1)
    cands.append((f"support-g={gamma}", smooth_vec(P0, av).argmax(1)))
r = referee(Y, FOLDS, cands)
print(f"  in_sample best={r['in_sample_best_score']:.4f} ({r['in_sample_best_label']}) "
      f"LOO={r['loo_honest']:.4f} d_base={r['loo_honest']-BASE_MEAN:+.4f} chose={r['loo_chosen']}")
out.append({"tag": "SS/loo", "loo": float(r["loo_honest"]), "chosen": r["loo_chosen"]})

with open("experiments/screen_iter3_smoothcls.json", "w") as fh:
    json.dump({"incumbent": INC_MEAN, "baseline": BASE_MEAN, "results": out}, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter3_smoothcls.json")
