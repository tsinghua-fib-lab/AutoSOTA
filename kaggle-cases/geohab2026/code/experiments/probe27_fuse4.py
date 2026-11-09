#!/usr/bin/env python3
"""Screen 15 — the remaining fusion knobs re-screened UNDER the FUSE-03 smoothing.

The smoothing changes the posterior landscape, so the round-1 knobs (tabular temperature a,
global weight w, reliability measure) may no longer sit at their old optima. Each is screened
with the referee over a deliberately NARROW grid: screen 14 showed that widening the (k,alpha)
grid lets leave-one-fold-out selection pick bad corners and lose several points, i.e. the LOO
gate itself degrades when the candidate set is padded with losers.
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
XY = np.column_stack([D["df"]["x"].to_numpy(), D["df"]["y"].to_numpy()])
TREE = cKDTree(XY)
DIST, IDX = TREE.query(XY, k=300)
BASE_VEC = fold_vec(Y, D["blend"].argmax(1), FOLDS)
BASE_MEAN = float(BASE_VEC.mean())


def sharpen(p, e):
    q = np.power(p, e)
    return q / q.sum(1, keepdims=True)


def measure(y, pred, c, mask, kind):
    tp = int(np.sum((y[mask] == c) & (pred[mask] == c)))
    fp = int(np.sum((pred[mask] == c) & (y[mask] != c)))
    fn = int(np.sum((y[mask] == c) & (pred[mask] != c)))
    s = tp + fn
    if s == 0:
        return None
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / s
    if kind == "F1":
        return 2 * pr * rec / (pr + rec) if (pr + rec) else 0.0
    return rec  # 'acc' == recall at the class level


def shares_for(mask, eps=0.5, kind="F1", tab=None, cnn=None):
    tab = TAB if tab is None else tab
    cnn = CNN if cnn is None else cnn
    sh = np.full(5, 0.5)
    for c in range(5):
        rt = measure(Y, tab.argmax(1), c, mask, kind)
        rc = measure(Y, cnn.argmax(1), c, mask, kind)
        if rt is None or rc is None:
            continue
        sh[c] = (rt + eps * 0.5) / (rt + rc + eps)
    return sh


def pool_oof(eps=0.5, kind="F1", atab=6.0, w_tab=0.40, tab=None, cnn=None):
    tab = TAB if tab is None else tab
    cnn = CNN if cnn is None else cnn
    ts = sharpen(tab, atab)
    P = np.zeros((len(Y), 5))
    for f in range(5):
        va = FOLDS == f
        sh = shares_for(FOLDS != f, eps, kind, tab=tab, cnn=cnn)
        p = w_tab * ts[va] * sh[None, :] + (1.0 - w_tab) * cnn[va] * (1 - sh)[None, :]
        P[va] = p / p.sum(1, keepdims=True)
    return P


def smooth(P, k, alpha):
    idx, d = IDX[:, :k], DIST[:, :k]
    w = (FOLDS[idx] == FOLDS[:, None]).astype(np.float64)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    Q = alpha * P + (1.0 - alpha) * np.einsum("ij,ijc->ic", w, P[idx])
    return Q / Q.sum(1, keepdims=True)


P0 = pool_oof()
INC_VEC = fold_vec(Y, smooth(P0, 120, 0.7).argmax(1), FOLDS)
INC_MEAN = float(INC_VEC.mean())
print(f"[incumbent iter2] mean={INC_MEAN:.6f}")
out = []

print("\n[LL] tabular temperature a under smoothing (narrow grid)")
CAND = {}
for a in [3.0, 4.0, 6.0, 8.0, 12.0]:
    CAND[a] = smooth(pool_oof(atab=a), 120, 0.7).argmax(1)
    fv = fold_vec(Y, CAND[a], FOLDS)
    print(f"  a={a:<5} mean={fv.mean():.4f} d_inc={fv.mean()-INC_MEAN:+.4f} "
          f"pos={int((fv-INC_VEC >= -1e-12).sum())}/5 {np.round(fv-INC_VEC,4).tolist()}")
r = referee(Y, FOLDS, [(f"a={a}", CAND[a]) for a in CAND])
print(f"  LOO={r['loo_honest']:.4f} d_base={r['loo_honest']-BASE_MEAN:+.4f} chose={r['loo_chosen']}")
out.append({"tag": "LL/loo-a", "loo": float(r["loo_honest"]), "chosen": r["loo_chosen"]})

print("\n[MM] global weight w_tab under smoothing (narrow grid)")
CAND2 = {}
for w in [0.35, 0.40, 0.45, 0.50]:
    CAND2[w] = smooth(pool_oof(w_tab=w), 120, 0.7).argmax(1)
    fv = fold_vec(Y, CAND2[w], FOLDS)
    print(f"  w={w:<5} mean={fv.mean():.4f} d_inc={fv.mean()-INC_MEAN:+.4f} "
          f"pos={int((fv-INC_VEC >= -1e-12).sum())}/5 {np.round(fv-INC_VEC,4).tolist()}")
r2 = referee(Y, FOLDS, [(f"w={w}", CAND2[w]) for w in CAND2])
print(f"  LOO={r2['loo_honest']:.4f} d_base={r2['loo_honest']-BASE_MEAN:+.4f} chose={r2['loo_chosen']}")
out.append({"tag": "MM/loo-w", "loo": float(r2["loo_honest"]), "chosen": r2["loo_chosen"]})

print("\n[NN] reliability measure (recall) under smoothing")
for kind in ["F1", "rec"]:
    for eps in [0.35, 0.5, 0.75]:
        fv = fold_vec(Y, smooth(pool_oof(eps=eps, kind=kind), 120, 0.7).argmax(1), FOLDS)
        print(f"  {kind:4s} eps={eps:<5} mean={fv.mean():.4f} d_inc={fv.mean()-INC_MEAN:+.4f} "
              f"pos={int((fv-INC_VEC >= -1e-12).sum())}/5 {np.round(fv-INC_VEC,4).tolist()}")
        out.append({"tag": f"NN/{kind}/eps={eps}", "mean": float(fv.mean()),
                    "delta_inc": float(fv.mean() - INC_MEAN)})

print("\n[OO] narrow (k, alpha) neighbourhood of the incumbent, LOO gate")
CAND3 = {}
for k in [100, 120, 160, 200]:
    for alpha in [0.65, 0.70, 0.75]:
        CAND3[(k, alpha)] = smooth(P0, k, alpha).argmax(1)
        fv = fold_vec(Y, CAND3[(k, alpha)], FOLDS)
        print(f"  k={k:<4} a={alpha} mean={fv.mean():.4f} d_inc={fv.mean()-INC_MEAN:+.4f} "
              f"pos={int((fv-INC_VEC >= -1e-12).sum())}/5 {np.round(fv-INC_VEC,4).tolist()}")
r3 = referee(Y, FOLDS, [(f"k={k},a={a}", CAND3[(k, a)]) for (k, a) in CAND3])
print(f"  LOO={r3['loo_honest']:.4f} d_base={r3['loo_honest']-BASE_MEAN:+.4f} chose={r3['loo_chosen']}")
out.append({"tag": "OO/loo-narrow-kalpha", "loo": float(r3["loo_honest"]), "chosen": r3["loo_chosen"]})

with open("experiments/screen_iter3_fuse4.json", "w") as fh:
    json.dump({"incumbent": INC_MEAN, "baseline": BASE_MEAN, "results": out}, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter3_fuse4.json")
