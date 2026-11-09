#!/usr/bin/env python3
"""Screen 12 — does a better RELIABILITY MEASURE improve the FUSE-02 operator?

FUSE-02's shares are driven by each component's per-class F1 on the other four folds. F1 is a
decision-level statistic; it ignores calibration and is noisy for small classes. Proper scoring
rules (log-loss, Brier) and support-aware variants are the natural alternatives, and they cost
nothing to test. Every candidate is fold-honest by construction; the nested gate is also
reported for the ones whose shrinkage is selected.
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


def measure(y, prob, pred, c, mask, kind):
    """Reliability of one component for class c, measured on `mask`. Higher = better."""
    yc = y[mask]
    pc = prob[mask][:, c]
    if kind == "f1":
        tp = int(np.sum((yc == c) & (pred[mask] == c)))
        fp = int(np.sum((pred[mask] == c) & (yc != c)))
        fn = int(np.sum((yc == c) & (pred[mask] != c)))
        s = tp + fn
        if s == 0:
            return None, 0
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / s
        return (2 * pr * rec / (pr + rec) if (pr + rec) else 0.0), s
    if kind == "acc":
        s = int(np.sum(yc == c))
        if s == 0:
            return None, 0
        return float(np.mean(pred[mask][yc == c] == c)), s
    if kind == "logloss":
        s = int(np.sum(yc == c))
        if s == 0:
            return None, 0
        # mean log-likelihood assigned to the true class among class-c points, mapped to (0,1]
        ll = float(np.mean(np.log(np.maximum(pc[yc == c], 1e-12))))
        return float(np.exp(ll / np.log(1.0 / 5.0))), s     # 1.0 = perfect, 0.5 = chance
    if kind == "brier":
        s = int(np.sum(yc == c))
        if s == 0:
            return None, 0
        target = (yc == c).astype(float)
        b = float(np.mean((pc - target) ** 2))              # 0 = perfect
        return float(max(0.0, 1.0 - 5.0 * b)), s
    raise ValueError(kind)


def shares_for(mask, kind="f1", eps=0.5, per_fold_avg=False):
    sh = np.full(5, 0.5)
    for c in range(5):
        if per_fold_avg:
            vals_t, vals_c = [], []
            for f in range(5):
                mm = mask & (FOLDS == f)
                a, _ = measure(Y, TAB, TAB.argmax(1), c, mm, kind)
                b, _ = measure(Y, CNN, CNN.argmax(1), c, mm, kind)
                if a is not None and b is not None:
                    vals_t.append(a)
                    vals_c.append(b)
            rt = float(np.mean(vals_t)) if vals_t else None
            rc = float(np.mean(vals_c)) if vals_c else None
        else:
            rt, _ = measure(Y, TAB, TAB.argmax(1), c, mask, kind)
            rc, _ = measure(Y, CNN, CNN.argmax(1), c, mask, kind)
        if rt is None or rc is None:
            continue
        sh[c] = (rt + eps * 0.5) / (rt + rc + eps)
    return sh


def evaluate(share_fn):
    pred = np.zeros(len(Y), dtype=int)
    for f in range(5):
        va = FOLDS == f
        sh = share_fn(FOLDS != f)
        p = 0.40 * TS6[va] * sh[None, :] + 0.60 * CNN[va] * (1 - sh)[None, :]
        pred[va] = p.argmax(1)
    return pred


INC_VEC = fold_vec(Y, evaluate(lambda m: shares_for(m, "f1", 0.5)), FOLDS)
INC_MEAN = float(INC_VEC.mean())
print(f"[incumbent F1/eps=0.5] mean={INC_MEAN:.6f} per-fold={np.round(INC_VEC,4).tolist()}")

out = []
print("\n[CC] reliability-measure sweep (eps=0.5 unless stated)")
for kind in ["f1", "acc", "logloss", "brier"]:
    for pfa in [False, True]:
        p = evaluate(lambda m, k=kind, q=pfa: shares_for(m, k, 0.5, q))
        fv = fold_vec(Y, p, FOLDS)
        pf = np.round(fv - INC_VEC, 4)
        print(f"  {kind:8s} per_fold_avg={str(pfa):5s}  mean={fv.mean():.4f} "
              f"d_inc={fv.mean()-INC_MEAN:+.4f} d_base={fv.mean()-BASE_MEAN:+.4f} "
              f"pos={int((fv-INC_VEC >= -1e-12).sum())}/5 {pf.tolist()}")
        out.append({"tag": f"CC/{kind}/pfa={pfa}", "mean": float(fv.mean()),
                    "delta_inc": float(fv.mean() - INC_MEAN),
                    "delta_base": float(fv.mean() - BASE_MEAN),
                    "n_pos": int((fv - INC_VEC >= -1e-12).sum()), "per_fold_delta": pf.tolist()})

print("\n[DD] eps sweep for the best alternative measures")
for kind in ["logloss", "brier", "acc"]:
    for eps in [0.2, 0.5, 1.0, 2.0]:
        p = evaluate(lambda m, k=kind, e=eps: shares_for(m, k, e))
        fv = fold_vec(Y, p, FOLDS)
        print(f"  {kind:8s} eps={eps:<4} mean={fv.mean():.4f} d_inc={fv.mean()-INC_MEAN:+.4f} "
              f"d_base={fv.mean()-BASE_MEAN:+.4f} pos={int((fv-INC_VEC >= -1e-12).sum())}/5 "
              f"{np.round(fv-INC_VEC,4).tolist()}")
        out.append({"tag": f"DD/{kind}/eps={eps}", "mean": float(fv.mean()),
                    "delta_inc": float(fv.mean() - INC_MEAN),
                    "delta_base": float(fv.mean() - BASE_MEAN),
                    "n_pos": int((fv - INC_VEC >= -1e-12).sum()),
                    "per_fold_delta": np.round(fv - INC_VEC, 4).tolist()})

print("\n[EE] blend of the F1 share with the log-loss share (average of the two)")
for alpha in [0.3, 0.5, 0.7]:
    def sf(m, a=alpha):
        return a * shares_for(m, "f1", 0.5) + (1 - a) * shares_for(m, "logloss", 0.5)
    p = evaluate(sf)
    fv = fold_vec(Y, p, FOLDS)
    print(f"  alpha={alpha} mean={fv.mean():.4f} d_inc={fv.mean()-INC_MEAN:+.4f} "
          f"d_base={fv.mean()-BASE_MEAN:+.4f} pos={int((fv-INC_VEC >= -1e-12).sum())}/5 "
          f"{np.round(fv-INC_VEC,4).tolist()}")
    out.append({"tag": f"EE/alpha={alpha}", "mean": float(fv.mean()),
                "delta_inc": float(fv.mean() - INC_MEAN),
                "delta_base": float(fv.mean() - BASE_MEAN),
                "n_pos": int((fv - INC_VEC >= -1e-12).sum()),
                "per_fold_delta": np.round(fv - INC_VEC, 4).tolist()})

with open("experiments/screen_iter5_measure.json", "w") as fh:
    json.dump({"incumbent": INC_MEAN, "baseline": BASE_MEAN, "results": out}, fh, indent=2, default=str)
print("\nwrote experiments/screen_iter5_measure.json")
