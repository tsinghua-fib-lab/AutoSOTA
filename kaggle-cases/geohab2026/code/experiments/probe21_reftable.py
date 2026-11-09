#!/usr/bin/env python3
"""REF-01 — frozen per-fold x per-class reference table (tab / cnn / blend / FUSE-02).

Turns the single number 0.7831 into a diagnosable table: which class, in which fold, is the
blend actually losing, and to which component.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "experiments")
from probe_referee import fold_vec, load, selftest  # noqa: E402

D = load()
selftest(D)
Y, FOLDS = D["y"], D["folds"]
TAB, CNN = D["tab"], D["cnn"]
CLASSES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]


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
    return {"P": pr, "R": rec, "F1": 2 * pr * rec / (pr + rec) if (pr + rec) else 0.0, "n": s}


def shares_for(mask, eps=0.5):
    sh = np.full(5, 0.5)
    for c in range(5):
        rt = cls_prf(Y[mask], TAB.argmax(1)[mask], c)
        rc = cls_prf(Y[mask], CNN.argmax(1)[mask], c)
        if rt is None or rc is None:
            continue
        sh[c] = (rt["F1"] + eps * 0.5) / (rt["F1"] + rc["F1"] + eps)
    return sh


F02 = np.zeros(len(Y), dtype=int)
for f in range(5):
    va = FOLDS == f
    sh = shares_for(FOLDS != f)
    F02[va] = (0.40 * TS6[va] * sh[None, :] + 0.60 * CNN[va] * (1 - sh)[None, :]).argmax(1)

preds = {"tab": TAB.argmax(1), "cnn": CNN.argmax(1), "blend": D["blend"].argmax(1), "fuse02": F02}
print("component fold-means:")
for k, p in preds.items():
    fv = fold_vec(Y, p, FOLDS)
    print(f"  {k:8s} {np.round(fv,4).tolist()} mean={fv.mean():.4f}")

hdr = f"{'fold':>4} {'class':>5} {'n':>5} | " + " | ".join(f"{k:>22s}" for k in preds)
print("\nper-fold per-class F1 (n = support):")
print(hdr)
rows = {}
for f in range(5):
    m = FOLDS == f
    for c in range(5):
        cells = []
        for k, p in preds.items():
            r = cls_prf(Y[m], p[m], c)
            cells.append("        --            " if r is None else
                         f"P{r['P']:.2f} R{r['R']:.2f} F{r['F1']:.3f} n{r['n']:<4d}")
        n = cls_prf(Y[m], preds["blend"][m], c)
        rows[f"{f}/{CLASSES[c]}"] = cells
        print(f"{f:>4} {CLASSES[c]:>5} {(n['n'] if n else 0):>5} | " + " | ".join(cells))

print("\nbiggest FUSE-02 losses vs the better component, weighted by within-fold support:")
deltas = []
for f in range(5):
    m = FOLDS == f
    for c in range(5):
        rb = cls_prf(Y[m], F02[m], c)
        if rb is None:
            continue
        rt = cls_prf(Y[m], TAB.argmax(1)[m], c)
        rc = cls_prf(Y[m], CNN.argmax(1)[m], c)
        best = max([x["F1"] for x in (rt, rc) if x is not None], default=0.0)
        w = rb["n"] / len(Y)
        deltas.append((w * (best - rb["F1"]), f, CLASSES[c], rb["F1"], best, rb["n"]))
deltas.sort(reverse=True)
for d, f, c, got, best, n in deltas[:12]:
    print(f"  fold{f} {c:>5} n={n:>4}  fuse02 F1={got:.3f} vs best-component F1={best:.3f}  "
          f"weighted gap={d:.4f}")

with open("experiments/ref_table_9002.json", "w") as fh:
    json.dump({"preds": {k: float(fold_vec(Y, p, FOLDS).mean()) for k, p in preds.items()},
               "rows": rows}, fh, indent=2)
print("\nwrote experiments/ref_table_9002.json")
