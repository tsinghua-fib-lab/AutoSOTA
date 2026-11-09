#!/usr/bin/env python3
"""Debug: recompute the FUSE-02 operator from frozen vs freshly-trained component matrices."""
import sys

import numpy as np

sys.path.insert(0, "experiments")
from probe_referee import fold_vec, load  # noqa: E402

D = load()
Y, FOLDS = D["y"], D["folds"]
BASE = fold_vec(Y, D["blend"].argmax(1), FOLDS)
print("baseline blend per-fold:", np.round(BASE, 4).tolist(), f"mean={BASE.mean():.6f}")


def sharpen(p, e):
    q = np.power(p, e)
    return q / q.sum(1, keepdims=True)


def cls_f1(y, pr, c):
    tp = int(((y == c) & (pr == c)).sum())
    fp = int(((pr == c) & (y != c)).sum())
    fn = int(((y == c) & (pr != c)).sum())
    s = tp + fn
    if s == 0:
        return None
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / s
    return 2 * p * r / (p + r) if p + r else 0.0


for tag, tpath in [("frozen", "/autosota_cache/frozen9002/tab_oof.npy"),
                   ("iter1 ", "outputs/iter1/tab_oof.npy")]:
    TAB = np.load(tpath)
    CNN = np.load("outputs/iter1/cnn_oof.npy")
    pred = np.zeros(len(Y), dtype=int)
    shares = {}
    for f in range(5):
        m = FOLDS != f
        sh = np.full(5, 0.5)
        for c in range(5):
            a = cls_f1(Y[m], TAB.argmax(1)[m], c)
            b = cls_f1(Y[m], CNN.argmax(1)[m], c)
            if a is None or b is None:
                continue
            sh[c] = (a + 0.025) / (a + b + 0.05)
        shares[f] = sh
        ts, cs = sharpen(TAB[m], 6.0), sharpen(CNN[m], 1.0)
        p = (0.4 * sh)[None, :] * ts + (0.6 * (1 - sh))[None, :] * cs
        pred[m] = p.argmax(1)
    fv = fold_vec(Y, pred, FOLDS)
    print(f"{tag}: mean={fv.mean():.6f} per-fold={np.round(fv, 4).tolist()}")
    print(f"        fold0 shares={np.round(shares[0], 4).tolist()}")

# raw tab argmax agreement between the two matrices
A = np.load("/autosota_cache/frozen9002/tab_oof.npy")
B = np.load("outputs/iter1/tab_oof.npy")
print("tab argmax disagreements frozen vs iter1:", int((A.argmax(1) != B.argmax(1)).sum()))
top2gap = np.sort(A, axis=1)[:, -1] - np.sort(A, axis=1)[:, -2]
print("smallest top-2 gap in frozen tab:", float(top2gap.min()))
