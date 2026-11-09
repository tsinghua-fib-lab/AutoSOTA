#!/usr/bin/env python3
"""Debug 2: is the pipeline's blend_oof.npy consistent with its own reported 0.7877,
and how does it differ from the offline reconstruction of the same operator?"""
import sys

import numpy as np

sys.path.insert(0, "experiments")
from probe_referee import fold_vec, load  # noqa: E402

D = load()
Y, FOLDS = D["y"], D["folds"]
TAB = np.load("outputs/iter1/tab_oof.npy")
CNN = np.load("outputs/iter1/cnn_oof.npy")
PIPE = np.load("outputs/iter1/blend_oof.npy")

print("pipeline blend_oof per-fold F1 :", np.round(fold_vec(Y, PIPE.argmax(1), FOLDS), 4).tolist(),
      f"mean={fold_vec(Y, PIPE.argmax(1), FOLDS).mean():.6f}")


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


OFF = np.zeros_like(PIPE)
for f in range(5):
    m = FOLDS == f
    tr = FOLDS != f
    sh = np.full(5, 0.5)
    for c in range(5):
        a = cls_f1(Y[tr], TAB.argmax(1)[tr], c)
        b = cls_f1(Y[tr], CNN.argmax(1)[tr], c)
        if a is None or b is None:
            continue
        sh[c] = (a + 0.025) / (a + b + 0.05)
    ts, cs = sharpen(TAB[m], 6.0), sharpen(CNN[m], 1.0)
    p = (0.4 * sh)[None, :] * ts + (0.6 * (1 - sh))[None, :] * cs
    OFF[m] = p / p.sum(1, keepdims=True)

print("offline reconstruction  per-fold:", np.round(fold_vec(Y, OFF.argmax(1), FOLDS), 4).tolist(),
      f"mean={fold_vec(Y, OFF.argmax(1), FOLDS).mean():.6f}")
print("argmax disagreements pipe vs offline:", int((PIPE.argmax(1) != OFF.argmax(1)).sum()))
print("max |pipe-offline| over all cells   :", float(np.abs(PIPE - OFF).max()))

# is the pipeline array the *scalar* pool result? check basis vectors
SC = 0.4 * sharpen(TAB, 6.0) + 0.6 * CNN
SC /= SC.sum(1, keepdims=True)
print("argmax disagreements pipe vs scalar :", int((PIPE.argmax(1) != SC.argmax(1)).sum()))

# where do pipe and offline disagree on argmax, per fold
d = PIPE.argmax(1) != OFF.argmax(1)
for f in range(5):
    print(f"  fold {f}: {int(d[FOLDS == f].sum())} disagreeing rows")
