#!/usr/bin/env python3
"""Probe 9: component sharpening (probability calibration) in the fixed 0.40 blend.

probe 8 showed a smooth monotone gain from raising the tabular probabilities to a
power a > 1 (a=1.0 -> 0.7785 ... a=3.0 -> 0.7826) and a mild one from smoothing the
CNN (b=0.8).  This probe maps the 2-D surface and, critically, re-checks the gain
under a leave-one-fold-out protocol so the reported number cannot be a
selection artifact.  Also checks the interaction with the blend weight.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CLS = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
REF = 0.7785
A_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0]
B_GRID = [0.6, 0.7, 0.8, 0.9, 1.0]


def wf1(y, pred):
    return float(f1_score(y, pred, average="weighted"))


def fold_mean(y, pred, folds):
    return float(np.mean([wf1(y[folds == f], pred[folds == f]) for f in sorted(set(folds.tolist()))]))


def fold_vec(y, pred, folds):
    return [round(wf1(y[folds == f], pred[folds == f]), 4) for f in sorted(set(folds.tolist()))]


def sharp(p, e):
    q = p ** e
    return q / q.sum(1, keepdims=True)


def main():
    tr = pd.read_csv("/paper_data/train.csv")
    folds = pd.read_csv("/paper_data/features/fold_indices.csv").sort_values("point_index")["fold"].to_numpy()
    y = np.array([CLS.index(c) for c in tr["class"]])
    tab = np.load("outputs/tab_oof.npy")
    cnn = np.load("outputs/cnn_oof.npy")
    uf = sorted(set(folds.tolist()))

    print("=== A. 2-D surface, fixed blend weight 0.40 (b=1.0 row = tab only sharpened) ===")
    print("      " + "".join(f"b={b:<8}" for b in B_GRID))
    for a in A_GRID:
        row = []
        for b in B_GRID:
            p = 0.4 * sharp(tab, a) + 0.6 * sharp(cnn, b)
            row.append(fold_mean(y, p.argmax(1), folds))
        print(f"  a={a:<4}" + "".join(f"{v:<10.4f}" for v in row))

    print("\n=== B. hard-vote limit (a -> inf): 0.4*onehot(tab_argmax) + 0.6*cnn ===")
    oh = np.zeros_like(tab)
    oh[np.arange(len(tab)), tab.argmax(1)] = 1.0
    p = 0.4 * oh + 0.6 * cnn
    print(f"  blend={fold_mean(y, p.argmax(1), folds):.4f}  per-fold={fold_vec(y, p.argmax(1), folds)}")

    print("\n=== C. LOO-honest selection over (a, b) ===")
    pf, ch = [], []
    for f in uf:
        m = folds != f
        bb, bs = None, -1
        for a in A_GRID:
            for b in B_GRID:
                p = 0.4 * sharp(tab, a) + 0.6 * sharp(cnn, b)
                s = wf1(y[m], p[m].argmax(1))
                if s > bs:
                    bs, bb = s, (a, b)
        a, b = bb
        p = 0.4 * sharp(tab, a) + 0.6 * sharp(cnn, b)
        pf.append(wf1(y[folds == f], p[folds == f].argmax(1)))
        ch.append(bb)
    print(f"  LOO fold-mean={np.mean(pf):.4f}  per-fold={[round(v,4) for v in pf]}  chosen={ch}")
    print(f"  delta vs iter-1 reference {REF}: {np.mean(pf)-REF:+.4f}")

    print("\n=== D. LOO-honest selection over a only (b fixed at 1.0, the reference CNN) ===")
    pf, ch = [], []
    for f in uf:
        m = folds != f
        bb, bs = None, -1
        for a in A_GRID:
            p = 0.4 * sharp(tab, a) + 0.6 * cnn
            s = wf1(y[m], p[m].argmax(1))
            if s > bs:
                bs, bb = s, a
        p = 0.4 * sharp(tab, bb) + 0.6 * cnn
        pf.append(wf1(y[folds == f], p[folds == f].argmax(1)))
        ch.append(bb)
    print(f"  LOO fold-mean={np.mean(pf):.4f}  per-fold={[round(v,4) for v in pf]}  chosen={ch}")

    print("\n=== E. interaction with the blend weight at a=3 (fixed) ===")
    for w in (0.30, 0.35, 0.40, 0.45, 0.50):
        p = w * sharp(tab, 3.0) + (1 - w) * cnn
        print(f"  w_tab={w:.2f} -> {fold_mean(y, p.argmax(1), folds):.4f}")
    print(f"  (published w=0.40 at a=1.0 -> {REF})")


if __name__ == "__main__":
    main()
