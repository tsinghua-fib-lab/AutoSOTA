#!/usr/bin/env python3
"""Probe 10: per-fold detail for the sharpening plateau, to pick a defensible point."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CLS = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
REF_FOLDS = [0.7857, 0.7788, 0.8246, 0.7602, 0.7430]   # iter-1 (a=1)


def wf1(y, pred):
    return float(f1_score(y, pred, average="weighted"))


def fold_mean(y, pred, folds):
    return float(np.mean([wf1(y[folds == f], pred[folds == f]) for f in sorted(set(folds.tolist()))]))


def fold_vec(y, pred, folds):
    return [round(wf1(y[folds == f], pred[folds == f]), 4) for f in sorted(set(folds.tolist()))]


def main():
    tr = pd.read_csv("/paper_data/train.csv")
    folds = pd.read_csv("/paper_data/features/fold_indices.csv").sort_values("point_index")["fold"].to_numpy()
    y = np.array([CLS.index(c) for c in tr["class"]])
    tab = np.load("outputs/tab_oof.npy")
    cnn = np.load("outputs/cnn_oof.npy")

    def sharp(p, e):
        q = p ** e
        return q / q.sum(1, keepdims=True)

    print(f"{'a':<8}{'blend':<9}{'delta':<9}per-fold delta vs iter-1        pos")
    cands = [(1.0, tab), (2.0, sharp(tab, 2)), (3.0, sharp(tab, 3)), (4.0, sharp(tab, 4)),
             (5.0, sharp(tab, 5)), (6.0, sharp(tab, 6)), (8.0, sharp(tab, 8)), (12.0, sharp(tab, 12))]
    oh = np.zeros_like(tab)
    oh[np.arange(len(tab)), tab.argmax(1)] = 1.0
    cands.append((float("inf"), oh))
    for a, t in cands:
        p = 0.4 * t + 0.6 * cnn
        fv = fold_vec(y, p.argmax(1), folds)
        dv = [round(x - c, 4) for x, c in zip(fv, REF_FOLDS)]
        npos = sum(1 for d in dv if d > 1e-9)
        tag = "hard-vote" if a == float("inf") else f"{a:g}"
        print(f"{tag:<8}{fold_mean(y, p.argmax(1), folds):<9.4f}"
              f"{fold_mean(y, p.argmax(1), folds)-0.7785:<+9.4f}{str(dv):<32}{npos}/5")

    print("\n=== per-class F1 at a=6 vs a=1 ===")
    from sklearn.metrics import precision_recall_fscore_support as prfs
    for a, t in ((1.0, tab), (6.0, sharp(tab, 6))):
        p = 0.4 * t + 0.6 * cnn
        _, _, f1, sup = prfs(y, p.argmax(1), labels=range(5), zero_division=0)
        print(f"  a={a:<4} " + "  ".join(f"{c}:F1={f1[i]:.3f}" for i, c in enumerate(CLS)))

    print("\n=== sanity: sharpening does NOT change the tabular argmax (only its spread) ===")
    print(f"  argmax(tab) == argmax(tab^6): {(tab.argmax(1) == sharp(tab, 6).argmax(1)).all()}")
    print(f"  tabular-only fold-mean stays: a=1 {fold_mean(y, tab.argmax(1), folds):.4f}  "
          f"a=6 {fold_mean(y, sharp(tab, 6).argmax(1), folds):.4f}   (guardrail LGB_tabular_F1 unchanged)")


if __name__ == "__main__":
    main()
