#!/usr/bin/env python3
"""Probe 4: spatial structure + transductive smoothing of fold-model probabilities.

Leakage argument
----------------
For a point p in fold f the blend probability is produced by models that never
saw fold f's labels.  Averaging p's probability with the probabilities of OTHER
points in the SAME fold f therefore uses only out-of-fold predictions of a single
model -> no validation label enters the smoothed value.  Labels are never
touched anywhere below.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score

CLS = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]


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
    xy = tr[["x", "y"]].to_numpy()
    tab = np.load("outputs/tab_oof.npy")
    cnn = np.load("outputs/cnn_oof.npy")
    p40 = 0.4 * tab + 0.6 * cnn

    print("=== spatial structure ===")
    uf = sorted(set(folds.tolist()))
    tree = cKDTree(xy)
    for f in uf:
        m = folds == f
        idx = np.where(m)[0]
        d, _ = tree.query(xy[idx], k=2)
        print(f"  fold {f}: n={m.sum():<5} extent x[{xy[idx,0].min():.0f},{xy[idx,0].max():.0f}] "
              f"y[{xy[idx,1].min():.0f},{xy[idx,1].max():.0f}]  median NN dist={np.median(d[:,1]):.2f} m")
    d_all, nn_all = tree.query(xy, k=2)
    same = folds[nn_all[:, 1]] == folds
    lab_agree = y[nn_all[:, 1]] == y
    print(f"  nearest neighbour in same fold: {same.mean()*100:.1f}%")
    print(f"  nearest neighbour has same label: {lab_agree.mean()*100:.1f}% "
          f"(chance ~ {max(np.bincount(y))/len(y)*100:.1f}%)")
    for f in uf:
        m = folds == f
        idx = np.where(m)[0]
        dd, nn = tree.query(xy[idx], k=2)
        print(f"  fold {f}: NN label agreement {np.mean(y[nn[:,1]]==y[idx])*100:.1f}%")

    print("\n=== smoothing: p' = (1-a)p + a * mean(p over k same-fold NN), within-fold only ===")
    print(f"{'k':<6}{'a':<7}{'fold-mean':<11}per-fold")
    best = None
    for k in (3, 5, 8, 12, 20, 35):
        for a in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
            sm = p40.copy()
            for f in uf:
                idx = np.where(folds == f)[0]
                _, nn = cKDTree(xy[idx]).query(xy[idx], k=min(k + 1, len(idx)))
                nb = idx[nn[:, 1:]]                       # k nearest same-fold neighbours
                sm[idx] = (1 - a) * p40[idx] + a * p40[nb].mean(1)
            fm = fold_mean(y, sm.argmax(1), folds)
            if best is None or fm > best[0]:
                best = (fm, k, a, fold_vec(y, sm.argmax(1), folds))
            print(f"{k:<6}{a:<7}{fm:<11.4f}{fold_vec(y, sm.argmax(1), folds)}")
    print(f"\n  best fixed (k,a) = ({best[1]},{best[2]}) -> {best[0]:.4f} {best[3]}")

    print("\n=== LOO-honest selection of (k,a) ===")
    pf, ch = [], []
    for f in uf:
        m = folds != f
        bb, bs = None, -1
        for k in (3, 5, 8, 12, 20, 35):
            for a in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
                sm = p40.copy()
                for g in uf:
                    idx = np.where(folds == g)[0]
                    _, nn = cKDTree(xy[idx]).query(xy[idx], k=min(k + 1, len(idx)))
                    sm[idx] = (1 - a) * p40[idx] + a * p40[idx[nn[:, 1:]]].mean(1)
                s = wf1(y[m], sm[m].argmax(1))
                if s > bs:
                    bs, bb = s, (k, a)
        k, a = bb
        sm = p40.copy()
        for g in uf:
            idx = np.where(folds == g)[0]
            _, nn = cKDTree(xy[idx]).query(xy[idx], k=min(k + 1, len(idx)))
            sm[idx] = (1 - a) * p40[idx] + a * p40[idx[nn[:, 1:]]].mean(1)
        pf.append(wf1(y[folds == f], sm[folds == f].argmax(1)))
        ch.append(bb)
    print(f"  LOO fold-mean={np.mean(pf):.4f}  per-fold={[round(v,4) for v in pf]}  chosen={ch}")


if __name__ == "__main__":
    main()
