#!/usr/bin/env python3
"""Offline probes for decision-rule / gating changes (exploration only).

A. per-fold component diagnostics
B. per-class multiplicative decision bias, LOO-selected
C. confidence-based per-point gating between the two components, LOO-selected
D. LightGBM seed determinism check (does random_state change the fit?)
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
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
    tab = np.load("outputs/tab_oof.npy")
    cnn = np.load("outputs/cnn_oof.npy")
    ufolds = sorted(set(folds.tolist()))

    # ---- A. per-fold component diagnostics ----
    print("=== A. per-fold component F1 ===")
    print(f"{'fold':<6}{'n':<7}{'tab':<9}{'cnn':<9}{'blend0.4':<10}{'best w':<9}{'best F1':<9}")
    for f in ufolds:
        m = folds == f
        ws = np.arange(0.0, 1.01, 0.05)
        fs = []
        for w in ws:
            p = w * tab + (1 - w) * cnn
            fs.append(wf1(y[m], p[m].argmax(1)))
        bi = int(np.argmax(fs))
        print(f"{f:<6}{int(m.sum()):<7}{wf1(y[m], tab[m].argmax(1)):<9.4f}{wf1(y[m], cnn[m].argmax(1)):<9.4f}"
              f"{wf1(y[m], (0.4*tab+0.6*cnn)[m].argmax(1)):<10.4f}{ws[bi]:<9.2f}{fs[bi]:<9.4f}")

    print("\n=== A2. per-class F1 (pre-Bayes OOF, blend 0.4/0.6) ===")
    p40 = 0.4 * tab + 0.6 * cnn
    from sklearn.metrics import precision_recall_fscore_support
    for name, mat in (("tab", tab), ("cnn", cnn), ("blend", p40)):
        pr, rc, f1, sup = precision_recall_fscore_support(y, mat.argmax(1), labels=range(5), zero_division=0)
        print(f"  {name:<6}" + "  ".join(f"{c}: P={pr[i]:.2f} R={rc[i]:.2f} F1={f1[i]:.2f}(n={sup[i]})" for i, c in enumerate(CLS)))

    # ---- B. per-class multiplicative bias, LOO-selected ----
    print("\n=== B. per-class bias argmax(p_c * b_c): grid 1.0-2.0 for SGAM/SGZ, 0.7-1.3 others ===")
    S = np.arange(1.0, 2.01, 0.1)
    O = np.arange(0.7, 1.31, 0.1)
    cands = []
    for b_sgam in S:
        for b_sgz in S:
            for b_fmat in O:
                for b_nvb in O:
                    if b_fmat < 1.0 and b_nvb < 1.0:
                        pass
                    cands.append((b_fmat, b_nvb, b_sgam, b_sgz))
    print(f"  grid size = {len(cands)}")

    def apply_bias(p, b):
        bb = np.array([1.0, b[0], b[1], b[2], b[3]])
        return (p * bb).argmax(1)

    # full-OOF optimum (exploration upper bound)
    best = None
    for b in cands:
        s = fold_mean(y, apply_bias(p40, b), folds)
        if best is None or s > best[0]:
            best = (s, b)
    print(f"  full-OOF optimum: fold-mean={best[0]:.4f}  b=[ALG=1, FMAT={best[1][0]:.1f}, NVB={best[1][1]:.1f}, "
          f"SGAM={best[1][2]:.1f}, SGZ={best[1][3]:.1f}]")
    print(f"  baseline (b=1):   fold-mean={fold_mean(y, p40.argmax(1), folds):.4f}")

    loo_pf, chosen = [], []
    for f in ufolds:
        m = folds != f
        bb, bs = None, -1
        for b in cands:
            s = wf1(y[m], apply_bias(p40, b)[m])
            if s > bs:
                bs, bb = s, b
        loo_pf.append(wf1(y[folds == f], apply_bias(p40, bb)[folds == f]))
        chosen.append(bb)
    print(f"  LOO-selected: fold-mean={np.mean(loo_pf):.4f}  per-fold={[round(v,4) for v in loo_pf]}")
    print(f"  LOO chosen: {Counter([tuple(np.round(c,2)) for c in chosen]).most_common(5)}")

    # ---- C. confidence gating ----
    print("\n=== C. per-point confidence gating: w_tab(x) = clip(c0 + c1*(mt-mc)/T) ===")
    mt = tab.max(1)
    mc = cnn.max(1)
    d = mt - mc
    print(f"  corr(mt-mc, w_tab*) per-fold... d stats: mean={d.mean():.3f} std={d.std():.3f}")
    loo_pf2, chosen2 = [], []
    grid = [(c0, c1) for c0 in [0.2, 0.3, 0.4, 0.5, 0.6] for c1 in [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]]
    for f in ufolds:
        m = folds != f
        bb, bs = None, -1
        for c0, c1 in grid:
            w = np.clip(c0 + c1 * d, 0.0, 1.0)
            p = w[:, None] * tab + (1 - w[:, None]) * cnn
            s = wf1(y[m], p[m].argmax(1))
            if s > bs:
                bs, bb = s, (c0, c1)
        c0, c1 = bb
        w = np.clip(c0 + c1 * d, 0.0, 1.0)
        p = w[:, None] * tab + (1 - w[:, None]) * cnn
        loo_pf2.append(wf1(y[folds == f], p[folds == f].argmax(1)))
        chosen2.append(bb)
    print(f"  LOO-selected: fold-mean={np.mean(loo_pf2):.4f}  per-fold={[round(v,4) for v in loo_pf2]}")
    print(f"  LOO chosen: {Counter(chosen2).most_common(5)}")

    # oracle per-point gate (upper bound, not achievable)
    print("\n=== C2. sanity: agreement structure ===")
    ag = tab.argmax(1) == cnn.argmax(1)
    print(f"  components agree on {ag.mean()*100:.1f}% of points; "
          f"F1 on agreeing subset (blend)={wf1(y[ag], p40[ag].argmax(1)):.4f}, disagreeing={wf1(y[~ag], p40[~ag].argmax(1)):.4f}")
    print(f"  on disagree subset: tab F1={wf1(y[~ag], tab[~ag].argmax(1)):.4f}  cnn F1={wf1(y[~ag], cnn[~ag].argmax(1)):.4f}")


if __name__ == "__main__":
    main()
