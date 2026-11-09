#!/usr/bin/env python3
"""Offline probe of the fusion layer (exploration only, no model training).

Reads the probability matrices written by pipeline/run_pipeline.py and measures
how much of the fold-mean weighted F1 is reachable by changing only the fusion
operator.  Two protocols are reported for every candidate:

  full   : parameters chosen by maximising the full 6256-point OOF  -> optimistic,
           reported ONLY as an upper bound for exploration.
  loo    : leave-one-fold-out honest protocol -- for each held-out fold f the
           parameters are chosen on the other four folds and then applied to f.
           This is the number that may legitimately be reported as a fold-mean.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CLS = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]


def load(paper_data="/paper_data", out_dir="outputs"):
    tr = pd.read_csv(f"{paper_data}/train.csv")
    folds = pd.read_csv(f"{paper_data}/features/fold_indices.csv").sort_values("point_index")["fold"].to_numpy()
    y = np.array([CLS.index(c) for c in tr["class"]])
    tab = np.load(f"{out_dir}/tab_oof.npy")
    cnn = np.load(f"{out_dir}/cnn_oof.npy")
    return y, folds, tab, cnn


def wf1(y, pred):
    return float(f1_score(y, pred, average="weighted"))


def fold_mean(y, pred, folds):
    return float(np.mean([wf1(y[folds == f], pred[folds == f]) for f in sorted(set(folds.tolist()))]))


def fold_vec(y, pred, folds):
    return [round(wf1(y[folds == f], pred[folds == f]), 4) for f in sorted(set(folds.tolist()))]


def loo_select(y, folds, cand_fn, score_fn):
    """cand_fn(fold) -> list of (label, param, prob_matrix). Returns honest fold-mean."""
    ufolds = sorted(set(folds.tolist()))
    per_fold, chosen = [], []
    for f in ufolds:
        train_mask = folds != f
        best, best_score = None, -1.0
        for label, param, probs in cand_fn(f):
            s = wf1(y[train_mask], probs[train_mask].argmax(1))
            if s > best_score:
                best_score, best = s, (label, param, probs)
        per_fold.append(wf1(y[folds == f], best[2][folds == f].argmax(1)))
        chosen.append(best[0])
    return float(np.mean(per_fold)), per_fold, chosen


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    y, folds, tab, cnn = load(out_dir=out_dir)
    print(f"[init] n={len(y)}  tab fold-mean={fold_mean(y, tab.argmax(1), folds):.4f} "
          f" overall={wf1(y, tab.argmax(1)):.4f}")
    print(f"[init]                cnn fold-mean={fold_mean(y, cnn.argmax(1), folds):.4f} "
          f" overall={wf1(y, cnn.argmax(1)):.4f}")

    # ---- 1. linear weight sweep -------------------------------------------
    print("\n[1] linear blend  w_tab in 0.05..0.95")
    rows = []
    for w in np.arange(0.05, 0.96, 0.05):
        p = w * tab + (1 - w) * cnn
        fm = fold_mean(y, p.argmax(1), folds)
        rows.append((round(float(w), 2), fm, fold_vec(y, p.argmax(1), folds)))
    for w, fm, fv in rows:
        star = " <-- current 0.40" if abs(w - 0.40) < 1e-9 else (" *" if fm > 0.7688 else "")
        print(f"   w_tab={w:.2f}  fold-mean={fm:.4f}  per-fold={fv}{star}")

    def cand_lin(_f):
        out = []
        for w in np.arange(0.05, 0.96, 0.05):
            out.append((round(float(w), 2), round(float(w), 2), w * tab + (1 - w) * cnn))
        return out

    lm, pf, ch = loo_select(y, folds, cand_lin, None)
    print(f"   LOO-selected linear: fold-mean={lm:.4f}  per-fold={[round(v,4) for v in pf]}  chosen={ch}")

    # ---- 2. geometric pooling ---------------------------------------------
    print("\n[2] geometric pool  p ~ tab^a * cnn^b   (a,b grid)")
    eps = 1e-12
    lt, lc = np.log(tab + eps), np.log(cnn + eps)
    best = []
    for a in [0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]:
        for b in [0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]:
            p = np.exp(a * lt + b * lc)
            fm = fold_mean(y, p.argmax(1), folds)
            best.append((fm, a, b))
    best.sort(reverse=True)
    for fm, a, b in best[:8]:
        print(f"   a={a:<5} b={b:<5} fold-mean={fm:.4f}")

    def cand_geo(_f):
        out = []
        for a in [0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]:
            for b in [0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]:
                out.append((f"a={a},b={b}", (a, b), np.exp(a * lt + b * lc)))
        return out

    gm, pf, ch = loo_select(y, folds, cand_geo, None)
    print(f"   LOO-selected geometric: fold-mean={gm:.4f}  per-fold={[round(v,4) for v in pf]}")
    from collections import Counter
    print(f"   LOO chosen: {Counter(ch).most_common(5)}")

    # reference: the fixed 0.40 blend
    print(f"\n[ref] fixed (0.40,0.60) fold-mean={fold_mean(y, (0.4 * tab + 0.6 * cnn).argmax(1), folds):.4f}")


if __name__ == "__main__":
    main()
