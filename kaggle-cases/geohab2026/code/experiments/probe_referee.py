#!/usr/bin/env python3
"""REF-02 — leave-one-fold-out referee for fusion-operator selection (exploration only).

Implements the OFFICIAL weighted-F1 definition copied from /paper_data/eval/score_oof.py
(support-weighted, zero-support classes skipped) so that re-derived numbers are directly
comparable to the scored pipeline output. Cross-checks itself against the frozen baseline
F1_fold_mean = 0.7831250711050196 before any candidate is trusted.

Protocol
--------
* in_sample : parameters chosen by maximising the full 6256-point OOF  -> optimistic bound.
* loo       : for each held-out fold f, parameters chosen on the other four folds only,
              then applied to f. This is the only number that may be reported as a gain.

Usage:
    python experiments/probe_referee.py [--frozen DIR]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

CLASSES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]


# ── official metric: copied verbatim in spirit from eval/score_oof.py ─────────
def weighted_f1(y_true, y_pred) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    num = 0.0
    for c in range(len(CLASSES)):
        tp = fp = fn = 0
        for t, p in zip(y_true, y_pred):
            if t == c and p == c:
                tp += 1
            elif p == c:
                fp += 1
            elif t == c:
                fn += 1
        support = tp + fn
        if support == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / support
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        num += support * f1
    return num / n


def wf1_fast(y_true, y_pred) -> float:
    """Vectorised equivalent of weighted_f1 (verified against it in --selftest)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    num = 0.0
    for c in range(len(CLASSES)):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        support = tp + fn
        if support == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / support
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        num += support * f1
    return num / n


def fold_vec(y, pred, folds):
    return np.array([wf1_fast(y[folds == f], pred[folds == f]) for f in range(len(set(folds.tolist())))])


def fold_mean(y, pred, folds) -> float:
    return float(fold_vec(y, pred, folds).mean())


# ── per-class P/R/F1 table ───────────────────────────────────────────────────
def per_class_table(y, pred, folds):
    out = {}
    for f in range(len(set(folds.tolist()))):
        m = folds == f
        rows = {}
        for c in range(len(CLASSES)):
            yt, yp = y[m], pred[m]
            tp = int(np.sum((yt == c) & (yp == c)))
            fp = int(np.sum((yp == c) & (yt != c)))
            fn = int(np.sum((yt == c) & (yp != c)))
            sup = tp + fn
            if sup == 0:
                rows[CLASSES[c]] = {"support": 0}
                continue
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / sup
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            rows[CLASSES[c]] = {"support": sup, "P": round(prec, 4), "R": round(rec, 4), "F1": round(f1, 4)}
        out[f] = rows
    return out


# ── the referee ──────────────────────────────────────────────────────────────
def referee(y, folds, candidates, mask_ok=None):
    """candidates: list of (label, preds) with preds an int array of length n (or None to skip).

    Returns dict with in_sample selection, LOO-honest fold-mean, per-fold deltas vs baseline.
    """
    ufolds = sorted(set(folds.tolist()))
    cands = [(lab, p) for lab, p in candidates if p is not None]
    full_scores = np.array([wf1_fast(y, p) for _, p in cands])
    best_full = int(np.argmax(full_scores))

    loo_per_fold, chosen = [], []
    for f in ufolds:
        tr = folds != f
        scores = np.array([wf1_fast(y[tr], p[tr]) for _, p in cands])
        if mask_ok is not None:
            for i, (lab, _) in enumerate(cands):
                if not mask_ok(f, lab):
                    scores[i] = -1.0
        b = int(np.argmax(scores))
        chosen.append(cands[b][0])
        loo_per_fold.append(wf1_fast(y[folds == f], cands[b][1][folds == f]))
    return {
        "in_sample_best_label": cands[best_full][0],
        "in_sample_best_score": float(full_scores[best_full]),
        "in_sample_best_preds": cands[best_full][1],
        "loo_honest": float(np.mean(loo_per_fold)),
        "loo_per_fold": [float(v) for v in loo_per_fold],
        "loo_chosen": chosen,
    }


def load(frozen="/autosota_cache/frozen9002", paper_data="/paper_data"):
    tr = pd.read_csv(f"{paper_data}/train.csv")
    folds = (pd.read_csv(f"{paper_data}/features/fold_indices.csv")
             .sort_values("point_index")["fold"].to_numpy().astype(int))
    y = np.array([CLASSES.index(c) for c in tr["class"]])
    return {
        "y": y, "folds": folds, "df": tr,
        "tab": np.load(f"{frozen}/tab_oof.npy"),
        "cnn": np.load(f"{frozen}/cnn_oof.npy"),
        "blend": np.load(f"{frozen}/blend_oof.npy"),
    }


def selftest(D):
    n = len(D["y"])
    rng = np.random.default_rng(0)
    fake = rng.integers(0, 5, size=n)
    a = weighted_f1(D["y"].tolist(), fake.tolist())
    b = wf1_fast(D["y"], fake)
    assert abs(a - b) < 1e-12, (a, b)
    base = fold_mean(D["y"], D["blend"].argmax(1), D["folds"])
    assert abs(base - 0.7831250711050196) < 1e-12, base
    print(f"[selftest] weighted_f1 implementations agree; frozen blend fold-mean={base:.16f} OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="/autosota_cache/frozen9002")
    ap.add_argument("--paper-data", default="/paper_data")
    a = ap.parse_args()
    D = load(a.frozen, a.paper_data)
    selftest(D)
    y, folds = D["y"], D["folds"]
    print("\nfold sizes:", [int((folds == f).sum()) for f in range(5)])
    print("\nper-fold class support:")
    for f in range(5):
        m = folds == f
        print(f"  fold {f}: " + ", ".join(f"{c}={int((y[m]==i).sum())}" for i, c in enumerate(CLASSES)))
    print("\nfrozen per-fold F1  tab / cnn / blend:")
    for name in ("tab", "cnn", "blend"):
        fv = fold_vec(y, D[name].argmax(1), folds)
        print(f"  {name:6s} {np.round(fv,4).tolist()}  mean={fv.mean():.4f}")
