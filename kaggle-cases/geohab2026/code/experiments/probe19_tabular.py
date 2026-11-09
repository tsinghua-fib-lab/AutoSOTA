#!/usr/bin/env python3
"""Screen 8 — tabular-branch variants (the only component trained live).

Replicates pipeline/run_pipeline.py §3+§5 exactly (same feature assembly, same frozen folds,
same LightGBM call) and then applies the SAME FUSE-02 fusion operator that is now the baseline,
so every variant is compared on the metric that is actually scored.

Guard: the `base` variant must reproduce the pipeline's tabular per-fold F1 bit-for-bit before
any other variant's number is trusted (REF-02 discipline).

Usage: python experiments/probe19_tabular.py [variant ...]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
from probe_referee import fold_vec, load, selftest, wf1_fast  # noqa: E402

PAPER_DATA = "/paper_data"
CLASSES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]

D = load()
selftest(D)
Y, FOLDS = D["y"], D["folds"]
CNN = D["cnn"]
BASE_VEC = fold_vec(Y, D["blend"].argmax(1), FOLDS)
BASE_MEAN = float(BASE_VEC.mean())

# ── §3 feature assembly (identical to the pipeline) ──────────────────────────────
v07_tr = pd.read_csv(f"{PAPER_DATA}/features/train_features_v07.csv")
v07_te = pd.read_csv(f"{PAPER_DATA}/features/test_features_v07.csv")
v09_full_tr = pd.read_csv(f"{PAPER_DATA}/features/train_features_v09.csv")
v09_full_te = pd.read_csv(f"{PAPER_DATA}/features/test_features_v09.csv")
v07_cols = [c for c in v07_tr.columns if c not in ("x", "y", "class")]
assert len(v07_cols) == 86
train_df = pd.read_csv(f"{PAPER_DATA}/train.csv")
assert np.allclose(v07_tr["x"], train_df["x"], atol=1e-4)

v07_te = v07_te.sort_values("ID").reset_index(drop=True)
v09_te_sorted = v09_full_te.sort_values("ID").reset_index(drop=True)
assert (v07_te["ID"].to_numpy() == v09_te_sorted["ID"].to_numpy()).all()


def build_X(extra_acr=False, ratios=False, nanidx=False):
    cols = list(v07_cols)
    tr_parts = [v07_tr[cols].to_numpy()]
    te_parts = [v07_te[cols].to_numpy()]
    if extra_acr:
        for c in ("acr_sapa_w3", "acr_tin_w3"):
            tr_parts.append(v09_full_tr[c].to_numpy()[:, None])
            te_parts.append(v09_te_sorted[c].to_numpy()[:, None])
            cols.append(c)
    tr_parts.append(v09_full_tr["acr_sapa_w21"].to_numpy()[:, None])
    te_parts.append(v09_te_sorted["acr_sapa_w21"].to_numpy()[:, None])
    cols.append("acr_sapa_w21")
    if ratios:
        pairs = [
            ("bathy_focal_std_w41", "bathy_focal_std_w3", "ratio"),
            ("bathy_focal_std_w11", "bathy_focal_std_w3", "ratio"),
            ("bathy_focal_range_w21", "bathy_focal_range_w3", "ratio"),
            ("bathy_focal_mean_w41", "bathy_focal_mean_w11", "diff"),
            ("bs_focal_std_w41", "bs_focal_std_w3", "ratio"),
            ("bs_focal_std_w11", "bs_focal_std_w3", "ratio"),
            ("bs_focal_range_w21", "bs_focal_range_w3", "ratio"),
            ("bs_focal_mean_w41", "bs_focal_mean_w11", "diff"),
        ]
        for a, b, kind in pairs:
            for src_tr, src_te, parts in ((v07_tr, v07_te, (tr_parts, te_parts)),):
                A, B = src_tr[a].to_numpy(), src_tr[b].to_numpy()
                if kind == "ratio":
                    m = np.isfinite(A) & np.isfinite(B) & (np.abs(B) > 1e-12)
                    val = np.where(m, A / np.where(m, B, 1.0), np.nan)
                else:
                    m = np.isfinite(A) & np.isfinite(B)
                    val = np.where(m, A - B, np.nan)
                parts[0].append(val[:, None])
                A, B = src_te[a].to_numpy(), src_te[b].to_numpy()
                if kind == "ratio":
                    m = np.isfinite(A) & np.isfinite(B) & (np.abs(B) > 1e-12)
                    val = np.where(m, A / np.where(m, B, 1.0), np.nan)
                else:
                    m = np.isfinite(A) & np.isfinite(B)
                    val = np.where(m, A - B, np.nan)
                parts[1].append(val[:, None])
            cols.append(f"{a}_{kind}_{b}")
    Xtr = np.column_stack(tr_parts)
    Xte = np.column_stack(te_parts)
    if nanidx:
        ntr = np.isnan(Xtr).sum(1).astype(float)[:, None]
        nte = np.isnan(Xte).sum(1).astype(float)[:, None]
        Xtr = np.column_stack([Xtr, ntr])
        Xte = np.column_stack([Xte, nte])
        cols = cols + ["n_missing"]
    return Xtr, Xte, cols


# ── §5 LightGBM training under a variant config ─────────────────────────────────
LGB_BASE = dict(n_estimators=1000, random_state=42, verbose=-1, n_jobs=8, num_leaves=31,
                max_depth=-1, learning_rate=0.1, min_child_samples=100, feature_fraction=1.0,
                bagging_fraction=1.0, bagging_freq=0, reg_alpha=0.0, reg_lambda=0.0,
                class_weight={0: 2.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})


def train_oof(X, labels, params, categorical=None, seeds=(42,), extra_params=None):
    """Returns the OOF matrix (averaged over seeds if several)."""
    from lightgbm import LGBMClassifier
    oof = np.zeros((len(labels), 5))
    for f in range(5):
        tr, va = FOLDS != f, FOLDS == f
        acc = np.zeros((int(va.sum()), 5))
        for sd in seeds:
            p = dict(params); p["random_state"] = sd
            if extra_params:
                p.update(extra_params)
            kw = {}
            if categorical:
                kw = dict(categorical_feature=categorical) if isinstance(categorical, list) \
                    else dict(categorical_feature=categorical(p))
            clf = LGBMClassifier(**p)
            clf.fit(X[tr], labels[tr], **kw)
            acc += clf.predict_proba(X[va])
        oof[va] = acc / len(seeds)
    return oof


TS_CACHE = {}


def sharpen(p, e):
    q = np.power(p, e)
    return q / q.sum(1, keepdims=True)


def cls_prf(y, pred, c):
    tp = int(np.sum((y == c) & (pred == c)))
    fp = int(np.sum((pred == c) & (y != c)))
    fn = int(np.sum((y == c) & (pred != c)))
    s = tp + fn
    if s == 0:
        return None
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / s
    return {"F1": 2 * pr * rec / (pr + rec) if (pr + rec) else 0.0, "s": s}


def fuse_blend(tab_oof, eps=0.5):
    """The current FUSE-02 operator, applied exactly as the pipeline does."""
    ts6 = sharpen(tab_oof, 6.0)
    pred = np.zeros(len(Y), dtype=int)
    for f in range(5):
        va = FOLDS == f
        sh = np.full(5, 0.5)
        for c in range(5):
            rt = cls_prf(Y[FOLDS != f], tab_oof.argmax(1)[FOLDS != f], c)
            rc = cls_prf(Y[FOLDS != f], CNN.argmax(1)[FOLDS != f], c)
            if rt is None or rc is None:
                continue
            sh[c] = (rt["F1"] + eps * 0.5) / (rt["F1"] + rc["F1"] + eps)
        p = 0.40 * ts6[va] * sh[None, :] + 0.60 * CNN[va] * (1 - sh)[None, :]
        pred[va] = p.argmax(1)
    return pred


def report(tag, tab_oof, t0):
    tf = fold_vec(Y, tab_oof.argmax(1), FOLDS)
    bp = fuse_blend(tab_oof)
    bf = fold_vec(Y, bp, FOLDS)
    print(f"  {tag:34s} tab={tf.mean():.4f} {np.round(tf,4).tolist()} | "
          f"blend={bf.mean():.4f} d_inc={bf.mean()-BASE_MEAN:+.4f} "
          f"pos={int((bf-fold_vec(Y, load()['blend'].argmax(1), FOLDS) >= -1e-12).sum())}/5 "
          f"({time.time()-t0:.0f}s)")
    return {"tag": tag, "tab_mean": float(tf.mean()), "tab_per_fold": np.round(tf, 4).tolist(),
            "blend_mean": float(bf.mean()), "blend_delta_vs_baseline": float(bf.mean() - BASE_MEAN),
            "blend_per_fold": np.round(bf, 4).tolist(), "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    VARIANTS = sys.argv[1:] or ["base"]
    out = []
    print(f"[ref] baseline blend fold-mean = {BASE_MEAN:.6f}")
    for v in VARIANTS:
        t0 = time.time()
        if v == "base":
            X, _, cols = build_X()
            oof = train_oof(X, Y, LGB_BASE)
        elif v.startswith("algw"):
            w = float(v[4:])
            X, _, cols = build_X()
            oof = train_oof(X, Y, LGB_BASE,
                            extra_params={"class_weight": {i: (w if i == 0 else 1.0) for i in range(5)}})
        elif v == "acr":
            X, _, cols = build_X(extra_acr=True)
            oof = train_oof(X, Y, LGB_BASE)
        elif v == "ratios":
            X, _, cols = build_X(ratios=True)
            oof = train_oof(X, Y, LGB_BASE)
        elif v == "acr_ratios":
            X, _, cols = build_X(extra_acr=True, ratios=True)
            oof = train_oof(X, Y, LGB_BASE)
        elif v == "nanidx":
            X, _, cols = build_X(nanidx=True)
            oof = train_oof(X, Y, LGB_BASE)
        elif v == "categorical":
            X, _, cols = build_X()
            idx = [cols.index("geomorphons"), cols.index("pennock")]
            oof = train_oof(X, Y, LGB_BASE, categorical=idx)
        elif v == "bagged":
            X, _, cols = build_X()
            oof = train_oof(X, Y, LGB_BASE, seeds=(42, 7, 2024),
                            extra_params={"feature_fraction": 0.6, "bagging_fraction": 0.8,
                                          "bagging_freq": 1})
        elif v == "bagged_acr":
            X, _, cols = build_X(extra_acr=True)
            oof = train_oof(X, Y, LGB_BASE, seeds=(42, 7, 2024),
                            extra_params={"feature_fraction": 0.6, "bagging_fraction": 0.8,
                                          "bagging_freq": 1})
        else:
            raise SystemExit(f"unknown variant {v}")
        r = report(f"{v} (n_feat={len(cols)})", oof, t0)
        np.save(f"experiments/tab_oof_{v}.npy", oof)
        out.append(r)

    with open(f"experiments/screen_iter2_tab_{'_'.join(VARIANTS)[:40]}.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("done")
