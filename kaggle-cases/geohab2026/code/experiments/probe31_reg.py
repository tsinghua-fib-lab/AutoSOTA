#!/usr/bin/env python3
"""Screen 19 — L1/L2 leaf regularization of the live tabular component.

`LGB_REG_ALPHA` and `LGB_REG_LAMBDA` are both 0.0 in the current CONFIG and were never swept.
Round 1's central finding was that this sample is dominated by near-duplicate points and that
*regularization* (the leaf floor) is the lever that works; L2 leaf shrinkage is the natural
next regularizer of the same kind and is orthogonal to min_child_samples. Regularizing leaf
values shrinks each leaf's log-odds toward the parent, which should help exactly where the
capacity correction helped.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "experiments")
from probe19_tabular import (BASE_MEAN, BASE_VEC, FOLDS, LGB_BASE, Y,  # noqa: E402
                             build_X, train_oof)
X, _, _cols = build_X()
assert X.shape == (6256, 87), X.shape
from probe_referee import fold_vec, referee  # noqa: E402

# The incumbent for this screen is the current SOTA pipeline (a=12 temperature + FUSE-03
# smoothing), so build the comparison from the pipeline's own operator.
from probe_referee import load  # noqa: E402
D = load()
TAB0 = np.load("experiments/tab_oof_base.npy")
assert np.allclose(TAB0, D["tab"], atol=1e-12), "cached base tab_oof is not the frozen one"

GRID = {
    "lam1": {"reg_lambda": 1.0},
    "lam5": {"reg_lambda": 5.0},
    "lam20": {"reg_lambda": 20.0},
    "lam50": {"reg_lambda": 50.0},
    "alp1": {"reg_alpha": 1.0},
    "alp5": {"reg_alpha": 5.0},
    "lam5alp5": {"reg_lambda": 5.0, "reg_alpha": 5.0},
}
wanted = sys.argv[1:] or list(GRID)
cands = {"base (lam0)": TAB0}
import time  # noqa: E402
for w in wanted:
    t = time.time()
    cands[w] = train_oof(X, Y, LGB_BASE, extra_params=GRID[w])
    np.save(f"experiments/tab_oof_{w}.npy", cands[w])
    print(f"  trained {w} {time.time()-t:.0f}s")


def full_operator(tab_oof):
    """The current pipeline operator: a=12 temperature, FUSE-02 shares, FUSE-03 smoothing."""
    from scipy.spatial import cKDTree
    XY = np.column_stack([D["df"]["x"].to_numpy(), D["df"]["y"].to_numpy()])
    tree = cKDTree(XY)
    _, nb_idx = tree.query(XY, k=120)
    same = FOLDS[nb_idx] == FOLDS[:, None]
    w = same.astype(float)
    w /= w.sum(1, keepdims=True)
    ts = np.power(tab_oof, 12.0)
    ts /= ts.sum(1, keepdims=True)
    P = np.zeros((len(Y), 5))
    for f in range(5):
        va = FOLDS == f
        sh = np.full(5, 0.5)
        for c in range(5):
            rt = _f1(Y[FOLDS != f], tab_oof.argmax(1)[FOLDS != f], c)
            rc = _f1(Y[FOLDS != f], D["cnn"].argmax(1)[FOLDS != f], c)
            if rt is None or rc is None:
                continue
            sh[c] = (rt + 0.25) / (rt + rc + 0.5)
        p = 0.40 * ts[va] * sh[None, :] + 0.60 * D["cnn"][va] * (1 - sh)[None, :]
        P[va] = p / p.sum(1, keepdims=True)
    Q = 0.7 * P + 0.3 * np.einsum("ij,ijc->ic", w, P[nb_idx])
    return (Q / Q.sum(1, keepdims=True)).argmax(1)


def _f1(y, pred, c):
    tp = int(np.sum((y == c) & (pred == c)))
    fp = int(np.sum((pred == c) & (y != c)))
    fn = int(np.sum((y == c) & (pred != c)))
    s = tp + fn
    if s == 0:
        return None
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / s
    return 2 * pr * rec / (pr + rec) if (pr + rec) else 0.0


INC_VEC = fold_vec(Y, full_operator(TAB0), FOLDS)
INC_MEAN = float(INC_VEC.mean())
print(f"[incumbent = current SOTA pipeline] mean={INC_MEAN:.6f}")
rows = []
for tag, oof in cands.items():
    tf = fold_vec(Y, oof.argmax(1), FOLDS)
    pred = full_operator(oof)
    fv = fold_vec(Y, pred, FOLDS)
    pf = np.round(fv - INC_VEC, 4)
    npos = int((fv - INC_VEC >= -1e-12).sum())
    rows.append({"tag": tag, "tab_mean": float(tf.mean()), "mean": float(fv.mean()),
                 "delta": float(fv.mean() - INC_MEAN), "n_pos": npos, "per_fold_delta": pf.tolist()})
    print(f"  {tag:14s} tab={tf.mean():.4f} blend={fv.mean():.6f} d={fv.mean()-INC_MEAN:+.4f} "
          f"pos={npos}/5 {pf.tolist()}")

print("\n[referee over the regularization grid]")
cand_preds = [(t, full_operator(o)) for t, o in cands.items()]
r = referee(Y, FOLDS, cand_preds)
print(f"  in_sample best={r['in_sample_best_score']:.4f} ({r['in_sample_best_label']}) "
      f"LOO={r['loo_honest']:.4f} d_base={r['loo_honest']-BASE_MEAN:+.4f} chose={r['loo_chosen']}")

with open("experiments/screen_iter4_reg.json", "w") as fh:
    json.dump({"incumbent": INC_MEAN, "rows": rows, "loo": float(r["loo_honest"]),
               "loo_chosen": r["loo_chosen"]}, fh, indent=2)
print("wrote experiments/screen_iter4_reg.json")
