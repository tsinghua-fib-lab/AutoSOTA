#!/usr/bin/env python3
"""Screen 11 — re-sweep the live tabular component's capacity under the FUSE-02 operator.

Round 1 chose min_child_samples=100 / num_leaves=31 by screening against the OLD fusion
(0.7688). FUSE-02 changed the operator, so the capacity optimum can move. The round-1 evidence
says the response is a broad plateau rather than a spike, which makes a small structured sweep
low-risk; the referee below reports both the in-sample best and the honest leave-one-fold-out
value so an OOF-selected capacity cannot be reported as a gain.

Usage: python experiments/probe23_capacity.py <tag:key=val,...> ...
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, "experiments")
from probe19_tabular import (BASE_MEAN, BASE_VEC, FOLDS, CNN, LGB_BASE, Y, build_X,  # noqa: E402
                             fuse_blend, train_oof)
from probe_referee import fold_vec, referee  # noqa: E402

X, _, cols = build_X()
assert X.shape == (6256, 87)

CANDIDATES = {
    "mcs60": {"min_child_samples": 60},
    "mcs80": {"min_child_samples": 80},
    "mcs150": {"min_child_samples": 150},
    "mcs200": {"min_child_samples": 200},
    "mcs300": {"min_child_samples": 300},
    "mcs400": {"min_child_samples": 400},
    "nl7": {"num_leaves": 7},
    "nl15": {"num_leaves": 15},
    "nl63": {"num_leaves": 63},
    "nl127": {"num_leaves": 127},
    "mcs200nl15": {"min_child_samples": 200, "num_leaves": 15},
    "mcs60nl63": {"min_child_samples": 60, "num_leaves": 63},
}

wanted = sys.argv[1:] or list(CANDIDATES)
cands = {}
print(f"[ref] baseline blend = {BASE_MEAN:.6f} ; incumbent FUSE-02 with mcs=100/nl=31")
t0 = time.time()
cands["mcs100nl31"] = np.load("experiments/tab_oof_base.npy")   # cached, verified identical
print(f"  mcs100nl31   (cached)                            {time.time()-t0:.0f}s")
for w in wanted:
    if w not in CANDIDATES:
        raise SystemExit(f"unknown {w}")
    t = time.time()
    oof = train_oof(X, Y, LGB_BASE, extra_params=CANDIDATES[w])
    np.save(f"experiments/tab_oof_{w}.npy", oof)
    cands[w] = oof
    print(f"  trained {w:12s} ({LGB_BASE['min_child_samples'] if 'min_child_samples' not in CANDIDATES[w] else CANDIDATES[w]['min_child_samples']}"
          f"/{CANDIDATES[w].get('num_leaves', LGB_BASE['num_leaves'])})  {time.time()-t:.0f}s")

rows = []
for tag, oof in cands.items():
    tf = fold_vec(Y, oof.argmax(1), FOLDS)
    bp = fuse_blend(oof)
    bf = fold_vec(Y, bp, FOLDS)
    pf = np.round(bf - BASE_VEC, 4)
    rows.append({"tag": tag, "tab_mean": float(tf.mean()), "blend_mean": float(bf.mean()),
                 "delta": float(bf.mean() - BASE_MEAN), "n_pos": int((bf - BASE_VEC >= -1e-12).sum()),
                 "per_fold_delta": pf.tolist()})
    print(f"  {tag:14s} tab={tf.mean():.4f} blend={bf.mean():.6f} d={bf.mean()-BASE_MEAN:+.4f} "
          f"pos={rows[-1]['n_pos']}/5 {pf.tolist()}")

print("\n[LOO referee over the capacity grid]")
cand_preds = [(t, fuse_blend(o)) for t, o in cands.items()]
r = referee(Y, FOLDS, cand_preds)
base_inc = float(np.mean([fold_vec(Y, fuse_blend(cands["mcs100nl31"]), FOLDS).mean()]))
inc_mean = float(fold_vec(Y, fuse_blend(cands["mcs100nl31"]), FOLDS).mean())
print(f"  in-sample best = {r['in_sample_best_score']:.4f} ({r['in_sample_best_label']})")
print(f"  LOO honest     = {r['loo_honest']:.4f}  d_vs_baseline={r['loo_honest']-BASE_MEAN:+.4f}  "
      f"d_vs_incumbent={r['loo_honest']-inc_mean:+.4f}")
print(f"  chosen         = {r['loo_chosen']}")

with open("experiments/screen_iter5_capacity.json", "w") as fh:
    json.dump({"rows": rows, "loo": r["loo_honest"], "loo_chosen": r["loo_chosen"],
               "in_sample_best": r["in_sample_best_label"], "incumbent": inc_mean}, fh, indent=2)
print("wrote experiments/screen_iter5_capacity.json")
