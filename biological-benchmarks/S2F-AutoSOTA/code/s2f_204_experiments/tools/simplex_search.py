#!/usr/bin/env python3
"""IDEA-602: offline (w, c, e) weight-simplex search over per-row component
score dumps produced by script/evaluate.py with S2F_DUMP_COMPONENTS=<dir>.

Row model (mirrors evaluate.py::get_prob exactly):
    fused  = (1-c) * ((1-w)*s2f + w*masked) + c*wt
    final  = (1 - e*cov_i) * fused + e*cov_i * pssm     (cov_i in {0,1})

Per-assay Spearman follows the torchdrug convention (average-rank Pearson);
the reported number is the mean over assays, exactly like the eval pipeline.

Usage:
    python tools/simplex_search.py <dump_dir> [--validate] [--fine STEP]
"""
import argparse
import glob
import os

import numpy as np

try:
    import torch
    from torchdrug import metrics
    HAVE_TORCHDRUG = True
except Exception:
    HAVE_TORCHDRUG = False


def _avg_rank(x):
    """torchdrug get_ranking(): 1-based ranks with ties averaged."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    u, inv = np.unique(x, return_inverse=True)
    if len(u) < len(x):
        mean_rank = np.zeros(len(u), dtype=np.float64)
        np.add.at(mean_rank, inv, ranks)
        mean_rank /= np.bincount(inv)
        ranks = mean_rank[inv]
    return ranks


def spearman_np(pred, target):
    p, t = _avg_rank(pred), _avg_rank(target)
    cov = (p * t).mean() - p.mean() * t.mean()
    return float(cov / (p.std() * t.std() + 1e-10))


def spearman_td(pred, target):
    return float(metrics.spearmanr(torch.as_tensor(pred, dtype=torch.float32),
                                   torch.as_tensor(target, dtype=torch.float32)))


def load_assays(dump_dir):
    assays = {}
    for f in sorted(glob.glob(os.path.join(dump_dir, "*.npz"))):
        z = np.load(f)
        assays[os.path.basename(f)[:-4]] = {
            k: z[k].astype(np.float64) for k in z.files}
    return assays


def rows_at(assays, w, c, e):
    """Return {assay: (final_row_scores, target)} for one (w, c, e) cell."""
    out = {}
    for name, d in assays.items():
        fused = (1 - c) * ((1 - w) * d["s2f"] + w * d["masked"]) + c * d["wt"]
        cov = (d["cov"] >= 1 - 1e-9).astype(np.float64)
        final = (1 - e * cov) * fused + e * cov * d["pssm"]
        t = d["target"]
        mask = np.isfinite(t)
        if not mask.all():
            final, t = final[mask], t[mask]
        out[name] = (final, t)
    return out


def mean_spearman(rows, metric):
    return float(np.mean([metric(p, t) for p, t in rows.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--validate", action="store_true",
                    help="re-check anchor cells with the exact torchdrug metric")
    ap.add_argument("--fine", type=float, default=0.01,
                    help="refinement step around the coarse argmax")
    args = ap.parse_args()

    assays = load_assays(args.dump_dir)
    if not assays:
        raise SystemExit("no .npz dumps under %s" % args.dump_dir)
    metric = spearman_np
    print("[simplex] %d assays, %d components" %
          (len(assays), len(next(iter(assays.values())))))
    n_nan = sum(1 for d in assays.values() if not np.isfinite(d["target"]).all())
    if n_nan:
        print("[simplex] NOTE: %d assays have non-finite targets; those rows "
              "are masked in the search" % n_nan)

    def score(w, c, e):
        return mean_spearman(rows_at(assays, w, c, e), metric)

    # Coarse grid: w x c x e
    ws = [0.0, 0.05, 0.10, 0.15, 0.20]
    cs = np.arange(0.0, 0.55, 0.025)
    es = np.arange(0.0, 0.60, 0.025)
    best = None
    for w in ws:
        for c in cs:
            for e in es:
                s = score(w, c, e)
                if best is None or s > best[0]:
                    best = (s, w, c, e)
    print("[simplex] coarse grid %d x %d x %d = %d cells" %
          (len(ws), len(cs), len(es), len(ws) * len(cs) * len(es)))
    print("[simplex] argmax: score=%.4f at w=%.3f c=%.3f e=%.3f" % best)

    # Fine refinement around the coarse argmax
    _, w0, c0, e0 = best
    fine = args.fine
    w_grid = sorted(set([0.0] + [w0 + d for d in (-fine, 0, fine)
                                if w0 + d >= 0]))
    c_grid = sorted(set([c0 + d for d in (-fine, 0, fine) if 0 <= c0 + d <= 1]))
    e_grid = sorted(set([e0 + d for d in (-fine, 0, fine) if 0 <= e0 + d <= 1]))
    best_f = None
    for w in w_grid:
        for c in c_grid:
            for e in e_grid:
                s = score(w, c, e)
                if best_f is None or s > best_f[0]:
                    best_f = (s, w, c, e)
    print("[simplex] refined argmax: score=%.4f at w=%.3f c=%.3f e=%.3f"
          % best_f)

    print("\n[simplex] anchor / reference cells (sel-80 mean Spearman):")
    refs = [
        ("champion (round-5 winner)", 0.0, 0.35, 0.0),
        ("iter-1 measured (dump config)", 0.0, 0.225, 0.25),
        ("S2F solo", 0.0, 0.0, 0.0),
        ("PSSM solo (covered rows)", 0.0, 0.0, 1.0),
        ("wt-ESM solo", 0.0, 1.0, 0.0),
        ("masked-ESM solo", 1.0, 0.0, 0.0),
    ]
    for label, w, c, e in refs:
        s = score(w, c, e)
        print("  %-32s w=%.2f c=%.2f e=%.2f -> %.4f" % (label, w, c, e, s))

    if args.validate and HAVE_TORCHDRUG:
        print("\n[simplex] torchdrug-metric validation (exact eval metric):")
        for label, w, c, e in refs + [("REFINED WINNER",) + best_f[1:]]:
            s = mean_spearman(rows_at(assays, w, c, e), spearman_td)
            print("  %-32s w=%.2f c=%.2f e=%.2f -> %.4f" % (label, w, c, e, s))
    elif args.validate:
        print("\n[simplex] WARNING: torchdrug unavailable; skipped validation")


if __name__ == "__main__":
    main()
