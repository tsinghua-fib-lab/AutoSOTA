"""Aggregate paper-metric eval JSONs into a summary table.

Paper metrics:
  Tri.  (triangle_density)        : RMSE / MAE / Bias
  Clus. (global_clustering_coeff) : RMSE / MAE / Bias
  Deg.  (degree_centrality)       : W1 / KS / Energy
  Eig.  (Laplacian eigenvalues)   : W1 / KS / Energy

Walks ``--eval_root`` for files matching r=<r>/<method>_seed<S>.json.
"""
import argparse
import glob
import json

import numpy as np

ALL_METHODS = ["syngr", "syngd", "syngd_mlp", "gran", "edge", "vgae"]


def load(eval_root, method, r, max_n=20):
    paths = sorted(glob.glob(f"{eval_root}/r={r}/{method}_seed*.json"))
    rows = []
    for p in paths:
        try:
            with open(p) as f:
                rows.append(json.load(f))
        except Exception:
            continue
    return rows[:max_n] if max_n else rows


def summarize(rows):
    keys = ["tri_rmse", "tri_mae", "tri_bias",
            "gcc_rmse", "gcc_mae", "gcc_bias",
            "deg_w1", "deg_ks", "deg_energy",
            "eig_w1", "eig_ks", "eig_energy"]
    out = {"n": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            a = np.asarray(vals, float)
            out[k] = {"mean": float(a.mean()),
                      "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}
    return out


def fmt(s, k):
    v = s.get(k)
    return "—" if v is None else f"{v['mean']:.4g} ± {v['std']:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default="runs/eval_paper",
                    help="dir containing r=<r>/<method>_seed<S>.json")
    ap.add_argument("--methods", default=",".join(ALL_METHODS),
                    help="comma-separated methods to include (in row order)")
    ap.add_argument("--rs", default="2,3,4")
    ap.add_argument("--max_n", type=int, default=20)
    ap.add_argument("--out_md", default="runs/eval_paper/summary.md")
    ap.add_argument("--out_json", default="runs/eval_paper/summary.json")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rs = [int(x) for x in args.rs.split(",")]

    results = {}
    for r in rs:
        for m in methods:
            results[f"{m}_r{r}"] = summarize(load(args.eval_root, m, r, args.max_n))

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    lines = ["# Paper metrics summary", ""]
    lines.append("Tri = triangle density, Clus = global clustering coefficient, "
                 "Deg = degree centrality, Eig = Laplacian eigenvalues.")
    lines.append("")
    lines.append("| Method | r | n | Tri RMSE | Tri MAE | Tri Bias | "
                 "Clus RMSE | Clus MAE | Clus Bias | Deg W1 | Deg KS | Deg Energy | "
                 "Eig W1 | Eig KS | Eig Energy |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        for m in methods:
            s = results.get(f"{m}_r{r}", {})
            n = s.get("n", 0)
            lines.append(
                f"| {m.upper()} | {r} | {n} | {fmt(s,'tri_rmse')} | {fmt(s,'tri_mae')} | {fmt(s,'tri_bias')} "
                f"| {fmt(s,'gcc_rmse')} | {fmt(s,'gcc_mae')} | {fmt(s,'gcc_bias')} "
                f"| {fmt(s,'deg_w1')} | {fmt(s,'deg_ks')} | {fmt(s,'deg_energy')} "
                f"| {fmt(s,'eig_w1')} | {fmt(s,'eig_ks')} | {fmt(s,'eig_energy')} |")
    md = "\n".join(lines)
    with open(args.out_md, "w") as f:
        f.write(md)
    print(md)
    print(f"\nWrote {args.out_md} / {args.out_json}")


if __name__ == "__main__":
    main()
