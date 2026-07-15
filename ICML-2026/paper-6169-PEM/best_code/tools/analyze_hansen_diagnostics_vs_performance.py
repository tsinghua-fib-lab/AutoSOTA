#!/usr/bin/env python3
"""
Merge Hansen fixed-budget performance with BERW residual-pool diagnostics.

Outputs (in --out-dir):
  - perf_vs_diagnostics.csv: per-(function,instance) performance deltas + diagnostics.
  - perf_vs_diagnostics.png: scatter plots for quick sanity checks.
  - worst_cases.md: small table of the most BERW-worse pairs (boundary examples).

This is meant to support the “mismatch decomposition is measurable” claim:
we do not claim diagnostics predict wins/losses, but that they make violations visible and refutable.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _project import BASE_DIR, repo_relpath


def _log10(x: float, *, eps: float) -> float:
    v = float(x)
    if not np.isfinite(v):
        return float("nan")
    return float(np.log10(max(float(eps), v + float(eps))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noisefree-csv",
        default=os.path.join(BASE_DIR, "evidence", "hansen_test_fixed_budget", "noisefree", "bbob_summary.csv"),
    )
    parser.add_argument(
        "--diagnostics-csv",
        default=os.path.join(BASE_DIR, "evidence", "hansen_test_fixed_budget", "diagnostics", "diagnostics_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(BASE_DIR, "evidence", "hansen_test_fixed_budget", "diagnostics"),
    )
    parser.add_argument("--rival", default="UH-CMA-ES(maxevals=30)")
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()

    perf = pd.read_csv(os.path.abspath(str(args.noisefree_csv)))
    diag = pd.read_csv(os.path.abspath(str(args.diagnostics_csv)))
    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    a = "BERW-Hetero"
    b = str(args.rival)
    pivot = perf.pivot_table(index=["function", "instance"], columns="algorithm", values="best_f", aggfunc="first")
    if a not in pivot.columns or b not in pivot.columns:
        raise SystemExit(f"Missing algorithms in noisefree summary: have {list(pivot.columns)}, need {a} and {b}.")

    tmp = pivot[[a, b]].reset_index()
    tmp["diff_berw_minus_rival"] = tmp[a] - tmp[b]
    tmp["log10_ratio_berw_over_rival"] = [
        _log10(xa, eps=float(args.eps)) - _log10(xb, eps=float(args.eps)) for xa, xb in zip(tmp[a], tmp[b])
    ]
    tmp["berw_worse"] = tmp["diff_berw_minus_rival"] > 0.0

    for c in diag.columns:
        if c.startswith(("mean_", "final_", "max_")):
            diag[c] = pd.to_numeric(diag[c], errors="coerce")
    merged = tmp.merge(diag, on=["function", "instance"], how="inner")

    keep_cols = [
        "function",
        "instance",
        a,
        b,
        "diff_berw_minus_rival",
        "log10_ratio_berw_over_rival",
        "berw_worse",
        "function_index",
        "budget_multiplier",
        "dimension",
        "mean_noise_z_pool_size",
        "mean_noise_z_clip_frac",
        "mean_noise_shape_w1",
        "mean_noise_drift_w1",
        "mean_noise_scale_fit_r2",
        "mean_noise_scale_pred_cv",
        "mean_noise_center_split_rel",
        "mean_noise_center_split_cv",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    out_csv = os.path.join(out_dir, "perf_vs_diagnostics.csv")
    merged[keep_cols].to_csv(out_csv, index=False)

    worst = merged.sort_values("log10_ratio_berw_over_rival", ascending=False).head(12)
    md_path = os.path.join(out_dir, "worst_cases.md")
    with open(md_path, "w") as f:
        f.write("# Hansen diagnostics: worst BERW vs UH pairs (auto-generated)\n\n")
        f.write(f"- rival: `{b}`\n")
        f.write(f"- score: `log10_ratio_berw_over_rival = log10(f_BERW+eps)-log10(f_rival+eps)` (positive = BERW worse)\n\n")
        f.write("| fidx | inst | log10_ratio | drift_w1 | shape_w1 | scale_r2 | center_rel |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in worst.iterrows():
            fidx = int(r.get("function_index", int(r["function"]) - 100))
            f.write(
                "| {fidx} | {inst} | {lr:.3f} | {dw:.3f} | {sw:.3f} | {r2:.3f} | {cr:.3f} |\n".format(
                    fidx=fidx,
                    inst=int(r["instance"]),
                    lr=float(r["log10_ratio_berw_over_rival"]),
                    dw=float(r.get("mean_noise_drift_w1", float("nan"))),
                    sw=float(r.get("mean_noise_shape_w1", float("nan"))),
                    r2=float(r.get("mean_noise_scale_fit_r2", float("nan"))),
                    cr=float(r.get("mean_noise_center_split_rel", float("nan"))),
                )
            )

    # Scatter plot (log10_ratio vs key diagnostics).
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), dpi=180)
    axes = axes.reshape(-1)
    y = pd.to_numeric(merged["log10_ratio_berw_over_rival"], errors="coerce").to_numpy(dtype=float)
    plots = [
        ("mean_noise_drift_w1", "drift W1"),
        ("mean_noise_shape_w1", "shape W1"),
        ("mean_noise_scale_fit_r2", "scale fit R2"),
        ("mean_noise_center_split_rel", "center split rel"),
        ("mean_noise_scale_pred_cv", "scale pred CV"),
        ("mean_noise_z_pool_size", "pool size"),
    ]
    for ax, (xk, title) in zip(axes, plots):
        x = pd.to_numeric(merged.get(xk, pd.Series([np.nan] * len(merged))), errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=14, alpha=0.75)
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.35)
        ax.set_xlabel(title)
        ax.set_ylabel("log10 ratio (BERW / rival)")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Hansen fixed-budget: performance vs residual diagnostics", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(out_dir, "perf_vs_diagnostics.png")
    plt.savefig(out_png)
    plt.close(fig)

    print("Wrote:", repo_relpath(out_csv))
    print("Wrote:", repo_relpath(md_path))
    print("Wrote:", repo_relpath(out_png))


if __name__ == "__main__":
    main()
