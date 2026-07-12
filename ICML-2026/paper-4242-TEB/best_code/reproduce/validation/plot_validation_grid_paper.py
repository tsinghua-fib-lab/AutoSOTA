#!/usr/bin/env python3
"""Reproduce figs/validation_grid.pdf from extracted data for comparison."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent
DATA = PLOT_DIR / "paper_data" / "validation_grid.json"
OUT_DIR = PLOT_DIR

with open(DATA) as f:
    data = json.load(f)

k_values = np.array(data["k_values"])
workloads = [
    ("in128 out1024",  data["workloads"]["decode_heavy_in128_out1024"]),
    ("in512 out512",   data["workloads"]["balanced_in512_out512"]),
    ("in1024 out128",  data["workloads"]["prefill_heavy_in1024_out128"]),
]
wl_labels = ["Decode-heavy", "Balanced", "Prefill-heavy"]

# --- Color scheme ---
C_SWEEP = "#2D6A4F"      # dark green for fixed-k sweep
C_CP    = "#E63946"       # red for chunked prefill (v1)
C_EB    = "#457B9D"       # medium blue for EB (ours)

fig, axes = plt.subplots(3, 3, figsize=(10, 6.5))
fig.subplots_adjust(hspace=0.05, wspace=0.20)

# Column titles (top row only)
col_titles = ["Throughput (tok/s)", "TPOT mean (ms)", "TPOT p99 (ms)"]

for i, (wl, wd) in enumerate(workloads):
    tp = wd["throughput"]
    tpot = wd["tpot"]
    ax_tp = axes[i, 0]
    ax_mean = axes[i, 1]
    ax_p99 = axes[i, 2]
    is_top = (i == 0)
    is_bottom = (i == 2)

    # --- Col 0: Throughput panel ---
    ax_tp.plot(k_values, tp["fixed_k"], color=C_SWEEP, marker="o", markersize=4,
               linewidth=1.2, label=r"Fixed $k$")
    ax_tp.axhline(tp["v1"], color=C_CP, linestyle="-", linewidth=2,
                  label=f"v1 ({tp['v1']})")
    ax_tp.axhline(tp["ours"], color=C_EB, linestyle="--", linewidth=2,
                  dashes=(6, 3),
                  label=r"EB($\hat{k}^*$)" + f" ({tp['ours']})")
    bk, bv = tp["best_k"], tp["best_val"]
    ax_tp.annotate(
        f"$k$={bk}",
        xy=(bk, bv),
        xytext=(bk - 150, bv + (tp["ours"] - tp["v1"]) * 0.18),
        fontsize=8, color=C_SWEEP, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_SWEEP, lw=0.8),
    )
    ax_tp.set_ylabel(wl_labels[i], fontsize=10, fontweight="bold")
    ax_tp.legend(fontsize=7, loc="best", framealpha=0.8)
    ax_tp.grid(True, alpha=0.2)
    ax_tp.tick_params(labelsize=8)

    # --- Col 1: TPOT mean panel ---
    ax_mean.plot(k_values, tpot["fixed_k_mean"], color=C_SWEEP, marker="o",
                 markersize=4, linewidth=1.2, label=r"Fixed $k$")
    ax_mean.axhline(tpot["v1_mean"], color=C_CP, linestyle="-", linewidth=2,
                    label=f"v1 ({tpot['v1_mean']})")
    ax_mean.axhline(tpot["ours_mean"], color=C_EB, linestyle="--", linewidth=2,
                    dashes=(6, 3),
                    label=r"EB($\hat{k}^*$)" + f" ({tpot['ours_mean']})")
    ax_mean.legend(fontsize=7, loc="best", framealpha=0.8)
    ax_mean.grid(True, alpha=0.2)
    ax_mean.tick_params(labelsize=8)

    # --- Col 2: TPOT p99 panel ---
    ax_p99.plot(k_values, tpot["fixed_k_p99"], color=C_SWEEP, marker="s",
                markersize=4, linewidth=1.2, label=r"Fixed $k$")
    ax_p99.axhline(tpot["v1_p99"], color=C_CP, linestyle="-", linewidth=2,
                   label=f"v1 ({tpot['v1_p99']})")
    ax_p99.axhline(tpot["ours_p99"], color=C_EB, linestyle="--", linewidth=2,
                   dashes=(6, 3),
                   label=r"EB($\hat{k}^*$)" + f" ({tpot['ours_p99']})")
    ax_p99.legend(fontsize=7, loc="best", framealpha=0.8)
    ax_p99.grid(True, alpha=0.2)
    ax_p99.tick_params(labelsize=8)

    # Column titles on top row only
    if is_top:
        ax_tp.set_title(col_titles[0], fontsize=10, fontweight="bold")
        ax_mean.set_title(col_titles[1], fontsize=10, fontweight="bold")
        ax_p99.set_title(col_titles[2], fontsize=10, fontweight="bold")

    # x-labels on bottom row only
    if is_bottom:
        for ax in [ax_tp, ax_mean, ax_p99]:
            ax.set_xlabel(r"$k$", fontsize=9)
    else:
        for ax in [ax_tp, ax_mean, ax_p99]:
            ax.set_xticklabels([])


out_pdf = OUT_DIR / "validation_grid_new.pdf"
out_png = OUT_DIR / "validation_grid_new.png"
plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
plt.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved: {out_pdf}\nSaved: {out_png}")
