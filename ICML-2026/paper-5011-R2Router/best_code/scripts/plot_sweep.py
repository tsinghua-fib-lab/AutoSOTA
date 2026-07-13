#!/usr/bin/env python3
"""Plot lambda sweep comparison — Global KNN (10% train, 100% eval)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data from Global KNN sweep (sweep_global_knn_24843736) ────────────────────

# Only plot lambda >= 0.9
lambdas = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]

# 4-model (235b, 80b, gemini-flash, haiku) — excl unlimited+1500
acc_4  = [70.27, 70.30, 70.27, 70.44, 70.47, 70.73, 65.71]
cost_4 = [0.0586, 0.0576, 0.0554, 0.0527, 0.0471, 0.0352, 0.0199]
arena_005_4  = [70.52, 70.55, 70.54, 70.72, 70.79, 71.15, 66.45]
arena_010_4  = [70.74, 70.78, 70.78, 70.98, 71.09, 71.53, 67.15]
arena_019_4  = [75.56, 75.71, 76.05, 76.50, 77.49, 80.05, 84.63]

# 5-model (+ gpt-5.2) — excl unlimited+1500
acc_5  = [73.04, 72.35, 70.95, 70.62, 70.59, 70.74, 65.71]
cost_5 = [0.9945, 0.6163, 0.1636, 0.0642, 0.0495, 0.0352, 0.0199]
arena_005_5  = [71.42, 71.19, 70.71, 70.82, 70.89, 71.15, 66.45]
arena_010_5  = [70.01, 70.17, 70.50, 71.00, 71.16, 71.54, 67.15]
arena_019_5  = [50.27, 54.61, 66.50, 74.77, 77.06, 80.05, 84.63]

# 9-model (no gpt4o, haiku) — excl unlimited+1500
acc_9  = [70.51, 70.49, 70.44, 70.55, 70.52, 70.27, 50.32]
cost_9 = [0.0378, 0.0375, 0.0365, 0.0355, 0.0336, 0.0281, 0.0019]
arena_005_9  = [70.91, 70.89, 70.85, 70.97, 70.96, 70.78, 51.54]
arena_010_9  = [71.27, 71.26, 71.23, 71.35, 71.36, 71.24, 52.70]
arena_019_9  = [79.41, 79.48, 79.72, 79.97, 80.44, 81.99, 95.30]

# 11-model (all except gemma-3n) — excl unlimited+1500
acc_11  = [70.89, 70.96, 70.92, 70.96, 70.90, 70.24, 50.32]
cost_11 = [0.0485, 0.0475, 0.0445, 0.0424, 0.0396, 0.0287, 0.0019]
arena_005_11 = [71.18, 71.26, 71.24, 71.30, 71.27, 70.74, 51.54]
arena_010_11 = [71.45, 71.53, 71.54, 71.61, 71.61, 71.20, 52.70]
arena_019_11 = [77.25, 77.44, 78.01, 78.44, 79.03, 81.80, 95.30]

# ── Leaderboard baselines (constant across lambda) ───────────────────────────
# Method 1: Acc=66.5%, Cost=$0.06/1kq
# Method 2: Acc=74.0%, Cost=$10.02/1kq
LB = {
    "M1": {"acc": 66.5, "cost": 0.06,
           "arena_005": 66.88, "arena_010": 67.24, "arena_019": 75.12},
    "M2": {"acc": 74.0, "cost": 10.02,
           "arena_005": 68.61, "arena_010": 64.34, "arena_019": 28.81},
    "M3": {"acc": 71.94, "cost": 0.0646,
           "arena_005": 72.08, "arena_010": 72.20, "arena_019": 74.79},
}

# ── Common styling ────────────────────────────────────────────────────────────

COLORS = {
    "4-model": "#2563eb",
    "5-model": "#9333ea",
    "9-model": "#16a34a",
    "11-model": "#dc2626",
}
MARKERS = {"4-model": "o", "5-model": "D", "9-model": "s", "11-model": "^"}
LABELS = {
    "4-model": "4-model (235b, 80b, flash, haiku)",
    "5-model": "5-model (+ gpt-5.2)",
    "9-model": "9-model (no gpt4o/haiku)",
    "11-model": "11-model (all excl. gemma-3n)",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def plot_one(ax, datasets, ylabel, title, ylim=None, baselines=None):
    x = np.arange(len(lambdas))
    for key, y in datasets:
        ax.plot(x, y, color=COLORS[key], marker=MARKERS[key],
                markersize=7, linewidth=2.2, label=LABELS[key])
    # Draw leaderboard baselines as horizontal dashed lines
    if baselines:
        for label, val, color, ls in baselines:
            ax.axhline(y=val, color=color, linestyle=ls, linewidth=1.8, alpha=0.7, label=label)
    ax.set_xlabel("Lambda")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}" for l in lambdas], fontsize=10)


# ── Create 5 separate figures ─────────────────────────────────────────────────

outdir = "/home/ah872032.ucf/jiaqi/router"

M1C, M2C, M3C = "#f59e0b", "#ec4899", "#06b6d4"  # amber, pink, cyan
M1S, M2S, M3S = "--", ":", "-."

plots = [
    ([("4-model", acc_4), ("5-model", acc_5), ("9-model", acc_9), ("11-model", acc_11)],
     "Accuracy (%)", "Accuracy vs Lambda (Global KNN)", None, "accuracy_vs_lambda.png",
     [("Method 1 (Acc=66.5%)", LB["M1"]["acc"], M1C, M1S),
      ("Method 2 (Acc=74.0%)", LB["M2"]["acc"], M2C, M2S),
      ("Method 3 (Acc=71.94%)", LB["M3"]["acc"], M3C, M3S)]),
    ([("4-model", cost_4), ("5-model", cost_5), ("9-model", cost_9), ("11-model", cost_11)],
     "Cost per 1K queries ($)", "Cost vs Lambda (Global KNN)", None, "cost_vs_lambda.png",
     [("Method 1 ($0.06)", LB["M1"]["cost"], M1C, M1S),
      ("Method 2 ($10.02)", LB["M2"]["cost"], M2C, M2S),
      ("Method 3 ($0.065)", LB["M3"]["cost"], M3C, M3S)]),
    ([("4-model", arena_010_4), ("5-model", arena_010_5), ("9-model", arena_010_9), ("11-model", arena_010_11)],
     "Arena Score", "Arena Score (beta=0.10) vs Lambda (Global KNN)", None, "arena_beta010_vs_lambda.png",
     [("Method 1 (67.24)", LB["M1"]["arena_010"], M1C, M1S),
      ("Method 2 (64.34)", LB["M2"]["arena_010"], M2C, M2S),
      ("Method 3 (72.20)", LB["M3"]["arena_010"], M3C, M3S)]),
    ([("4-model", arena_005_4), ("5-model", arena_005_5), ("9-model", arena_005_9), ("11-model", arena_005_11)],
     "Arena Score", "Arena Score (beta=0.05) vs Lambda (Global KNN)", None, "arena_beta005_vs_lambda.png",
     [("Method 1 (66.88)", LB["M1"]["arena_005"], M1C, M1S),
      ("Method 2 (68.61)", LB["M2"]["arena_005"], M2C, M2S),
      ("Method 3 (72.08)", LB["M3"]["arena_005"], M3C, M3S)]),
    ([("4-model", arena_019_4), ("5-model", arena_019_5), ("9-model", arena_019_9), ("11-model", arena_019_11)],
     "Arena Score", "Arena Score (beta=19) vs Lambda (Global KNN)", None, "arena_beta19_vs_lambda.png",
     [("Method 1 (75.12)", LB["M1"]["arena_019"], M1C, M1S),
      ("Method 2 (28.81)", LB["M2"]["arena_019"], M2C, M2S),
      ("Method 3 (74.79)", LB["M3"]["arena_019"], M3C, M3S)]),
]

for datasets, ylabel, title, ylim, fname, baselines in plots:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_one(ax, datasets, ylabel, title, ylim, baselines=baselines)
    fig.tight_layout()
    outpath = f"{outdir}/{fname}"
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)

print("\nDone - 5 plots saved.")
