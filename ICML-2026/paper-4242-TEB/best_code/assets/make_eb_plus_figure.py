#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""README figure: EB+ adaptively wins across all scenarios.

Visualizes EB+ (hybrid scheduler) throughput gain over v1 (MB) under
moderate-to-high traffic and non-stationary workloads. Data from paper
Table 4 (traffic) and Table 5 (non-stationary).
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# (Scenario label, GPU, EB(k̂*) % gain vs v1, EB+ % gain vs v1)
# Source: paper Table 4 (traffic) and Table 5 (non-stationary)
data = [
    # Traffic-level sensitivity (μ_L=512, μ_O=256)
    ("c=512", "RTX 6000", +62.7, +62.2),  # Table 4
    ("c=512", "H200", -0.9, +0.0),
    ("c=2048", "RTX 6000", +49.3, +49.7),
    ("c=2048", "H200", +3.1, +2.6),
    # Non-stationary
    ("Distrib. shift", "RTX 6000", +37.3, +36.4),  # Table 5
    ("Distrib. shift", "H200", +1.8, +3.9),
    ("Concur. shift", "RTX 6000", +23.7, +22.6),
    ("Concur. shift", "H200", -3.6, +0.6),
]

n = len(data)
x = np.arange(n)
width = 0.36

eb_gains = [d[2] for d in data]
ebp_gains = [d[3] for d in data]

fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=150)

# Background shading
ax.axhspan(0, 50, alpha=0.04, color="#2E8B57", zorder=0)
ax.axhspan(-30, 0, alpha=0.04, color="#C04A4A", zorder=0)
ax.axhline(0, color="black", linewidth=1.0, zorder=1)

# EB(k̂*) bars: muted blue
eb_bars = ax.bar(
    x - width / 2,
    eb_gains,
    width,
    label=r"EB($\hat{k}^*$)",
    color="#8FA8C9",
    edgecolor="black",
    linewidth=0.6,
)

# EB+ bars: highlight gold
ebp_bars = ax.bar(
    x + width / 2,
    ebp_gains,
    width,
    label=r"EB$^+$ (hybrid)",
    color="#E8A33D",
    edgecolor="black",
    linewidth=0.8,
)

# Value labels
for bar, v in zip(eb_bars, eb_gains):
    h = bar.get_height()
    y = h + 1.5 if h >= 0 else h - 1.5
    va = "bottom" if h >= 0 else "top"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y,
        f"{v:+.1f}",
        ha="center",
        va=va,
        fontsize=8.5,
        color="#445",
    )

for bar, v in zip(ebp_bars, ebp_gains):
    h = bar.get_height()
    y = h + 1.5 if h >= 0 else h - 2.0
    va = "bottom" if h >= 0 else "top"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y,
        f"{v:+.1f}",
        ha="center",
        va=va,
        fontsize=9,
        fontweight="bold",
        color="#7a4a00",
    )

# X labels: two-line (scenario / GPU)
labels = [f"{s}\n{g}" for s, g, _, _ in data]
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)

# Visual group separation: vertical dividers between traffic and non-stationary,
# and between c=512/c=2048
for sep in (1.5, 3.5, 5.5):
    ax.axvline(sep, color="#cccccc", linewidth=0.7, linestyle="--", zorder=0)

# Group labels at top
ax.text(
    (0 + 1) / 2,
    78,
    "Moderate load\n(c=512)",
    ha="center",
    fontsize=10,
    color="#444",
    style="italic",
)
ax.text(
    (2 + 3) / 2,
    78,
    "High load\n(c=2048)",
    ha="center",
    fontsize=10,
    color="#444",
    style="italic",
)
ax.text(
    (4 + 5) / 2,
    78,
    "Distribution shift\n(non-stationary)",
    ha="center",
    fontsize=10,
    color="#444",
    style="italic",
)
ax.text(
    (6 + 7) / 2,
    78,
    "Concurrency shift\n(non-stationary)",
    ha="center",
    fontsize=10,
    color="#444",
    style="italic",
)

# Y axis
ax.set_ylabel("Throughput change vs v1 (MB)  (%)", fontsize=11.5)
ax.set_ylim(-15, 88)
ax.set_yticks(range(-10, 71, 10))
ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)

# Legend: upper right, just below the "Concurrency shift" group label
ax.legend(
    loc="upper right",
    bbox_to_anchor=(0.995, 0.88),
    fontsize=10.5,
    framealpha=0.95,
    edgecolor="#888",
    ncol=1,
)

# (No headline annotation box — title + data labels + group labels suffice.)

# Title
ax.set_title(
    "EB$^+$: hybrid scheduler that adaptively picks "
    "the better of {EB, MB}\n"
    r"$\it{Qwen3\!-\!8B,\ paper\ Tables\ 4\ and\ 5}$",
    fontsize=13,
    fontweight="bold",
    pad=10,
    linespacing=1.3,
)

plt.tight_layout()
out_path = "/data/yuzhou/projects/aproj/vllm-sched/eb-vllm/assets/eb_plus_advantage.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
