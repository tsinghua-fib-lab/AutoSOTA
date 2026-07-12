#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dual-panel README header: EB+ Throughput gain + TPOT reduction vs GPU bandwidth.

Numbers reflect EB+ (hybrid) vs v1 (MB) on the WildChat workload (paper
Tables 5 & 6 for Qwen3-8B). Because EB+ by construction picks max(EB, MB),
it tracks EB's gain on bandwidth-constrained GPUs (L40S, RTX PRO 6000) and
stays close to v1 on high-bandwidth GPUs (H200, B300).
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

# (GPU, BW TB/s, EB+ Throughput gain %, EB+ TPOT reduction %)
# Ordered from highest bandwidth (left) to lowest (right) so the bars
# rise left-to-right, visualizing: as bandwidth shrinks, EB+ wins more.
# Values from paper Table 5 (H200, RTX PRO 6000 WildChat) and Table 6
# (L40S, B300 scalability).
data = [
    ("B300", 8.0, 3.7, 7.8),
    ("H200", 4.8, 3.0, 35.7),
    ("RTX PRO 6000", 1.792, 11.3, 34.7),
    ("L40S", 0.864, 41.9, 27.2),
]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=150)

x = np.arange(len(data))
gpus = [d[0] for d in data]
bws = [d[1] for d in data]
tps = [d[2] for d in data]
tpots = [d[3] for d in data]


# Color functions
def th_color(g):
    if g > 5:
        return "#2E8B57"
    if g > -3:
        return "#B0B0B0"
    return "#C04A4A"


def tpot_color(t):
    if t > 10:
        return "#2E8B57"
    if t > 1:
        return "#7BB892"
    return "#B0B0B0"


# ---------------- LEFT PANEL: Throughput ----------------
bars_th = axL.bar(
    x,
    tps,
    color=[th_color(g) for g in tps],
    edgecolor="black",
    linewidth=0.7,
    width=0.62,
)
axL.axhspan(5, 55, alpha=0.06, color="#2E8B57", zorder=0)
axL.axhspan(-3, 5, alpha=0.05, color="#888888", zorder=0)
axL.axhspan(-15, -3, alpha=0.06, color="#C04A4A", zorder=0)
axL.axhline(0, color="black", linewidth=0.9, zorder=1)

for i, (bar, g) in enumerate(zip(bars_th, tps)):
    h = bar.get_height()
    label = f"{g:+.1f}%"
    if h >= 0:
        axL.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.5,
            label,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    else:
        axL.text(
            bar.get_x() + bar.get_width() / 2,
            h - 1.5,
            label,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

axL.set_xticks(x)
axL.set_xticklabels([f"{gpu}\n{bw} TB/s" for gpu, bw in zip(gpus, bws)], fontsize=10.5)
axL.set_ylabel(r"EB$^+$ throughput gain over v1 (%)", fontsize=11.5)
axL.set_ylim(-12, 52)
axL.set_yticks(range(-10, 51, 10))
axL.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
axL.set_title(
    "Throughput (higher is better for EB$^+$)", fontsize=12, fontweight="bold", pad=8
)

# Trend annotation — arrow rises from B300 (left, ≈0%) to L40S (right, +42%)
axL.annotate(
    "",
    xy=(3.18, 36),
    xytext=(0.05, 1),
    arrowprops=dict(
        arrowstyle="->", color="#444", lw=1.4, connectionstyle="arc3,rad=-0.20"
    ),
    zorder=2,
)
axL.text(
    0.75,
    32,
    "Bandwidth ↓\n→ EB$^+$ advantage grows",
    fontsize=9.5,
    color="#333",
    style="italic",
    ha="left",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888", alpha=0.95
    ),
    zorder=3,
)

# ---------------- RIGHT PANEL: TPOT reduction ----------------
bars_tpot = axR.bar(
    x,
    tpots,
    color=[tpot_color(t) for t in tpots],
    edgecolor="black",
    linewidth=0.7,
    width=0.62,
)
axR.axhspan(10, 45, alpha=0.06, color="#2E8B57", zorder=0)
axR.axhspan(0, 10, alpha=0.04, color="#888888", zorder=0)
axR.axhline(0, color="black", linewidth=0.9, zorder=1)

for i, (bar, t) in enumerate(zip(bars_tpot, tpots)):
    h = bar.get_height()
    axR.text(
        bar.get_x() + bar.get_width() / 2,
        h + 1.0,
        f"−{t:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

axR.set_xticks(x)
axR.set_xticklabels([f"{gpu}\n{bw} TB/s" for gpu, bw in zip(gpus, bws)], fontsize=10.5)
axR.set_ylabel(r"EB$^+$ TPOT reduction vs v1 (%)", fontsize=11.5)
axR.set_ylim(-2, 45)
axR.set_yticks(range(0, 46, 10))
axR.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
axR.set_title(
    "TPOT (higher reduction is better for EB$^+$)",
    fontsize=12,
    fontweight="bold",
    pad=8,
)

# (No annotation box on TPOT panel — title + data labels suffice.)

# Suptitle
fig.suptitle(
    "EB$^+$ outperforms MB on bandwidth-constrained GPUs\n"
    r"$\it{Qwen3\!-\!8B,\ WildChat\ workload}$",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()
out_path = "/data/yuzhou/projects/aproj/vllm-sched/eb-vllm/assets/eb_vllm_header.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
