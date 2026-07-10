"""
Nature-style roadmap figure for the theoretical analysis section.

Replaces Fig/flowchart.pdf with a cleaner, vector-only design:
- Four numbered stages connected by arrows
- Soft palette (Tableau-style, color-blind friendly)
- LaTeX math rendered through matplotlib
- Tight bounding box, no extra whitespace
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "Fig", "flowchart.pdf")
OUT = os.path.abspath(OUT)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Layout: 4 stages stacked vertically
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.6)
ax.set_aspect("equal")
ax.axis("off")

# Stage palette — soft pastel, all from the same hue family
PALETTE = ["#e8f0fb", "#dceaf6", "#c9dcee", "#b3cce4"]
EDGE    = ["#3b6ba5", "#3b6ba5", "#3b6ba5", "#3b6ba5"]
NUMCOL  = "#1f3b66"

# Stage definitions: (number, title, body_math)
stages = [
    ("1", "Problem setup",
     r"$M\subset\mathbb{R}^{N}$ compact;  $\mathcal{F}_{M,D}=C^{0}(M,D)$"),
    ("2", "Cauchy kernel function",
     r"$K(\boldsymbol{\xi},\mathbf{x})=\prod_{i=1}^{N}(\xi^{i}-x_{i})^{-1}$"),
    ("3", "Cauchy approximation theorem",
     r"$\sup_{\mathbf{x}\in M}\,|f(\mathbf{x})-\sum_{k=1}^{m}\theta_{k}\,K(\boldsymbol{\xi}_{k},\mathbf{x})|<\varepsilon$"),
    ("4", "Universal approximation for CauchyNet",
     r"$\sup_{\mathbf{x}\in M}|f(\mathbf{x})-\Re(N_{\mathbf{B},\mathbf{C}}(\mathbf{x}))|<\varepsilon$"),
]

box_w, box_h = 8.8, 1.30
x_left = 0.8
y_centers = [6.5, 4.8, 3.1, 1.4]  # top to bottom, tighter spacing

for (label, title, body), y, fill, edge in zip(stages, y_centers, PALETTE, EDGE):
    # Main rounded rectangle
    box = FancyBboxPatch(
        (x_left, y - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(box)
    # Number circle on the left edge
    ax.add_patch(plt.Circle(
        (x_left + 0.55, y), 0.34,
        facecolor=NUMCOL, edgecolor="none", zorder=4,
    ))
    ax.text(x_left + 0.55, y, label, ha="center", va="center",
            color="white", fontsize=11, fontweight="bold", zorder=5)
    # Title (small caps style)
    ax.text(x_left + 1.20, y + 0.32, title, ha="left", va="center",
            fontsize=10.5, fontweight="bold", color=NUMCOL, zorder=5)
    # Body math
    ax.text(x_left + 1.20, y - 0.32, body, ha="left", va="center",
            fontsize=10.5, color="#222222", zorder=5)

# Arrows between consecutive stages
for y1, y2 in zip(y_centers[:-1], y_centers[1:]):
    arr = FancyArrowPatch(
        (x_left + box_w / 2 + 0.0, y1 - box_h / 2),
        (x_left + box_w / 2 + 0.0, y2 + box_h / 2 + 0.02),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.2, color="#5c7fad", zorder=1,
    )
    ax.add_patch(arr)

fig.tight_layout(pad=0.15)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=240, bbox_inches="tight", pad_inches=0.02)
print(f"Saved {OUT}")
print(f"Saved {OUT.replace('.pdf', '.png')}")
