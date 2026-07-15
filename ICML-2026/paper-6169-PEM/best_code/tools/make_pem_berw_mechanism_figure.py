#!/usr/bin/env python3
"""
Generate a mechanism diagram for PEM/BERW (+ optional ProbeSwitch inset).

Output:
  - evidence/figures_conceptual/pem_berw_mechanism.png
  - evidence/figures_conceptual/pem_berw_mechanism.pdf
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _project import BASE_DIR, repo_relpath

def _box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    fc: str = "#F8FAFC",
    ec: str = "#334155",
    lw: float = 1.2,
    fontsize: int = 10,
    align: str = "center",
):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        color="#0F172A",
        wrap=True,
    )
    return patch


def _arrow(ax, a: tuple[float, float], b: tuple[float, float], *, color: str = "#334155", lw: float = 1.3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=lw, color=color))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="evidence/figures_conceptual",
        help="Output directory (writes pem_berw_mechanism.(png|pdf)).",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.2, 4.3), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Main pipeline (PEM/BERW).
    ax.text(0.02, 0.96, "PEM/BERW: selection-stage uncertainty integration", fontsize=13, weight="bold", color="#0F172A")

    y0 = 0.58
    h = 0.22
    w = 0.16
    gap = 0.03

    b1 = _box(
        ax,
        (0.03, y0),
        (w, h),
        "Sample offspring\n"
        r"$x_{1:\lambda}\sim\mathcal{N}(m_t,\sigma_t^2 C_t)$",
        fc="#EFF6FF",
    )
    b2 = _box(
        ax,
        (0.03 + (w + gap) * 1, y0),
        (w, h),
        "Noisy eval\n" r"$y_i=f(x_i)+\varepsilon_i$",
        fc="#FFF7ED",
    )
    b3 = _box(
        ax,
        (0.03 + (w + gap) * 2, y0),
        (w, h),
        "Boundary re-eval\n"
        "build residual pool\n"
        r"$\mathcal{Z}$  (standardized)",
        fc="#F0FDF4",
    )
    b4 = _box(
        ax,
        (0.03 + (w + gap) * 3, y0),
        (w, h),
        "Bootstrap ranks\n"
        r"$\tilde y_i^{(b)}=\bar y_i + z\,\hat s(\bar y_i)$"
        "\n"
        r"$\hat w_i=\frac{1}{B}\sum_b w(\mathrm{rank}_i^{(b)})$",
        fc="#F5F3FF",
        fontsize=9,
    )
    b5 = _box(
        ax,
        (0.03 + (w + gap) * 4, y0),
        (w, h),
        "Rank-based update\n"
        r"$m_{t+1}\!=m_t+\eta\sum_i \hat w_i(x_i-m_t)$",
        fc="#F8FAFC",
        fontsize=9,
    )

    def _mid_right(patch):
        x, y = patch.get_x(), patch.get_y()
        return (x + patch.get_width(), y + patch.get_height() / 2.0)

    def _mid_left(patch):
        x, y = patch.get_x(), patch.get_y()
        return (x, y + patch.get_height() / 2.0)

    _arrow(ax, _mid_right(b1), _mid_left(b2))
    _arrow(ax, _mid_right(b2), _mid_left(b3))
    _arrow(ax, _mid_right(b3), _mid_left(b4))
    _arrow(ax, _mid_right(b4), _mid_left(b5))

    # Inset: ProbeSwitch as cost-aware decision.
    ax.text(0.02, 0.42, "ProbeSwitch (optional): cost-aware gating to avoid negative transfer", fontsize=12, weight="bold", color="#0F172A")
    y1 = 0.10
    h2 = 0.24
    w2 = 0.24
    b6 = _box(
        ax,
        (0.03, y1),
        (w2, h2),
        "Probe (cheap)\n"
        "two noisy draws on a small set\n"
        r"$p \leftarrow \mathrm{RD}(y^{(1)},y^{(2)})$",
        fc="#FEF2F2",
        fontsize=10,
    )
    b7 = _box(
        ax,
        (0.03 + w2 + 0.05, y1),
        (0.22, h2),
        "Decision (VOI)\n"
        r"if $p\geq \tau$ use BERW\n"
        r"else use CMA",
        fc="#F1F5F9",
        fontsize=10,
    )
    b8 = _box(
        ax,
        (0.03 + w2 + 0.05 + 0.22 + 0.05, y1),
        (0.36, h2),
        "Run remaining budget\n"
        "— CMA in low-misranking regime\n"
        "— BERW-Hetero/Robust in high-misranking regime\n"
        "(budget-aware sample-efficiency trade-off)",
        fc="#ECFEFF",
        fontsize=10,
    )
    _arrow(ax, _mid_right(b6), _mid_left(b7))
    _arrow(ax, _mid_right(b7), _mid_left(b8))

    out_png = out_dir / "pem_berw_mechanism.png"
    out_pdf = out_dir / "pem_berw_mechanism.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", repo_relpath(str(out_png)))
    print("Wrote:", repo_relpath(str(out_pdf)))


if __name__ == "__main__":
    main()
