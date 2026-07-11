"""
Reusable plotting helpers that appeared in multiple notebooks.

Only the genuinely-duplicated, configuration-independent plots live
here.  Notebook-specific figures (the bespoke layouts for Figures 1–7
of the paper) stay in their respective notebooks, since they each
contain visual tuning specific to one panel and aren't reused.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl


def plot_metric_boxplots_two_panels(
    results_lp: pd.DataFrame,
    model_names: dict[int, str],
    model_order: list[int],
    train_fraction: float,
    metrics: tuple[str, str] = ("accuracy", "f1"),
    ratio: tuple[float, float] = (6, 3),
    scale: float = 2,
    width_ratios: list[float] = [1, 1],
    xlims: list[tuple[float, float]] | None = None,
    cmap_name: str = "tab10",
):
    """Two-panel horizontal-boxplot view of a metric per model.

    For each model, the prompt-level distribution of ``metric_mean`` is
    drawn as a violin with a narrow box overlay.  Used for the
    accuracy/F1 panels in Section 4.2 (Figure 4) and for the saturation
    plots in the training-size sensitivity analysis.
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=[scale * x for x in ratio],
        sharey=True,
        width_ratios=width_ratios,
    )

    cmap = plt.get_cmap(cmap_name)

    df_prompt = (
        results_lp
        .groupby(["model_id", "prompt_id", "train_fraction"])
        .agg(
            f1_mean=("f1", "mean"),
            accuracy_mean=("accuracy", "mean"),
            mean_n_train=("n_train", "mean"),
        )
        .reset_index()
    )

    for it, metric in enumerate(metrics):
        ax = axes[it]
        data: list = []
        labels: list = []
        colors: list = []

        for i, mid in enumerate(reversed(model_order)):
            name = model_names[mid]
            vals = df_prompt[
                (df_prompt["model_id"] == mid)
                & (df_prompt["train_fraction"] == train_fraction)
            ][f"{metric}_mean"].values
            if len(vals) > 0:
                data.append(vals)
                labels.append(name)
                colors.append(cmap(i % cmap.N))

        positions = np.arange(1, len(data) + 1)

        # violins
        vp = ax.violinplot(
            data,
            widths=0.75,
            positions=positions,
            vert=False,
            showmeans=False,
            showextrema=False,
        )
        for body, c in zip(vp["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.35)
            body.set_edgecolor("none")

        # boxes
        bp = ax.boxplot(
            data,
            positions=positions,
            vert=False,
            widths=0.25,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for box, c in zip(bp["boxes"], colors):
            box.set_facecolor(c)
            box.set_alpha(0.6)

        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_title(metric.capitalize())
        ax.grid(True, axis="x", alpha=0.4)
        if xlims is not None:
            ax.set_xlim(xlims[it])

    fig.tight_layout()
    return fig, axes


# ── Palette ───────────────────────────────────────────────────────────────────

EDGE_COLORS = {
    "GG": "#6E9B34",  # green
    "HH": "#AA4D39",  # red
    "GH": "#27586B",  # blue
}

VERTEX_COLORS = {
    "G": "#6E9B34",
    "H": "#AA4D39",
}

BORDER_COLOR = 'black'

class MK:
    BG      = "#272822"   # mkBg      figure / canvas background
    PANEL   = "#3E3D32"   # mkLight   axes background
    DARK    = "#1E1F1C"   # mkDark    deep background
    TEXT    = "#F8F8F2"   # mkText    near-white body text
    GREEN   = "#A6E22E"   # mkGreen   C0  (was #77AC30)
    PINK    = "#F92672"   # mkPink    C1  (was #D95319)
    CYAN    = "#66D9E8"   # mkCyan
    YELLOW  = "#E6DB74"   # mkYellow
    ORANGE  = "#FD971F"   # mkOrange
    PURPLE  = "#AE81FF"   # mkPurple
    COMMENT = "#75715E"   # mkComment muted / secondary / spines
    # Additional for tab10
    RED     = "#FF6188"
    SAND    = "#C9B37E"
    LIME    = "#B8E986"
    GRAY    = "#CFCFC2"


# ── Default colour cycle (used by plot(), scatter() when no colour given) ─────
_CYCLE   = [MK.GREEN, MK.PINK, MK.CYAN, MK.YELLOW, MK.ORANGE, MK.PURPLE]
MK_TAB_10  = [MK.GREEN, MK.PINK, MK.SAND, MK.ORANGE, MK.PURPLE, MK.RED, MK.CYAN, MK.YELLOW, MK.LIME, MK.GRAY]
TAB_10 = plt.get_cmap("tab10")

def apply_monokai():
    """Apply the Monokai dark theme as global rcParams."""
    mpl.rcParams.update({
        # ── Fonts ─────────────────────────────────────────────────────────
        "font.family":            "serif",
        "font.serif":             ["Computer Modern"],
        "font.size":              14,
        "text.usetex":            True,
        "text.latex.preamble":    r"\usepackage{amsfonts}",
        "text.color":             MK.TEXT,

        # ── Figure ────────────────────────────────────────────────────────
        "figure.facecolor":       MK.BG,
        "figure.edgecolor":       MK.BG,

        # ── Save: transparent so figures overlay cleanly on dark slides ───
        "savefig.facecolor":      "none",
        "savefig.edgecolor":      "none",
        "savefig.transparent":    True,

        # ── Axes ──────────────────────────────────────────────────────────
        "axes.facecolor":         MK.PANEL,
        "axes.edgecolor":         MK.COMMENT,
        "axes.labelcolor":        MK.TEXT,
        "axes.titlecolor":        MK.TEXT,
        "axes.spines.top":        False,
        "axes.spines.right":      False,
        "axes.prop_cycle":        mpl.cycler(color=_CYCLE),

        # ── Ticks ─────────────────────────────────────────────────────────
        "xtick.color":            MK.TEXT,
        "ytick.color":            MK.TEXT,
        "xtick.labelcolor":       MK.TEXT,
        "ytick.labelcolor":       MK.TEXT,

        # ── Grid ──────────────────────────────────────────────────────────
        "grid.color":             MK.COMMENT,
        "grid.alpha":             0.4,
        "grid.linestyle":         "--",

        # ── Legend ────────────────────────────────────────────────────────
        "legend.facecolor":       MK.PANEL,
        "legend.edgecolor":       MK.COMMENT,
        "legend.labelcolor":      MK.TEXT,

        # ── Scatter / patches ─────────────────────────────────────────────
        "patch.edgecolor":        MK.COMMENT,
    })

    TAB_10.colors = MK_TAB_10

    EDGE_COLORS["GG"] = MK.GREEN
    EDGE_COLORS["HH"] = MK.PINK
    EDGE_COLORS["GH"] = MK.CYAN

    VERTEX_COLORS["G"] = MK.GREEN
    VERTEX_COLORS["H"] = MK.PINK

    global BORDER_COLOR
    BORDER_COLOR = MK.COMMENT