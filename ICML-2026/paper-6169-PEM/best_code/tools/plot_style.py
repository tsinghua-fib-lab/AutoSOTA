#!/usr/bin/env python3
"""
Unified plotting style for paper figures.

This module provides consistent styling helpers for all plotting scripts under `tools/`.
It is intentionally lightweight and degrades gracefully if optional dependencies
are missing.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

# ============================================================================
# Color palettes
# ============================================================================

# Colorblind-friendly palette (Paul Tol's bright scheme)
COLORS = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
    "black": "#000000",
}

# Algorithm-specific colors (stable across figures)
ALGO_COLORS: dict[str, str] = {
    # Residual bootstrapping
    "BERW": "#0077BB",
    "BERW-Hetero": "#0077BB",
    "BERW-HeteroRobust": "#0077BB",

    # ProbeSwitch
    "ProbeSwitch": "#009988",
    "ProbeSwitch-MR(t=0.12)": "#009988",
    "ProbeSwitch-MR(t=0.120)": "#009988",
    "ProbeSwitch-MR(t=0.18)": "#009988",
    "ProbeSwitch-MR(t=0.22)": "#009988",

    # CMA-ES baseline
    "CMA-ES": "#CC3311",
    "CMA-ES-sep": "#CC3311",
    "Sep-CMA-ES": "#CC3311",

    # Resampling variants
    "CMA-ES-Resample(k=5)": "#EE7733",
    "CMA-ES-Resample(k=10)": "#997700",

    # UH-CMA-ES
    "UH-CMA-ES": "#EE3377",
    "UH-CMA-ES(maxevals=10)": "#EE3377",
    "UH-CMA-ES(maxevals=30)": "#AA4499",
}

PRIMARY_ALGOS = {
    "BERW", "BERW-Hetero", "BERW-HeteroRobust",
    "ProbeSwitch", "ProbeSwitch-MR(t=0.12)", "ProbeSwitch-MR(t=0.120)", "ProbeSwitch-MR(t=0.18)", "ProbeSwitch-MR(t=0.22)",
    "CMA-ES", "CMA-ES-sep", "Sep-CMA-ES",
}

LINEWIDTH_PRIMARY = 0.9
LINEWIDTH_SECONDARY = 0.6
LINEWIDTH_DEFAULT = 0.8

# Standard paper widths (inches)
WIDTHS = {
    "single": 3.5,
    "double": 7.0,
}


_STYLE_APPLIED = False


def _try_import_scienceplots() -> bool:
    try:
        import scienceplots  # noqa: F401
        return True
    except ImportError:
        return False


def _apply_fallback_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.8,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def apply_style(style: str | list[str] = "science", *, usetex: bool = False) -> None:
    """Apply a consistent matplotlib style once per process."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    has_scienceplots = _try_import_scienceplots()
    if has_scienceplots:
        styles = [style] if isinstance(style, str) else list(style)
        if not usetex and "no-latex" not in styles:
            styles.append("no-latex")
        try:
            plt.style.use(styles)
        except OSError:
            _apply_fallback_style()
    else:
        _apply_fallback_style()

    _STYLE_APPLIED = True


def get_figsize(width: str = "single", *, aspect: float = 0.75) -> tuple[float, float]:
    w = WIDTHS.get(width, WIDTHS["single"])
    return float(w), float(w * aspect)


def get_subplot_figsize(rows: int, cols: int, width: str = "double", *, subplot_aspect: float = 0.75) -> tuple[float, float]:
    w = WIDTHS.get(width, WIDTHS["double"])
    return float(w), float(w * subplot_aspect * rows / max(1, cols))


def get_algo_color(algo: str) -> str:
    return ALGO_COLORS.get(str(algo), "#444444")


def get_algo_linewidth(algo: str) -> float:
    if str(algo) in PRIMARY_ALGOS:
        return LINEWIDTH_PRIMARY
    return LINEWIDTH_SECONDARY


def add_grid(ax: Axes, *, which: str = "major", axis: str = "both", alpha: float = 0.25) -> None:
    ax.grid(True, which=which, axis=axis, alpha=float(alpha), linewidth=0.4, color="#888888")


def save_figure(
    fig: Figure,
    path: str,
    *,
    formats: list[str] | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> list[str]:
    """Save a figure (default: pdf+png). Accepts `path` with or without extension."""
    if formats is None:
        formats = ["pdf", "png"]

    base, ext = os.path.splitext(path)
    if ext:
        formats = [ext.lstrip(".")]

    saved: list[str] = []
    for fmt in formats:
        out_path = f"{base}.{fmt}"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(
            out_path,
            format=fmt,
            dpi=dpi if fmt in ("png", "jpg", "jpeg") else None,
            transparent=transparent,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        saved.append(out_path)
    return saved

