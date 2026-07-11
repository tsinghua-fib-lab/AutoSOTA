"""Global Matplotlib styling for paper-quality figures.

This is intended to be called once at import-time by plotting modules.
"""

from __future__ import annotations


def apply_mpl_style() -> None:
    """Apply a global Matplotlib style (ICML-ish, clean + legible).

    Safe to call multiple times.
    """
    import matplotlib as mpl
    import matplotlib.style as mplstyle
    from cycler import cycler

    # A clean base style; ships with Matplotlib (>=3.6 typically).
    # If unavailable in a given environment, we just rely on rcParams below.
    try:
        mplstyle.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    # Colorblind-friendly-ish palette.
    colors = [
        "#4C78A8",  # blue
        "#F58518",  # orange
        "#54A24B",  # green
        "#E45756",  # red
        "#72B7B2",  # teal
        "#B279A2",  # purple
        "#FF9DA6",  # pink
        "#9D755D",  # brown
        "#BAB0AC",  # gray
    ]

    mpl.rcParams.update(
        {
            # Typography
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            # Lines / markers
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            # Axes / grid
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler("color", colors),
            # Figure / savefig
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
