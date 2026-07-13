"""Plot style used for the paper figures."""

from __future__ import annotations

import matplotlib as mpl


COLORS = {
    "smooth": "#3A5F3A",
    "kink": "#C9A961",
    "fit": "#5C7A5C",
    "theory": "#D4AF6A",
    "grid": "#E8E0D0",
    "baseline": "#303030",
    "neutral": "#777777",
    "teal": "#00897B",
    "dark_gold": "#9A6A2F",
    "rose": "#AA3377",
    "soft_rose": "#B85C8A",
    "muted_blue": "#4477AA",
}

SCHEDULE_COLORS = {
    "none": COLORS["baseline"],
    "constant": COLORS["kink"],
    "reverse_step": COLORS["smooth"],
    "big_step": COLORS["teal"],
    "step": COLORS["teal"],
    "linear": COLORS["dark_gold"],
    "reverse_linear": COLORS["theory"],
    "double": COLORS["soft_rose"],
    "triple": "#7A4E00",
    "v_shape": "#CC3311",
    "inverse_v": "#0077BB",
}

FIELD_PALETTE = [
    COLORS["smooth"],
    COLORS["kink"],
    "#E63946",
    "#1E88E5",
    "#7B2CBF",
    "#FF6F00",
    COLORS["teal"],
    "#C62828",
    "#5E35B1",
    COLORS["dark_gold"],
]

ABLATION_MODE_STYLES = {
    "both": "-",
    "attn_only": "--",
    "mlp_only": ":",
}

ABLATION_MODE_COLORS = {
    "both": COLORS["smooth"],
    "attn_only": COLORS["theory"],
    "mlp_only": COLORS["muted_blue"],
}

ABLATION_MODE_NAMES = {
    "both": "Both (Attn + MLP)",
    "attn_only": "Attention only",
    "mlp_only": "MLP only",
}


def schedule_color(schedule: str, default: str = "#333333") -> str:
    return SCHEDULE_COLORS.get(schedule, default)


def apply_paper_style() -> None:
    """Use the same basic styling across the paper plots."""

    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.edgecolor": "#333333",
            "axes.labelsize": 12.5,
            "axes.titlesize": 13.5,
            "axes.titleweight": "regular",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.2,
            "lines.markersize": 5.4,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.6,
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#C8C8C8",
        }
    )
