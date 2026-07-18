import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict


def set_dave_style_plot():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.3",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.8,
        "xtick.color": "0.2",
        "ytick.color": "0.2",
        "text.color": "0.1",
        "axes.labelcolor": "0.1",
    })


def plot_curves(
    out_path: Path,
    curves_mean: Dict[str, np.ndarray],
    num_pixels_frac: float,
    title: str,
):
    set_dave_style_plot()
    plt.figure(figsize=(9, 4))
    
    for name, mean_curve in curves_mean.items():
        x = np.linspace(0, 100 * num_pixels_frac, len(mean_curve))
        plt.plot(x, mean_curve, linewidth=2, label=name, color="#8E7DBE")

    plt.grid(alpha=1, linestyle="dashed")
    plt.xlabel("% of pixels masked", fontsize=12)
    plt.ylabel("Target score (softmax)", fontsize=12)
    plt.xlim(0, 90)
    plt.ylim(0.2, 1.0)
    plt.title(title, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
