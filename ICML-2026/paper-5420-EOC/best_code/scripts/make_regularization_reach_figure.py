#!/usr/bin/env python
"""Generate the regularization-reach weight figure for App. B.4.

Produces a single-panel plot of the per-layer weight
    w_ell = xi_c * (1 - exp(-(L - ell) / xi_c))
for several values of L / xi_c, illustrating monotone decrease in ell and the
saturation that controls the comparative statics.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dropout_mft.paths import project_root
from dropout_mft.plotting import save_figure
from dropout_mft.style import COLORS, apply_paper_style

REPO_ROOT = project_root()
OUT = REPO_ROOT / "figures" / "paper"


def w_ell(ell: np.ndarray, L: int, xi_c: float) -> np.ndarray:
    return xi_c * (1.0 - np.exp(-(L - ell) / xi_c))


def reach_ratio(tau: np.ndarray, f: float) -> np.ndarray:
    inv_tau = 1.0 / tau
    numer = f - inv_tau * (np.exp(-tau * (1.0 - f)) - np.exp(-tau))
    denom = 1.0 - inv_tau * (1.0 - np.exp(-tau))
    return (numer / denom) / f


def make_figure() -> Path:
    apply_paper_style()

    L = 12
    ell = np.arange(1, L + 1)
    xi_values = [2.0, 4.0, 8.0, 24.0]
    palette = [
        COLORS["kink"],
        COLORS["teal"],
        COLORS["smooth"],
        COLORS["baseline"],
    ]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))

    for xi, color in zip(xi_values, palette):
        label = rf"$\xi_c = {xi:g}$" if np.isfinite(xi) else r"$\xi_c \to \infty$"
        ax.plot(ell, w_ell(ell, L, xi), marker="o", color=color, label=label)

    f = 1.0 / 3.0
    f_layers = int(round(f * L))
    ax.axvspan(
        0.5,
        f_layers + 0.5,
        color=COLORS["teal"],
        alpha=0.08,
        linewidth=0,
        label="step (early), $f=1/3$",
    )
    ax.set_xlim(0.5, L + 0.5)
    ax.set_xlabel(r"layer index $\ell$")
    ax.set_ylabel(r"regularization reach $w_\ell$")
    ax.set_title(r"(a) Per-layer reach $w_\ell = \xi_c\,(1-e^{-(L-\ell)/\xi_c})$")
    ax.legend(loc="upper right", fontsize=8.5)

    tau = np.logspace(np.log10(0.2), np.log10(20.0), 200)
    for fval, color, ls in [
        (1.0 / 3.0, COLORS["teal"], "-"),
        (1.0 / 2.0, COLORS["dark_gold"], "--"),
    ]:
        ax2.plot(
            tau,
            reach_ratio(tau, fval),
            color=color,
            linestyle=ls,
            label=rf"$f = {fval:.2f}$",
        )
    ax2.axhline(1.0, color=COLORS["neutral"], linewidth=1.0, linestyle=":")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\tau \equiv L/\xi_c$")
    ax2.set_ylabel(r"$\mathcal{R}(\tau, f) = \mathcal{V}_{\rm step}/\mathcal{V}_{\rm const}$")
    ax2.set_title("(b) Front-loaded vs. constant, at fixed budget")
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_path = OUT / "regularization_reach.pdf"
    save_figure(fig, fig_path)
    plt.close(fig)
    print(f"wrote {fig_path}")
    return fig_path


def copy_to_paper(fig_path: Path, dest: Path) -> None:
    target_pdf = dest / "theory" / "regularization_reach.pdf"
    target_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig_path, target_pdf)
    print(f"copied {fig_path.relative_to(REPO_ROOT)} -> {target_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--copy-to-paper",
        type=Path,
        metavar="DIR",
        help="Copy the figure into the given LaTeX paper figures/ directory.",
    )
    args = parser.parse_args()

    fig_path = make_figure()
    if args.copy_to_paper is not None:
        copy_to_paper(fig_path, args.copy_to_paper)


if __name__ == "__main__":
    main()
