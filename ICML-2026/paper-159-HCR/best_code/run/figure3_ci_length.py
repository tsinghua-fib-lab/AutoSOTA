"""Generate Figure 3 (CI length across calibration size) from the paper.

This recreates ``figure3_confidence–interval length across calibration size.ipynb``
without requiring a notebook run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib
import numpy as np

from llm_judge_reporting.allocation import allocate_calibration_sample
from llm_judge_reporting.calibration import confidence_interval

matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["font.size"] = 16

# Import pyplot after rcParams are set.
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUTPUT = Path("figures/figure3_ci_length.png")
LEGEND_FONT_SIZE = 14


def parse_p_list(raw: str) -> List[float]:
    """Parse comma-separated p values."""
    return [float(val.strip()) for val in raw.split(",") if val.strip()]


def compute_ci_lengths(
    p: float, q0: float, q1: float, n: int, m_values: Iterable[int], alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return CI lengths for equal split and adaptive split strategies."""
    equal_ci, adapt_ci = [], []
    for m in m_values:
        ci_equal = confidence_interval(p, q0, q1, n, m // 2, m // 2, alpha)
        equal_ci.append(ci_equal[1] - ci_equal[0])

        m0, m1 = allocate_calibration_sample(m, p, q0, q1, 0)
        ci_adapt = confidence_interval(p, q0, q1, n, m0, m1, alpha)
        adapt_ci.append(ci_adapt[1] - ci_adapt[0])
    return np.array(equal_ci), np.array(adapt_ci)


def generate_figure(
    p_list: Iterable[float],
    q0: float = 0.7,
    q1: float = 0.9,
    alpha: float = 0.05,
    n: int = int(1e9),
    m_start: int = 100,
    m_stop: int = 10000,
    m_step: int = 10,
) -> tuple[plt.Figure, np.ndarray]:
    """Build the CI length curves for equal vs adaptive calibration splits."""
    m_values = np.arange(m_start, m_stop + 1, m_step)
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for p, color in zip(p_list, colors, strict=False):
        equal_ci, adapt_ci = compute_ci_lengths(p, q0, q1, n, m_values, alpha)
        ax.plot(m_values, adapt_ci, "-", color=color, linewidth=2, label=rf"$\hat{{p}}={p}$")
        ax.plot(m_values, equal_ci, "--", color=color, linewidth=2, alpha=0.5)

    ax.set_xlim(m_values.min(), m_values.max())
    ax.set_xlabel(r"calibration set size ($m_0+m_1$)")
    ax.set_ylabel("confidence interval length")
    ax.set_xscale("log")
    ax.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.legend(prop={"size": LEGEND_FONT_SIZE})
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, m_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3: CI length vs calibration budget (equal vs adaptive)."
    )
    parser.add_argument("--p-list", type=str, default="0.3,0.5,0.7,0.9", help="Comma-separated p values.")
    parser.add_argument("--q0", type=float, default=0.7, help="Specificity q0 (default: 0.7).")
    parser.add_argument("--q1", type=float, default=0.9, help="Sensitivity q1 (default: 0.9).")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
    parser.add_argument("--n", type=int, default=int(1e9), help="Test set size n (default: 1e9).")
    parser.add_argument("--m-start", type=int, default=100, help="Minimum calibration size (default: 100).")
    parser.add_argument("--m-stop", type=int, default=10000, help="Maximum calibration size (default: 10000).")
    parser.add_argument("--m-step", type=int, default=10, help="Step for calibration size grid (default: 10).")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to save the PNG (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p_list = parse_p_list(args.p_list)
    fig, _ = generate_figure(
        p_list=p_list,
        q0=args.q0,
        q1=args.q1,
        alpha=args.alpha,
        n=args.n,
        m_start=args.m_start,
        m_stop=args.m_stop,
        m_step=args.m_step,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, bbox_inches="tight", dpi=300)
        print(f"Saved figure to {args.output}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
