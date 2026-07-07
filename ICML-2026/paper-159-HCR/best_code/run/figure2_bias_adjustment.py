import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
import numpy as np

from llm_judge_reporting.calibration import confidence_interval, point_estimator

matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["font.size"] = 18

# Lazily import pyplot after configuring rcParams.
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUTPUT = Path("figures/figure2_bias_adjustment.png")
LEGEND_FONT_SIZE = 14

def expected_observed_rate(theta: np.ndarray, q0: float, q1: float) -> Tuple[np.ndarray, float]:
    """Return expected observed rate p_hat and turning point where bias sign flips."""
    p_hat = (q0 + q1 - 1) * theta + (1 - q0)
    turning_point = (1 - q0) / (2 - q0 - q1)
    return p_hat, turning_point

def adjusted_estimates(
    p_hat: Iterable[float], q0: float, q1: float, n: int, m0: int, m1: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bias-adjusted point estimates and confidence intervals across p_hat."""
    theta_hats, ci_lows, ci_highs = [], [], []
    for p_temp in p_hat:
        theta_hat = point_estimator(p_temp, q0, q1)
        ci_low, ci_high = confidence_interval(p_temp, q0, q1, n, m0, m1, alpha=0.05)
        theta_hats.append(theta_hat)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
    return np.array(theta_hats), np.array(ci_lows), np.array(ci_highs)


def plot_bias(ax: plt.Axes, theta: np.ndarray, p_hat: np.ndarray, turning_point: float, q0: float, q1: float) -> None:
    """Plot bias in the raw observed rate."""
    mask_low = theta <= turning_point
    ax.plot(
        theta[mask_low],
        p_hat[mask_low],
        color="#4D96FF",
        linewidth=3,
        label=r"$\mathbb{E}\;[\hat{p}]$ (overestimated)",
    )
    ax.plot(
        theta[~mask_low],
        p_hat[~mask_low],
        color="#FF6B6B",
        linewidth=3,
        label=r"$\mathbb{E}\;[\hat{p}]$ (underestimated)",
    )

    ax.axline((0, 0), slope=1, color="black", alpha=0.4, linestyle="--")

    ax.annotate(
        "",
        xy=(0, 1 - q0),
        xytext=(0, 0),
        color="#6BCB77",
        arrowprops=dict(arrowstyle="->", linewidth=2, color="#6BCB77"),
    )
    ax.text(0.015, 0.24, r"$\mathbf{1}$-$\mathbf{q_0}$", color="#6BCB77", ha="left")

    ax.annotate(
        "",
        xy=(1, q1),
        xytext=(1, 1),
        color="#6BCB77",
        arrowprops=dict(arrowstyle="->", linewidth=2, color="#6BCB77"),
    )
    ax.text(0.835, 0.93, r"$\mathbf{1}$-$\mathbf{q_1}$", color="#6BCB77", ha="left")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"true accuracy ($\theta$)")
    ax.set_ylabel(r"the expected value")
    ax.legend(prop={"size": LEGEND_FONT_SIZE})
    ax.grid(alpha=0.3)


def plot_adjustment(ax: plt.Axes, theta: np.ndarray, theta_hats: np.ndarray, ci_lows: np.ndarray, ci_highs: np.ndarray) -> None:
    """Plot bias-adjusted estimates and confidence intervals."""
    ax.plot(theta, theta_hats, linewidth=3, color="black", label=r"$\mathbb{E}\;[\hat{\theta}]$ (unbiased)")
    ax.fill_between(theta, ci_lows, ci_highs, color="gray", alpha=0.3, label="95% CI")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"true accuracy ($\theta$)")
    ax.set_ylabel(r"the expected value")
    ax.legend(prop={"size": LEGEND_FONT_SIZE})
    ax.grid(alpha=0.3)


def generate_figure(
    q0: float = 0.7,
    q1: float = 0.9,
    n: int = int(1e9),
    m0: int = 500,
    m1: int = 500,
    points: int = 200,
) -> plt.Figure:
    """Build the two-panel figure and return the matplotlib Figure."""
    theta = np.linspace(0, 1, points)
    p_hat, turning_point = expected_observed_rate(theta, q0, q1)
    theta_hats, ci_lows, ci_highs = adjusted_estimates(p_hat, q0, q1, n, m0, m1)

    fig, (ax_bias, ax_adjustment) = plt.subplots(1, 2, figsize=(10, 5))
    plot_bias(ax_bias, theta, p_hat, turning_point, q0, q1)
    plot_adjustment(ax_adjustment, theta, theta_hats, ci_lows, ci_highs)

    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 2 (bias and its adjustment).")
    parser.add_argument("--q0", type=float, default=0.7, help="Specificity q0 (default: 0.7).")
    parser.add_argument("--q1", type=float, default=0.9, help="Sensitivity q1 (default: 0.9).")
    parser.add_argument("--n", type=int, default=int(1e9), help="Test set size n (default: 1e9).")
    parser.add_argument("--m0", type=int, default=500, help="Calibration size m0 (default: 500).")
    parser.add_argument("--m1", type=int, default=500, help="Calibration size m1 (default: 500).")
    parser.add_argument("--points", type=int, default=200, help="Number of grid points for theta (default: 200).")
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
    fig = generate_figure(q0=args.q0, q1=args.q1, n=args.n, m0=args.m0, m1=args.m1, points=args.points)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, bbox_inches="tight", dpi=300)
        print(f"Saved figure to {args.output}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
