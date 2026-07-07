import argparse
from math import ceil, sqrt
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

from llm_judge_reporting.allocation import allocate_calibration_sample
from llm_judge_reporting.calibration import clip, confidence_interval, point_estimator

matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["font.size"] = 16

# Import pyplot after rcParams are set.
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUTPUT = Path("figures/figure4_monte_carlo.png")
LEGEND_FONT_SIZE = 14


def confidence_interval_for_p_hat(p: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    z = norm.ppf(1 - alpha / 2)
    se = sqrt(p * (1 - p) / n)
    return clip(p - z * se), clip(p + z * se)


def _calculate_result(theta: float, p_hat: float, q0_hat: float, q1_hat: float, n: int, m0: int, m1: int, alpha: float):
    result = {"theta": theta}

    result["theta_hat"] = point_estimator(p_hat, q0_hat, q1_hat)
    result["theta_lo"], result["theta_hi"] = confidence_interval(p_hat, q0_hat, q1_hat, n, m0, m1, alpha=alpha)
    result["theta_covered"] = (result["theta_lo"] <= theta) and (theta <= result["theta_hi"])

    result["p_hat"] = p_hat
    result["p_lo"], result["p_hi"] = confidence_interval_for_p_hat(p_hat, n, alpha=alpha)
    result["p_covered"] = (result["p_lo"] <= theta) and (theta <= result["p_hi"])

    return result


def simulate_once(theta: float, q0: float, q1: float, n: int, m: int, rng: np.random.Generator, m_pilot: int, alpha: float):
    assert m >= 2 * m_pilot, f"m must be at least {2 * m_pilot} to include the pilot samples"
    assert m % 2 == 0, "m must be an even integer"

    result = {"equal": {}, "adaptive": {}}

    # Test dataset: simulate true labels and noisy judge.
    n_true = rng.binomial(1, theta, size=n).sum()
    p_hat = (rng.binomial(1, 1 - q0, size=n - n_true).sum() + rng.binomial(1, q1, size=n_true).sum()) / n

    # Equal-split calibration.
    m0, m1 = int(m / 2), int(m / 2)
    q0_hat = rng.binomial(1, q0, size=m0).sum() / m0
    q1_hat = rng.binomial(1, q1, size=m1).sum() / m1
    result["equal"] = _calculate_result(theta, p_hat, q0_hat, q1_hat, n, m0, m1, alpha)

    # Adaptive allocation with pilot samples.
    m0_q0_temp = rng.binomial(1, q0, size=m_pilot).sum()
    m1_q1_temp = rng.binomial(1, q1, size=m_pilot).sum()

    q0_temp = m0_q0_temp / m_pilot
    q1_temp = m1_q1_temp / m_pilot

    m0, m1 = allocate_calibration_sample(m, p_hat, q0_temp, q1_temp, m_pilot)

    q0_hat = (rng.binomial(1, q0, size=m0 - m_pilot).sum() + m0_q0_temp) / m0
    q1_hat = (rng.binomial(1, q1, size=m1 - m_pilot).sum() + m1_q1_temp) / m1

    result["adaptive"] = _calculate_result(theta, p_hat, q0_hat, q1_hat, n, m0, m1, alpha)
    return result


def run_simulation(
    thetas: np.ndarray,
    replication: int,
    q0: float,
    q1: float,
    n: int,
    m: int,
    m_pilot: int,
    alpha: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows_equal, rows_adaptive = [], []

    for theta in thetas:
        for _ in range(replication):
            result = simulate_once(theta, q0, q1, n, m, rng, m_pilot=m_pilot, alpha=alpha)
            rows_equal.append(result["equal"])
            rows_adaptive.append(result["adaptive"])

    return pd.DataFrame(rows_equal), pd.DataFrame(rows_adaptive)


def build_figure(df_equal: pd.DataFrame, df_adaptive: pd.DataFrame) -> plt.Figure:
    df_equal_sample = df_equal.drop_duplicates(subset="theta", keep="first")
    df_equal_mean = df_equal.groupby("theta", as_index=False).mean()
    df_adaptive_mean = df_adaptive.groupby("theta", as_index=False).mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: estimator & CI bounds vs theta.
    ax = axes[0]
    ax.plot(df_equal_sample.theta, df_equal_sample.p_hat, "o-", label=r"$\hat{p}$", color="#4D96FF")
    ax.fill_between(
        df_equal_sample.theta,
        df_equal_sample.p_lo,
        df_equal_sample.p_hi,
        color="#4D96FF",
        alpha=0.3,
        label=r"$\hat{p}+{\pm}z_{\alpha}SE$",
    )
    ax.plot(df_equal_sample.theta, df_equal_sample.theta_hat, "o-", label=r"$\hat{\theta}$", color="black")
    ax.fill_between(
        df_equal_sample.theta,
        df_equal_sample.theta_lo,
        df_equal_sample.theta_hi,
        color="lightgray",
        alpha=0.4,
        label=r"$\tilde{\theta}+d\tilde{\theta}{\pm}z_{\alpha}SE$",
    )
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"true accuracy ($\theta$)")
    ax.set_ylabel(r"estimates")
    ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE)
    ax.grid(alpha=0.3)

    # Plot 2: coverage probability.
    ax = axes[1]
    ax.plot(
        df_equal_mean.theta,
        df_equal_mean.p_covered,
        marker="o",
        linestyle="-",
        color="#4D96FF",
        label=r"$\hat{p}+{\pm}z_{\alpha}SE$",
    )
    ax.plot(
        df_equal_mean.theta,
        df_equal_mean.theta_covered,
        marker="o",
        linestyle="-",
        color="black",
        label=r"$\tilde{\theta}+d\tilde{\theta}{\pm}z_{\alpha}SE$",
    )
    ax.axhline(0.95, linestyle="--", linewidth=1, label="95%", color="black")
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"true accuracy ($\theta$)")
    ax.set_ylabel(r"coverage probability")
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.grid(alpha=0.3)

    # Plot 3: CI length.
    ax = axes[2]
    ax.plot(
        df_equal_mean.theta,
        df_equal_mean.theta_hi - df_equal_mean.theta_lo,
        "o--",
        label=r"$m_0=m_1$",
        color="gray",
    )
    ax.plot(
        df_adaptive_mean.theta,
        df_adaptive_mean.theta_hi - df_adaptive_mean.theta_lo,
        "o-",
        label=r"$m_0\approx (1/\hat{p}-1)\sqrt{\kappa} \cdot m_1$",
        color="black",
    )
    max_ylim = ceil(
        max(
            (df_equal_mean.theta_hi - df_equal_mean.theta_lo).max(),
            (df_adaptive_mean.theta_hi - df_adaptive_mean.theta_lo).max(),
        )
        * 20
    ) / 20
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max_ylim)
    ax.set_xlabel(r"true accuracy ($\theta$)")
    ax.set_ylabel(r"confidence interval length")
    ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 4: Monte Carlo simulation plots.")
    parser.add_argument("--theta-start", type=float, default=0.0, help="Start of theta grid (default: 0).")
    parser.add_argument("--theta-stop", type=float, default=1.0, help="End of theta grid (default: 1).")
    parser.add_argument("--theta-num", type=int, default=21, help="Number of theta grid points (default: 21).")
    parser.add_argument("--replication", type=int, default=10_000, help="Number of replicates per theta (default: 10000).")
    parser.add_argument("--q0", type=float, default=0.7, help="Specificity q0 (default: 0.7).")
    parser.add_argument("--q1", type=float, default=0.9, help="Sensitivity q1 (default: 0.9).")
    parser.add_argument("--n", type=int, default=1000, help="Test set size n (default: 1000).")
    parser.add_argument("--m", type=int, default=500, help="Calibration set size m (default: 500).")
    parser.add_argument("--m-pilot", type=int, default=10, help="Pilot samples per side for adaptive allocation (default: 10).")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234).")
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
    thetas = np.linspace(args.theta_start, args.theta_stop, args.theta_num)
    df_equal, df_adaptive = run_simulation(
        thetas=thetas,
        replication=args.replication,
        q0=args.q0,
        q1=args.q1,
        n=args.n,
        m=args.m,
        m_pilot=args.m_pilot,
        alpha=args.alpha,
        seed=args.seed,
    )
    fig = build_figure(df_equal, df_adaptive)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, bbox_inches="tight", dpi=300)
        print(f"Saved figure to {args.output}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
