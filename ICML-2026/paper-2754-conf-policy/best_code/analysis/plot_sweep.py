"""Plot CPC-LLM sweep results: risk, avg score, and cumulative max score.

Reads a JSON file produced by ``extract_sweep_data.py`` and generates a
3-panel figure comparing CPC-constrained policies at different alpha levels
against the uncontrolled baseline.

Usage:
    # Extract data first (requires Modal)
    modal run analysis/extract_sweep_data.py

    # Generate plots (local, no Modal needed)
    python analysis/plot_sweep.py sweep_data.json
    python analysis/plot_sweep.py sweep_data.json --output figures/sweep.png
"""

from __future__ import annotations

import argparse
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PALETTE = {
    1.0: "#DC143C",
    0.8: "#8B008B",
    0.6: "#0000CD",
    0.4: "#00BCD4",
}
LABELS = {
    1.0: "Uncontrolled",
    0.8: r"CPC, $\alpha$=0.8",
    0.6: r"CPC, $\alpha$=0.6",
    0.4: r"CPC, $\alpha$=0.4",
}
MARKERS = {1.0: "X", 0.8: "o", 0.6: "o", 0.4: "o"}


def _val(row: dict | None, metric: str) -> float:
    if row is None:
        return np.nan
    v = row.get(metric)
    if v is None:
        return np.nan
    return float(v)


def _mean_se(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(arr, axis=0)
    n = np.sum(~np.isnan(arr), axis=0)
    se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
    return mean, se


def build_lookup(
    data: list[dict], alphas: list[float], seeds: list[int]
) -> dict[tuple, dict]:
    """Build (alpha, seed, round_idx) -> row lookup with round-0 fill."""
    lookup: dict[tuple, dict] = {}
    for row in data:
        key = (row["alpha"], row["seed"], row["round_idx"])
        lookup[key] = row

    # Round 0 is alpha-independent (same init SFT). Fill missing alphas.
    for s in seeds:
        r0 = None
        for a in alphas:
            if (a, s, 0) in lookup:
                r0 = lookup[(a, s, 0)]
                break
        if r0:
            for a in alphas:
                if (a, s, 0) not in lookup:
                    lookup[(a, s, 0)] = dict(r0)
    return lookup


def get_series(
    lookup: dict, alpha: float, seeds: list[int], max_round: int, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Gather per-seed values for a metric across rounds.

    Args:
        lookup: (alpha, seed, round_idx) -> row dict.
        alpha: Alpha value to query.
        seeds: Seed values to include.
        max_round: Maximum round index.
        metric: Key to extract from each row.

    Returns:
        Tuple of (rounds array, per-seed values array of shape (n_seeds, n_rounds)).
    """
    rounds = np.arange(0, max_round + 1)
    per_seed = []
    for s in seeds:
        per_seed.append([_val(lookup.get((alpha, s, r)), metric) for r in rounds])
    return rounds, np.array(per_seed)


def get_cummax_series(
    lookup: dict, alpha: float, seeds: list[int], max_round: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-seed cumulative max of max_score across rounds.

    Args:
        lookup: (alpha, seed, round_idx) -> row dict.
        alpha: Alpha value to query.
        seeds: Seed values to include.
        max_round: Maximum round index.

    Returns:
        Tuple of (rounds array, per-seed cummax array of shape (n_seeds, n_rounds)).
    """
    rounds = np.arange(0, max_round + 1)
    per_seed = []
    for s in seeds:
        cummax: list[float] = []
        running = -np.inf
        for r in rounds:
            v = _val(lookup.get((alpha, s, r)), "max_score")
            if not np.isnan(v):
                running = max(running, v)
            cummax.append(running if running > -np.inf else np.nan)
        per_seed.append(cummax)
    return rounds, np.array(per_seed)


def plot_sweep(data: list[dict], output: str = "sweep_results.png") -> None:
    """Generate 3-panel sweep figure.

    Args:
        data: List of per-round metric dicts from extract_sweep_data.
        output: Path to save the figure.
    """
    sns.set_theme(style="white", font_scale=1.1)

    alphas = sorted({r["alpha"] for r in data})
    seeds = sorted({r["seed"] for r in data})
    max_round = max(r["round_idx"] for r in data)
    lookup = build_lookup(data, alphas, seeds)

    # Plot CPC lines first (background), uncontrolled last (foreground)
    plot_order = sorted(alphas)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Left: Fraction Infeasible (Risk)
    ax = axes[0]
    for alpha in plot_order:
        rounds, arr = get_series(lookup, alpha, seeds, max_round, "frac_infeasible")
        mean, se = _mean_se(arr)
        c = PALETTE.get(alpha, "gray")
        ax.plot(
            rounds,
            mean,
            marker=MARKERS.get(alpha, "o"),
            color=c,
            label=LABELS.get(alpha, f"$\\alpha$={alpha}"),
            linewidth=1.8,
            markersize=6,
        )
        ax.fill_between(rounds, mean - se, mean + se, color=c, alpha=0.15)
    for alpha in sorted(a for a in alphas if a < 1.0):
        ax.axhline(
            y=alpha,
            color=PALETTE.get(alpha, "gray"),
            linestyle="--",
            alpha=0.5,
            label=rf"Target $\alpha$={alpha}",
        )
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3, label="B=1.0")
    ax.set_xlabel("Timestep of Policy Improvement")
    ax.set_ylabel(r"Fraction Infeasible (Risk) [$\leftarrow$]")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.3, max_round + 0.3)

    # Middle: Avg Score per Round
    ax = axes[1]
    for alpha in plot_order:
        rounds, arr = get_series(lookup, alpha, seeds, max_round, "mean_score")
        mean, se = _mean_se(arr)
        c = PALETTE.get(alpha, "gray")
        ax.plot(
            rounds,
            mean,
            marker=MARKERS.get(alpha, "o"),
            color=c,
            label=LABELS.get(alpha, f"$\\alpha$={alpha}"),
            linewidth=1.8,
            markersize=6,
        )
        ax.fill_between(rounds, mean - se, mean + se, color=c, alpha=0.15)
    ax.set_xlabel("Timestep of Policy Improvement")
    ax.set_ylabel(r"Avg. Score per Round [$\rightarrow$]")
    ax.set_xlim(-0.3, max_round + 0.3)

    # Right: Max Score Over Rounds (cumulative)
    ax = axes[2]
    for alpha in plot_order:
        rounds, arr = get_cummax_series(lookup, alpha, seeds, max_round)
        mean, se = _mean_se(arr)
        c = PALETTE.get(alpha, "gray")
        ax.plot(
            rounds,
            mean,
            marker=MARKERS.get(alpha, "o"),
            color=c,
            label=LABELS.get(alpha, f"$\\alpha$={alpha}"),
            linewidth=1.8,
            markersize=6,
        )
        ax.fill_between(rounds, mean - se, mean + se, color=c, alpha=0.15)
    ax.set_xlabel("Timestep of Policy Improvement")
    ax.set_ylabel(r"Max Score Over Rounds [$\rightarrow$]")
    ax.set_xlim(-0.3, max_round + 0.3)

    # Shared legend
    handles, leg_labels = [], []
    for ax in axes:
        h, labels = ax.get_legend_handles_labels()
        for hi, li in zip(h, labels):
            if li not in leg_labels:
                handles.append(hi)
                leg_labels.append(li)
    fig.legend(
        handles,
        leg_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.08),
        fontsize=9,
        frameon=True,
    )
    plt.tight_layout()
    sns.despine()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Path to sweep_data.json")
    parser.add_argument(
        "--output",
        "-o",
        default="sweep_results.png",
        help="Output image path (default: sweep_results.png)",
    )
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    plot_sweep(data, output=args.output)


if __name__ == "__main__":
    main()
