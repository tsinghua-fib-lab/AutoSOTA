"""Compute odd Fourier energy bucketed into degree-1, degree-3, and degree-5+.

For each game instance, evaluates the game on all 2^n coalitions, applies the
Fast Walsh-Hadamard Transform to recover Fourier coefficients, then buckets
squared odd-degree coefficients into: degree 1, degree 3, degree 5+.
Produces one bar chart per game type showing mean energy fraction per bucket
across all instances, with percentage labels.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapiq_benchmark.load import GameFactory


# ---------------------------------------------------------------------------
# Game configs
# ---------------------------------------------------------------------------
GAMES = {
    "SentimentIMDBDistilBERT14": {
        "config": "shapiq-benchmark/configurations_exhaustive/SentimentAnalysisLocalXAI.json",
        "config_id": 1,
        "n_players": 14,
        "label": "DistilBERT ($d=14$)",
        "n_games": 30,
    },
    "ViT4by4Patches": {
        "config": "shapiq-benchmark/configurations_exhaustive/ViT4by4Patches.json",
        "config_id": 1,
        "n_players": 16,
        "label": "ViT16 ($d=16$)",
        "n_games": 30,
    },
}

BUCKET_LABELS = ["Degree 1", "Degree 3", "Degree 5+"]
BUCKET_COLORS = ["#4477AA", "#EE6677", "#228833"]


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform
# ---------------------------------------------------------------------------

def fwht(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform (unnormalized)."""
    a = a.copy()
    N = len(a)
    h = 1
    while h < N:
        a = a.reshape(-1, 2 * h)
        top = a[:, :h] + a[:, h:]
        bot = a[:, :h] - a[:, h:]
        a = np.hstack([top, bot])
        h *= 2
    return a.ravel()


def odd_fourier_energy_bucketed(game_fn, n: int, batch_size: int = 2048) -> np.ndarray:
    """Return bucketed odd-degree energy array of shape (3,).

    Buckets: [degree-1 energy, degree-3 energy, degree-5+ energy].
    Even degrees are ignored entirely. Fractions are computed relative to
    total odd energy only.

    Args:
        game_fn   : callable (coalitions: np.ndarray shape (m, n)) -> np.ndarray (m,)
        n         : number of players
        batch_size: number of coalitions to query at once

    Returns:
        buckets : np.ndarray shape (3,) — raw energies for [1, 3, 5+]
    """
    N = 2 ** n

    indices = np.arange(N, dtype=np.int32)
    all_coalitions = ((indices[:, None] >> np.arange(n, dtype=np.int32)[None, :]) & 1).astype(float)

    values = np.empty(N)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        values[start:end] = game_fn(all_coalitions[start:end])

    coeffs = fwht(values) / N
    sq = coeffs ** 2

    degrees = np.array([bin(i).count("1") for i in range(N)], dtype=np.int32)

    e1   = sq[degrees == 1].sum()
    e3   = sq[degrees == 3].sum()
    e5p  = sq[degrees >= 5].sum()  # only odd degrees ≥ 5
    # restrict 5+ to odd degrees
    e5p  = sq[(degrees >= 5) & (degrees % 2 == 1)].sum()

    return np.array([e1, e3, e5p])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_odd_energy_buckets(bucket_matrix: np.ndarray, title: str, save_path: Path):
    """Box-and-whisker plot of odd Fourier energy fractions for 3 buckets.

    Args:
        bucket_matrix : shape (n_instances, 3) — raw energies [deg1, deg3, deg5+]
        title         : plot title
        save_path     : where to save the PDF
    """
    # Convert each instance's raw energies to fractions of its own odd total
    odd_totals = bucket_matrix.sum(axis=1, keepdims=True)
    frac_matrix = bucket_matrix / odd_totals  # (n_instances, 3)

    fig, ax = plt.subplots(figsize=(6, 5))

    bp = ax.boxplot(
        [frac_matrix[:, i] * 100 for i in range(3)],
        labels=BUCKET_LABELS,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, linestyle="none", alpha=0.6),
    )

    for patch, color in zip(bp["boxes"], BUCKET_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("% of odd Fourier energy", fontsize=12)
    ax.set_title(title, fontsize=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for game_name, cfg in GAMES.items():
        n = cfg["n_players"]
        print(f"\n{'='*60}")
        print(f"Game: {game_name}  (n={n}, 2^n={2**n} evaluations per instance)")
        print(f"{'='*60}")

        game_generator = GameFactory.load_configuration_file(
            config_path=cfg["config"],
            config_id=cfg["config_id"],
            n_games=cfg["n_games"],
            check_pre_computed=True,
            only_pre_computed=True,
        )

        bucket_matrix = []
        for idx, game_instance in enumerate(game_generator):
            print(f"  Instance {idx+1}/{cfg['n_games']}  ", end="", flush=True)
            buckets = odd_fourier_energy_bucketed(game_instance, n)
            bucket_matrix.append(buckets)
            odd_total = buckets.sum()
            print(
                f"deg1={100*buckets[0]/odd_total:.1f}%  "
                f"deg3={100*buckets[1]/odd_total:.1f}%  "
                f"deg5+={100*buckets[2]/odd_total:.1f}%"
            )

        bucket_matrix = np.array(bucket_matrix)  # (n_games, 3)

        out_dir = Path("plots/fourier_energy_odd")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{game_name}_odd_buckets.npy", bucket_matrix)

        plot_odd_energy_buckets(
            bucket_matrix,
            title=cfg["label"],
            save_path=out_dir / f"{game_name}_odd_fourier_energy.pdf",
        )

        mean_b = bucket_matrix.mean(axis=0)
        odd_total = mean_b.sum()
        print(f"\n  Summary for {game_name} (odd degrees only):")
        print(f"  {'Bucket':<12} {'Mean energy':>14} {'% of odd total':>16}")
        print(f"  {'-'*44}")
        for label, val in zip(BUCKET_LABELS, mean_b):
            print(f"  {label:<12} {val:>14.6f} {100*val/odd_total:>15.1f}%")
