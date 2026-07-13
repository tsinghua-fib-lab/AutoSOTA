import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# Horizons
T = np.array([5000, 10000, 15000, 20000, 30000, 40000])

# Regret data
regret = {
    "ATC (time-varying)": [35.657, 36.204, 37.369, 37.124, 36.869, 38.350],
    r"$\gamma=6\sigma$":  [36.159, 36.824, 37.682, 37.776, 37.561, 39.005],
    r"$\gamma=4.81\sigma$":[32.404, 33.675, 34.436, 34.850, 37.124, 39.013],
    r"$\gamma=7.78\sigma$":[36.202, 36.884, 38.013, 37.671, 37.411, 38.734],
}

# FA for gamma = 4.81 sigma
FA_481 = np.array([0.05, 0.09, 0.14, 0.22, 0.33, 0.45])

def parse_args():
    p = argparse.ArgumentParser(description="Plot average regret vs horizon.")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for saving the figure.",
    )
    p.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Optional full output path for the figure.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Figure: regret
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))

    # Plot regret curves in requested legend order
    legend_order = [
        r"$\gamma=7.78\sigma$",
        r"$\gamma=6\sigma$",
        r"$\gamma=4.81\sigma$",
        "ATC (time-varying)",
    ]
    color_map = {
        r"$\gamma=7.78\sigma$": "tab:purple",
        r"$\gamma=6\sigma$": "tab:orange",
        r"$\gamma=4.81\sigma$": "tab:green",
        "ATC (time-varying)": "tab:blue",
    }
    for label in legend_order:
        values = regret[label]
        ax1.plot(
            T,
            values,
            marker="o",
            linewidth=1.8,
            label=label,
            color=color_map[label],
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Horizon $T$", fontsize=13)
    ax1.set_ylabel("Average regret", fontsize=13)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(
        lines1,
        labels1,
        fontsize=13,
        frameon=True,
        loc="lower right"
    )
    plt.tight_layout()

    if args.out_path:
        out_path = args.out_path
    elif args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "regret_vs_horizon.png")
    else:
        out_path = None

    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"Wrote: {out_path}")
    else:
        plt.show()

    # Figure: FA vs T
    fig_fa, ax_fa = plt.subplots(figsize=(7.5, 4.8))
    fa_series = {
        r"$\gamma=4.81\sigma$": FA_481,
        "ATC (time-varying)": np.full_like(T, 0.02, dtype=float),
    }
    for label, fa_vals in fa_series.items():
        ax_fa.plot(
            T,
            fa_vals,
            marker="o",
            linewidth=1.8,
            label=label,
            color=color_map[label],
        )

    ax_fa.set_xscale("log")
    ax_fa.set_xlabel("Horizon $T$", fontsize=13)
    ax_fa.set_ylabel("False alarm rate", fontsize=13)
    ax_fa.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax_fa.legend(fontsize=13, frameon=True, loc="upper left")
    plt.tight_layout()

    if out_path:
        base, ext = os.path.splitext(out_path)
        fa_path = f"{base}_fa{ext or '.png'}"
        plt.savefig(fa_path, dpi=200)
        print(f"Wrote: {fa_path}")
    elif args.out_dir:
        fa_path = os.path.join(args.out_dir, "fa_vs_horizon.png")
        plt.savefig(fa_path, dpi=200)
        print(f"Wrote: {fa_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
