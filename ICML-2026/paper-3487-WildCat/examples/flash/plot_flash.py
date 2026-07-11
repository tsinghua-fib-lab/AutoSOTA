#!/usr/bin/env python3

"""Plot WildCat speed-up over FlashAttention and error on dual y-axes.

By default, this script reads `results/results.txt`, filters
`wildcat` rows for r=64 and B=16, and overlays:
1) speed-up = flash_median_ms / wildcat_median_ms (left y-axis)
2) max_abs_error (right y-axis)
as a function of sequence length.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot speed-up over flash and max_abs_error vs sequence length "
            "from results/results.txt."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/results.txt"),
        help="Path to results.txt (default: results/results.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plot_flash.pdf"),
        help="Output image path",
    )
    parser.add_argument("--r", type=int, default=64, help="WildCat r to plot")
    parser.add_argument(
        "--B", type=int, default=16, help="WildCat B to plot"
    )
    parser.add_argument(
        "--min-seq",
        type=int,
        default=8192,
        help="Minimum sequence length to include",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=262144,
        help="Maximum sequence length to include",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window in addition to saving the plot",
    )
    return parser.parse_args()


def load_rows(results_path: Path) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    with results_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("seq") or line.startswith("---"):
                continue

            parts = line.split()
            if len(parts) != 8:
                continue

            rows.append(
                {
                    "seq_len": int(parts[0]),
                    "method": parts[1],
                    "r": None if parts[2] == "-" else int(parts[2]),
                    "num_bins": None if parts[3] == "-" else int(parts[3]),
                    "median": float(parts[5]),
                    "max_abs_error": None if parts[7] == "-" else float(parts[7]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()

    base_font_size = 16
    label_font_size = 18
    tick_font_size = 15
    legend_font_size = 15
    plt.rcParams.update({"font.size": base_font_size})

    if not args.results.exists():
        raise FileNotFoundError(f"Could not find results file: {args.results}")

    rows = load_rows(args.results)

    flash_by_seq = {
        int(row["seq_len"]): float(row["median"])
        for row in rows
        if row["method"] == "flash"
    }

    wildcat_rows = [
        row
        for row in rows
        if row["method"] == "wildcat"
        and row["r"] == args.r
        and row["num_bins"] == args.B
        and args.min_seq <= int(row["seq_len"]) <= args.max_seq
    ]

    if not wildcat_rows:
        raise ValueError(
            "No matching wildcat rows found for "
            f"r={args.r}, num_bins={args.B}, "
            f"seq in [{args.min_seq}, {args.max_seq}]"
        )

    wildcat_rows.sort(key=lambda row: int(row["seq_len"]))

    seq_lens: list[int] = []
    speedups: list[float] = []
    errors: list[float] = []

    for row in wildcat_rows:
        seq_len = int(row["seq_len"])
        if seq_len not in flash_by_seq:
            continue

        wildcat_median_ms = float(row["median"])
        flash_median_ms = flash_by_seq[seq_len]
        error = row["max_abs_error"]
        if error is None:
            continue

        seq_lens.append(seq_len)
        speedups.append(flash_median_ms / wildcat_median_ms)
        errors.append(float(error))

    if not seq_lens:
        raise ValueError(
            "No overlapping sequence lengths between flash and selected wildcat rows."
        )

    fig, ax_left = plt.subplots(figsize=(8.4, 5.2))
    ax_right = ax_left.twinx()

    left_color = "#1f77b4"
    right_color = "#d62728"

    line_speedup = ax_left.plot(
        seq_lens,
        speedups,
        marker="o",
        linewidth=2,
        color=left_color,
        label="Speed-up over FlashAttention 2",
    )
    line_error = ax_right.plot(
        seq_lens,
        errors,
        marker="s",
        linewidth=2,
        linestyle="--",
        color=right_color,
        label=r"Approximation error $\| \mathbf{O} - \mathbf{\hat{O}}\|_{\text{max}}$",
    )

    ax_left.set_xscale("log", base=2)
    ax_left.set_xticks(seq_lens)
    ax_left.set_xticklabels([f"{n // 1024}k" for n in seq_lens], fontsize=tick_font_size)
    ax_left.set_xlabel(r"Sequence length $n$", fontsize=label_font_size)
    ax_left.set_ylabel("Speed-up", color=left_color, fontsize=label_font_size)
    ax_right.set_ylabel("Error", color=right_color, fontsize=label_font_size)
    ax_left.tick_params(axis="x", labelsize=tick_font_size)
    ax_left.tick_params(axis="y", labelcolor=left_color, labelsize=tick_font_size)
    ax_right.tick_params(axis="y", labelcolor=right_color, labelsize=tick_font_size)
    ax_left.grid(True, alpha=0.3)

    # Add a combined legend for both axes.
    lines = line_speedup + line_error
    labels = [line.get_label() for line in lines]
    ax_left.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=legend_font_size,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0), pad=0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()