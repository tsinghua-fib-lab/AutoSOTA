#!/usr/bin/env python3
"""Plot CDF/scatter for UTKFace bounded-vs-alpha comparison JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot UTKFace bounded-vs-alpha comparison.")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default="outputs/utkface_bounded_vs_alpha.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if not samples:
        raise RuntimeError("No samples found in input JSON.")

    bounded = np.asarray([s["ecg_radius_mean_over_trials"] for s in samples], dtype=float)
    alpha = np.asarray([s["alpha_result"]["radius_alpha"] for s in samples], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # CDF
    ax = axes[0]
    for arr, label in [(bounded, "(E,C,G)+M"), (alpha, "Alpha-trimming")]:
        x = np.sort(arr)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, linewidth=2, label=label)
    ax.set_xlabel("Certified radius")
    ax.set_ylabel("CDF")
    ax.set_title("UTKFace radius CDF")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Scatter
    ax = axes[1]
    ax.scatter(alpha, bounded, alpha=0.7, s=24)
    max_val = float(max(np.max(alpha), np.max(bounded)))
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5)
    ax.set_xlabel("Alpha-trimming radius")
    ax.set_ylabel("(E,C,G)+M radius")
    ax.set_title("Per-point radius comparison")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()

