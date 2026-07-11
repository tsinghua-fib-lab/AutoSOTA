#!/usr/bin/env python3
"""
Plot appendix-style convergence/coverage/radius results for UTKFace single-point analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def mean_sem(x: np.ndarray) -> tuple[float, float]:
    if len(x) == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Plot UTKFace appendix-style convergence analysis.")
    p.add_argument("--input", type=str, required=True)
    p.add_argument(
        "--output",
        type=str,
        default="outputs/utkface_appendix_convergence.png",
        help="Output figure path",
    )
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    part1 = data["part1"]
    part2 = data["part2"]

    N_values = [int(n) for n in part1["N_values"]]
    gt = part1["ground_truth"]
    C_true = float(gt["C"])
    theta_true = float(gt["theta"])
    G_true = float(gt["G_norm"])

    C_means, C_stds, C_cov = [], [], []
    theta_means, theta_stds, theta_cov = [], [], []
    G_cov = []

    for N in N_values:
        trials1 = part1["results_by_N"][str(N)]

        C_hat = np.array([t["C_hat"] for t in trials1], dtype=float)
        C_low = np.array([t["C_lower"] for t in trials1], dtype=float)
        C_up = np.array([t["C_upper"] for t in trials1], dtype=float)
        th_hat = np.array([t["theta_hat"] for t in trials1], dtype=float)
        th_low = np.array([t["theta_lower"] for t in trials1], dtype=float)
        th_up = np.array([t["theta_upper"] for t in trials1], dtype=float)
        G_low = np.array([t["G_norm_lower"] for t in trials1], dtype=float)
        G_up = np.array([t["G_norm_upper"] for t in trials1], dtype=float)

        C_means.append(float(np.mean(C_hat)))
        C_stds.append(float(np.std(C_hat, ddof=1)) if len(C_hat) > 1 else 0.0)
        C_cov.append(float(np.mean((C_low <= C_true) & (C_true <= C_up))))

        theta_means.append(float(np.mean(th_hat)))
        theta_stds.append(float(np.std(th_hat, ddof=1)) if len(th_hat) > 1 else 0.0)
        theta_cov.append(float(np.mean((th_low <= theta_true) & (theta_true <= th_up))))

        G_cov.append(float(np.mean((G_low <= G_true) & (G_true <= G_up))))

    n_trials = len(part1["results_by_N"][str(N_values[0])])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) C convergence
    ax = axes[0]
    C_band = [1.96 * s / np.sqrt(n_trials) for s in C_stds]
    ax.plot(N_values, C_means, "o-", linewidth=2, markersize=8, label=r"$\hat{C}$", color="#2E86AB")
    ax.fill_between(
        N_values,
        np.asarray(C_means) - np.asarray(C_band),
        np.asarray(C_means) + np.asarray(C_band),
        alpha=0.3,
        color="#2E86AB",
        label="95% CI band",
    )
    ax.axhline(C_true, linestyle="--", linewidth=2, label=r"$C_{\mathrm{ref}}$", color="#A23B72")
    ax.set_xscale("log")
    ax.set_title(r"Variance Convergence with CI Band ($\sigma=0.06$)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample Size N", fontweight="bold")
    ax.set_ylabel("Variance C (years$^2$)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (b) theta convergence
    ax = axes[1]
    theta_band = [1.96 * s / np.sqrt(n_trials) for s in theta_stds]
    ax.plot(N_values, theta_means, "s-", linewidth=2, markersize=8, label=r"$\hat{\theta}$", color="#F18F01")
    ax.fill_between(
        N_values,
        np.asarray(theta_means) - np.asarray(theta_band),
        np.asarray(theta_means) + np.asarray(theta_band),
        alpha=0.3,
        color="#F18F01",
        label="95% CI band",
    )
    ax.axhline(theta_true, linestyle="--", linewidth=2, label=r"$\theta_{\mathrm{ref}}$", color="#A23B72")
    ax.set_xscale("log")
    ax.set_title("Squared Gradient Norm Convergence with CI Band", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample Size N", fontweight="bold")
    ax.set_ylabel(r"$\theta=\|G\|_2^2$ (years$^2$/px$^2$)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

