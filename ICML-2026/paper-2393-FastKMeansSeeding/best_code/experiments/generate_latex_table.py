#!/usr/bin/env python3
"""
Generate LaTeX table with scaling law results for each dataset.

Columns: Dataset, ε (from beta), R²_β, R²_η
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Results directory
RESULTS_DIR = Path(__file__).parent / "results" / "scaling_laws"

# Dataset display names
DATASET_NAMES = {
    "mnist": "MNIST",
    "fmnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "mnist_clip": "MNIST-CLIP",
    "fmnist_clip": "FMNIST-CLIP",
    "cifar10_clip": "CIFAR-10-CLIP",
    "cifar100_clip": "CIFAR-100-CLIP",
    "reddit": "Reddit",
    "har": "HAR",
    "susy": "SUSY",
    "stackexchange": "StackExchange",
}

# Dataset order
DATASET_ORDER = [
    "mnist", "fmnist", "cifar10",
    "cifar100", "mnist_clip", "fmnist_clip",
    "cifar10_clip", "cifar100_clip", "reddit",
    "har", "susy", "stackexchange"
]


def load_dataset_results(dataset: str) -> pd.DataFrame:
    """Load individual dataset results CSV."""
    results_path = RESULTS_DIR / f"{dataset}_results.csv"
    if not results_path.exists():
        return None
    return pd.read_csv(results_path)


def compute_scaling(k_values: np.ndarray, values: np.ndarray):
    """Compute log-log linear regression for scaling law."""
    log_k = np.log(k_values)
    log_val = np.log(values)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_val)
    r_squared = r_value ** 2

    return slope, intercept, r_squared


def generate_latex_table():
    """Generate LaTeX table with scaling results."""

    rows = []

    for dataset in DATASET_ORDER:
        df = load_dataset_results(dataset)
        if df is None:
            continue

        k_values = df["k"].values
        beta_mean = df["beta_mean"].values
        eta_mean = df["eta_mean"].values

        # Compute beta scaling (ε)
        eps_beta, _, r2_beta = compute_scaling(k_values, beta_mean)

        # Compute eta scaling (ε/2)
        eps_eta_half, _, r2_eta = compute_scaling(k_values, eta_mean)

        display_name = DATASET_NAMES.get(dataset, dataset)
        rows.append({
            'dataset': display_name,
            'eps': eps_beta,
            'r2_beta': r2_beta,
            'r2_eta': r2_eta,
        })

    # Generate LaTeX table
    latex = r"""\begin{table}[htbp]
\centering
\caption{Scaling law parameters across datasets. $\varepsilon$ is estimated from $\beta \sim k^{\varepsilon}$. $R^2_\beta$ and $R^2_\eta$ are the coefficients of determination for the $\beta$ and $\eta$ scaling fits respectively.}
\label{tab:scaling_results}
\begin{tabular}{lccc}
\toprule
\textbf{Dataset} & $\boldsymbol{\varepsilon}$ & $\boldsymbol{R^2_\beta}$ & $\boldsymbol{R^2_\eta}$ \\
\midrule
"""

    for row in rows:
        latex += f"{row['dataset']} & {row['eps']:.4f} & {row['r2_beta']:.4f} & {row['r2_eta']:.4f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def main():
    print("=" * 60)
    print("Generating LaTeX Table for Scaling Law Results")
    print("=" * 60)

    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        return

    latex_table = generate_latex_table()

    # Print to console
    print("\n" + latex_table)

    # Save to file
    output_path = RESULTS_DIR / "scaling_results_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
