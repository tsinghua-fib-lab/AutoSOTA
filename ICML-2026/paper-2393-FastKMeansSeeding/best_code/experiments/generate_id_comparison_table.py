#!/usr/bin/env python3
"""
Generate LaTeX table comparing intrinsic dimension estimates:
- MLE estimate (from compute_intrinsic_dim.py)
- 2/ε estimate (from beta scaling law)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Results directories
RESULTS_DIR = Path(__file__).parent / "results" / "scaling_laws"
ID_RESULTS_PATH = Path(__file__).parent / "results" / "intrinsic_dim" / "mle_intrinsic_dim.csv"

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


def load_scaling_results(dataset: str) -> pd.DataFrame:
    """Load individual dataset scaling results CSV."""
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


def load_mle_intrinsic_dim():
    """Load MLE intrinsic dimension estimates."""
    if not ID_RESULTS_PATH.exists():
        return {}

    df = pd.read_csv(ID_RESULTS_PATH)

    # Get mean ID for k=20 (a reasonable middle value)
    # Or average across k values
    mle_estimates = {}

    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        # Use k=20 as reference, or average if not available
        k20 = subset[subset['k'] == 20]
        if len(k20) > 0:
            mle_estimates[dataset] = k20['id_mean'].values[0]
        else:
            # Use average across all k
            mle_estimates[dataset] = subset['id_mean'].mean()

    return mle_estimates


def generate_latex_table():
    """Generate LaTeX table comparing ID estimates."""

    mle_estimates = load_mle_intrinsic_dim()

    rows = []

    for dataset in DATASET_ORDER:
        df = load_scaling_results(dataset)
        if df is None:
            continue

        k_values = df["k"].values
        beta_mean = df["beta_mean"].values

        # Compute beta scaling (ε)
        eps_beta, _, r2_beta = compute_scaling(k_values, beta_mean)

        # Compute 2/ε
        d_from_eps = 2 / eps_beta if eps_beta > 0 else float('inf')

        # Get MLE estimate
        mle_id = mle_estimates.get(dataset, None)

        display_name = DATASET_NAMES.get(dataset, dataset)
        rows.append({
            'dataset': display_name,
            'eps': eps_beta,
            'd_eps': d_from_eps,
            'mle_id': mle_id,
        })

    # Generate LaTeX table
    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparison of intrinsic dimension estimates. $\varepsilon$ is the scaling exponent from $\beta \sim k^{\varepsilon}$. $d_{\varepsilon} = 2/\varepsilon$ is the intrinsic dimension estimate from the scaling law. $d_{\text{MLE}}$ is the maximum likelihood estimate of intrinsic dimension (using $k=20$ neighbors).}
\label{tab:intrinsic_dim_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Dataset} & $\boldsymbol{\varepsilon}$ & $\boldsymbol{d_{\varepsilon} = 2/\varepsilon}$ & $\boldsymbol{d_{\text{MLE}}}$ \\
\midrule
"""

    for row in rows:
        mle_str = f"{row['mle_id']:.2f}" if row['mle_id'] is not None else "---"
        latex += f"{row['dataset']} & {row['eps']:.4f} & {row['d_eps']:.2f} & {mle_str} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def main():
    print("=" * 60)
    print("Generating LaTeX Table: ID Comparison (MLE vs 2/ε)")
    print("=" * 60)

    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        return

    latex_table = generate_latex_table()

    # Print to console
    print("\n" + latex_table)

    # Save to file
    output_path = RESULTS_DIR / "intrinsic_dim_comparison_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
