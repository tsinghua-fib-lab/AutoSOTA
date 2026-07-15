#!/usr/bin/env python3
"""Evaluation script for Paper 5668 - Expanding the AI Evaluation Toolbox with Statistical Models.

Reproduces metrics from Table C.6 (Setting A, Baseline) by computing summary statistics
from the pre-computed simulation results CSV.

Usage: python3 eval_metrics.py [--all]
  --all: Compute metrics for all conditions (Table C.6), not just baseline.
"""

import pandas as pd
import numpy as np
import sys
import os
import json

SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "simulations", "simulation_results")

# Fall back to /repo if running from installed location
if not os.path.isdir(SIM_DIR):
    SIM_DIR = "/repo/simulations/simulation_results"


def compute_metrics(df):
    """Compute all Table C.6 metrics from simulation results dataframe."""
    # Remove rows with NA GLMM estimates (matching R code)
    df = df[~df["marginal_estimate_glmm"].isna()].copy()

    # Interval widths
    iw_rf = df["marginal_upper_rf"] - df["marginal_lower_rf"]
    iw_glmm = df["marginal_upper_glmm"] - df["marginal_lower_glmm"]

    # Coverage: boolean -> numeric
    cov_rf = df["marginal_coverage_rf"].astype(bool).astype(float)
    cov_glmm = df["marginal_coverage_glmm"].astype(bool).astype(float)

    # Errors
    err_rf = df["marginal_estimate_rf"] - df["marginal_true_value"]
    err_glmm = df["marginal_estimate_glmm"] - df["marginal_true_value"]

    return {
        "RF_Bias": float(np.mean(err_rf)),
        "GLMM_Bias": float(np.mean(err_glmm)),
        "RF_RMSE": float(np.sqrt(np.mean(err_rf ** 2))),
        "GLMM_RMSE": float(np.sqrt(np.mean(err_glmm ** 2))),
        "RF_Coverage": float(np.mean(cov_rf)),
        "GLMM_Coverage": float(np.mean(cov_glmm)),
        "RF_CI_Width": float(np.mean(iw_rf)),
        "GLMM_CI_Width": float(np.mean(iw_glmm)),
        "RF_CI_Width_SD": float(np.std(iw_rf, ddof=1)),
        "GLMM_CI_Width_SD": float(np.std(iw_glmm, ddof=1)),
    }


def main():
    baseline_file = os.path.join(SIM_DIR, "simulation_estimates_baseline.csv")

    if not os.path.exists(baseline_file):
        print(f"ERROR: Baseline simulation file not found: {baseline_file}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(baseline_file)
    metrics = compute_metrics(df)

    print("=" * 60)
    print("Paper 5668 - Table C.6 Setting A Baseline Metrics")
    print("=" * 60)
    print(f"  N observations: {len(df)}")
    print()
    print(f"{'Metric':<22} {'RF':>12} {'GLMM':>12}  {'Paper(G)':>10}")
    print(f"{'-'*60}")

    paper_vals = {
        "Bias": (0.00109, 0.00170),
        "RMSE": (0.0408, 0.0406),
        "Coverage": (0.941, 0.943),
        "CI_Width": (0.163, 0.154),
        "CI_Width_SD": (0.0290, 0.0238),
    }

    for key, (rf_paper, glmm_paper) in paper_vals.items():
        rf_val = metrics[f"RF_{key}"]
        glmm_val = metrics[f"GLMM_{key}"]
        print(f"  {key:<20} {rf_val:12.6f} {glmm_val:12.6f}  {glmm_paper:10.5f}")

    print()
    print("--- JSON OUTPUT ---")
    print(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    main()
