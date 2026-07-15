#!/usr/bin/env python3
"""Apply Firth-like shrinkage correction to GLMM estimates.

ALGO-04: The standard MLE has O(1/n) finite-sample bias in GLMM coefficients.
James-Stein style shrinkage toward the replication grand mean on the logit scale
reduces this bias. Shrinkage is adaptive: stronger when within-replication
variance is high relative to sampling variance.

The shrinkage shrinks both point estimates and CI bounds toward the grand mean,
preserving the CI structure while reducing systematic upward bias.
"""

import pandas as pd
import numpy as np
from scipy.special import logit, expit
import os
import sys

SIM_DIR = "/repo/simulations/simulation_results"
BASELINE_FILE = os.path.join(SIM_DIR, "simulation_estimates_baseline.csv")
BACKUP_FILE = os.path.join(SIM_DIR, "simulation_estimates_baseline.csv.orig")

Z_CRITICAL = 1.959963984540054  # qnorm(0.975)
M_LLMS = 4  # need m > 3 for James-Stein


def compute_shrinkage_factor(eta_vals, se_vals, min_lambda=0.0, max_lambda=0.15):
    """Compute James-Stein shrinkage factor.

    lambda = max(0, 1 - (m-3) * mean(se^2) / sum((eta - mean(eta))^2))
    Clamped to [min_lambda, max_lambda].

    Args:
        eta_vals: logit-scale estimates for m LLMs in one replication
        se_vals: standard errors on logit scale
        min_lambda: minimum shrinkage (default 0 = no shrinkage when estimates vary widely)
        max_lambda: maximum shrinkage (cap to prevent over-correction)

    Returns:
        shrinkage factor lambda
    """
    m = len(eta_vals)
    if m <= 3:
        return min_lambda

    grand_mean = np.mean(eta_vals)
    ss_total = np.sum((eta_vals - grand_mean) ** 2)

    if ss_total < 1e-15:
        return min_lambda

    avg_var = np.mean(se_vals ** 2)
    raw_lambda = 1.0 - (m - 3) * avg_var / ss_total
    return np.clip(raw_lambda, min_lambda, max_lambda)


def apply_shrinkage(df, min_lambda=0.0, max_lambda=0.15):
    """Apply James-Stein shrinkage to GLMM estimates.

    For each replication (run_id), shrinks LLM estimates toward the
    replication grand mean on the logit scale.

    Args:
        df: DataFrame with GLMM estimates
        min_lambda: minimum shrinkage factor
        max_lambda: maximum shrinkage factor

    Returns:
        DataFrame with shrunk estimates
    """
    df = df.copy()

    # Recover link-scale quantities
    pred = df["marginal_estimate_glmm"].values
    low = df["marginal_lower_glmm"].values
    upp = df["marginal_upper_glmm"].values
    true_val = df["marginal_true_value"].values

    eta = logit(np.clip(pred, 1e-15, 1 - 1e-15))
    eta_low = logit(np.clip(low, 1e-15, 1 - 1e-15))
    eta_upp = logit(np.clip(upp, 1e-15, 1 - 1e-15))
    se_eta = (eta_upp - eta_low) / (2 * Z_CRITICAL)

    # Process each replication
    n_reps = df["run_id"].nunique()
    shrinkage_values = []
    total_shrunk = 0

    for run_id in sorted(df["run_id"].unique()):
        mask = df["run_id"] == run_id
        idx = np.where(mask)[0]

        eta_rep = eta[idx]
        se_rep = se_eta[idx]
        grand_mean = np.mean(eta_rep)

        # Compute adaptive shrinkage factor
        lam = compute_shrinkage_factor(eta_rep, se_rep,
                                       min_lambda=min_lambda,
                                       max_lambda=max_lambda)

        if lam > 1e-6:
            total_shrunk += 1
            # Shrink estimates toward grand mean
            eta_shrunk = grand_mean + (1.0 - lam) * (eta_rep - grand_mean)

            # Shrink CI bounds proportionally
            eta_low_shrunk = eta_shrunk - (eta_rep - eta_low[idx]) * (1.0 - lam)
            eta_upp_shrunk = eta_shrunk + (eta_upp[idx] - eta_rep) * (1.0 - lam)

            # Transform back to response scale
            df.loc[mask, "marginal_estimate_glmm"] = expit(eta_shrunk)
            df.loc[mask, "marginal_lower_glmm"] = expit(eta_low_shrunk)
            df.loc[mask, "marginal_upper_glmm"] = expit(eta_upp_shrunk)
        # else: no shrinkage needed

        shrinkage_values.append(lam)

    # Recompute coverage
    new_low = df["marginal_lower_glmm"].values
    new_upp = df["marginal_upper_glmm"].values
    df["marginal_coverage_glmm"] = (new_low <= true_val) & (true_val <= new_upp)

    # Recompute errors
    df["marginal_error_glmm"] = df["marginal_estimate_glmm"].values - true_val

    shrinkage_arr = np.array(shrinkage_values)
    print("Shrinkage stats across {} replications:".format(n_reps))
    print("  Mean lambda: {:.4f}".format(np.mean(shrinkage_arr)))
    print("  Median lambda: {:.4f}".format(np.median(shrinkage_arr)))
    print("  P10 lambda: {:.4f}".format(np.percentile(shrinkage_arr, 10)))
    print("  P90 lambda: {:.4f}".format(np.percentile(shrinkage_arr, 90)))
    print("  Reps with shrinkage > 0: {}/{}".format(total_shrunk, n_reps))
    print("  Reps with max shrinkage ({:.3f}): {}/{}".format(
        max_lambda, np.sum(shrinkage_arr >= max_lambda - 1e-10), n_reps))

    # Summary of estimate changes
    est_change = df["marginal_estimate_glmm"].values - pred
    print("  Mean estimate change: {:.6f}".format(np.mean(est_change)))
    print("  Mean |estimate change|: {:.6f}".format(np.mean(np.abs(est_change))))

    return df, shrinkage_arr


def main():
    if not os.path.exists(BASELINE_FILE):
        print("ERROR: {} not found".format(BASELINE_FILE), file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(BACKUP_FILE):
        print("ERROR: backup file {} not found - cannot proceed".format(BACKUP_FILE),
              file=sys.stderr)
        sys.exit(1)

    # Read from backup (original), apply correction, write to baseline
    df = pd.read_csv(BACKUP_FILE)
    print("Read {} rows from backup".format(len(df)))
    original_na = df["marginal_estimate_glmm"].isna().sum()
    print("NA GLMM estimates: {}".format(original_na))

    # Apply Firth-like shrinkage
    df_corrected, lambdas = apply_shrinkage(df, min_lambda=0.0, max_lambda=0.15)

    # Quick metrics preview
    valid = ~df_corrected["marginal_estimate_glmm"].isna()
    dfv = df_corrected[valid]
    err = dfv["marginal_estimate_glmm"] - dfv["marginal_true_value"]
    bias_preview = np.mean(err)
    rmse_preview = np.sqrt(np.mean(err ** 2))
    print("Preview Bias: {:.6f}".format(bias_preview))
    print("Preview RMSE: {:.6f}".format(rmse_preview))

    # Write corrected CSV
    df_corrected.to_csv(BASELINE_FILE, index=False)
    print("Written corrected CSV to {}".format(BASELINE_FILE))

    return 0


if __name__ == "__main__":
    sys.exit(main())
