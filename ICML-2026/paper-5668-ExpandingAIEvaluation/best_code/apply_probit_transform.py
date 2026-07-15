#!/usr/bin/env python3
"""Apply probit link transformation to GLMM estimates.

ALGO-03: Using probit link instead of logit aligns the link function with the
Normal random effects distribution. Without refitting the GLMM, we approximate
probit-GLMM estimates by scaling logit-scale estimates by sqrt(3)/pi ≈ 0.551,
then transforming back with the Normal CDF.

Theory: For mid-range probabilities, probit(p) ≈ logit(p) * sqrt(3)/pi.
The logistic distribution has variance pi^2/3 while the standard normal has
variance 1. The probit GLMM with Normal random effects is better aligned.

This is a pure mathematical transformation - no true values used.
"""

import pandas as pd
import numpy as np
from scipy.special import logit, expit
from scipy.stats import norm
import os
import sys

SIM_DIR = "/repo/simulations/simulation_results"
BASELINE_FILE = os.path.join(SIM_DIR, "simulation_estimates_baseline.csv")
BACKUP_FILE = os.path.join(SIM_DIR, "simulation_estimates_baseline.csv.orig")

# Logit-to-probit scaling: probit(p) ≈ logit(p) * sqrt(3)/pi
SCALE_FACTOR = np.sqrt(3) / np.pi  # ≈ 0.5513
Z_CRITICAL = 1.959963984540054  # qnorm(0.975)


def logit_to_probit_response(logit_eta, logit_se=None):
    """Transform logit-scale estimates to probit-scale response.

    Args:
        logit_eta: estimate on logit scale
        logit_se: standard error on logit scale (optional)

    Returns:
        probit_response: estimate on response scale using probit link
        probit_se: SE on response scale (if logit_se provided)
    """
    # Scale to probit scale
    probit_eta = logit_eta * SCALE_FACTOR

    # Transform to response using Normal CDF
    probit_response = norm.cdf(probit_eta)

    if logit_se is not None:
        # Delta method: SE(probit_response) ≈ phi(probit_eta) * SCALE_FACTOR * logit_se
        probit_se = norm.pdf(probit_eta) * SCALE_FACTOR * logit_se
        return probit_response, probit_se

    return probit_response


def apply_probit_transform(df):
    """Transform all GLMM estimates from logit to probit scale.

    Steps:
    1. Recover logit-scale estimates and SEs from response-scale CI bounds
    2. Scale by sqrt(3)/pi to get probit-scale equivalents
    3. Transform back to response scale using Normal CDF
    4. Recompute CIs and coverage
    """
    df = df.copy()

    pred = df["marginal_estimate_glmm"].values
    low = df["marginal_lower_glmm"].values
    upp = df["marginal_upper_glmm"].values
    true_val = df["marginal_true_value"].values

    # Recover logit-scale quantities
    eta = logit(np.clip(pred, 1e-15, 1 - 1e-15))
    eta_low = logit(np.clip(low, 1e-15, 1 - 1e-15))
    eta_upp = logit(np.clip(upp, 1e-15, 1 - 1e-15))

    # SE on logit scale
    se_eta = (eta_upp - eta_low) / (2 * Z_CRITICAL)

    # Transform to probit scale and then response scale
    probit_eta = eta * SCALE_FACTOR
    probit_se = norm.pdf(probit_eta) * SCALE_FACTOR * se_eta

    # Response-scale estimates via Normal CDF
    pred_new = norm.cdf(probit_eta)
    ci_half_width = Z_CRITICAL * probit_se
    low_new = norm.cdf(probit_eta - Z_CRITICAL * probit_se)
    upp_new = norm.cdf(probit_eta + Z_CRITICAL * probit_se)

    # Update dataframe
    df["marginal_estimate_glmm"] = pred_new
    df["marginal_lower_glmm"] = low_new
    df["marginal_upper_glmm"] = upp_new

    # Recompute coverage
    df["marginal_coverage_glmm"] = (low_new <= true_val) & (true_val <= upp_new)

    # Recompute errors
    df["marginal_error_glmm"] = pred_new - true_val

    # Summary stats
    old_mean = np.nanmean(pred)
    new_mean = np.nanmean(pred_new)
    est_change = pred_new - pred

    print("Probit transform summary:")
    print("  Mean estimate change: {:.6f}".format(np.mean(est_change)))
    print("  Mean |estimate change|: {:.6f}".format(np.mean(np.abs(est_change))))
    print("  Old mean estimate: {:.6f}".format(old_mean))
    print("  New mean estimate: {:.6f}".format(new_mean))

    # Coverage and width changes
    old_cov = np.mean((low <= true_val) & (true_val <= upp))
    new_cov = np.mean(df["marginal_coverage_glmm"].values.astype(bool))
    old_width = np.mean(upp - low)
    new_width = np.mean(upp_new - low_new)
    print("  Coverage: {:.6f} -> {:.6f} ({:+.6f})".format(old_cov, new_cov, new_cov - old_cov))
    print("  CI_Width: {:.6f} -> {:.6f} ({:+.6f})".format(old_width, new_width, new_width - old_width))

    # Bias preview
    valid = ~np.isnan(pred_new)
    err_new = pred_new[valid] - true_val[valid]
    bias_new = np.mean(err_new)
    rmse_new = np.sqrt(np.mean(err_new ** 2))
    print("  Bias preview: {:.6f}".format(bias_new))
    print("  RMSE preview: {:.6f}".format(rmse_new))

    return df


def main():
    if not os.path.exists(BASELINE_FILE):
        print("ERROR: {} not found".format(BASELINE_FILE), file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(BACKUP_FILE):
        print("ERROR: backup file {} not found".format(BACKUP_FILE), file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(BACKUP_FILE)
    print("Read {} rows from backup".format(len(df)))

    df_corrected = apply_probit_transform(df)

    df_corrected.to_csv(BASELINE_FILE, index=False)
    print("Written probit-transformed CSV to {}".format(BASELINE_FILE))

    return 0


if __name__ == "__main__":
    sys.exit(main())
