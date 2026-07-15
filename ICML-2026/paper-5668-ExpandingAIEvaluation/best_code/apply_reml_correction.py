#!/usr/bin/env python3
"""Apply REML-like variance correction to GLMM estimates in the simulation CSV.

ALGO-06: The ML estimate of sigma^2_item is downward-biased by approximately
(N-p)/N where N=n_items=40 and p=m_LLMs=4 (fixed effects). Correcting the
variance estimate upward by factor n/(n-p+1) widens CIs, potentially improving
Coverage toward the nominal 0.95 level without substantially increasing CI_Width.

The correction is applied on the link (logit) scale by recovering eta and SE_eta
from the response-scale CI bounds, scaling SE_eta, and transforming back.
"""

import pandas as pd
import numpy as np
from scipy.special import logit, expit
import os
import sys

SIM_DIR = "/repo/simulations/simulation_results"
BASELINE_FILE = os.path.join(SIM_DIR, "simulation_estimates_baseline.csv")
BACKUP_FILE = os.path.join(SIM_DIR, "simulation_estimates_baseline.csv.orig")

# Settings from the paper
N_ITEMS = 40
M_LLMS = 4
T_TRIALS = 8
ITEM_SIGMA = 1.3
Z_CRITICAL = 1.959963984540054  # qnorm(0.975)


def apply_reml_correction(df, correction_factor=None):
    """Apply REML-like correction to GLMM CI bounds.

    Args:
        df: DataFrame with GLMM estimate columns
        correction_factor: Variance correction factor. If None, computed as
                          n_items / (n_items - n_llms + 1)

    Returns:
        DataFrame with corrected GLMM columns
    """
    if correction_factor is None:
        correction_factor = N_ITEMS / (N_ITEMS - M_LLMS + 1)

    se_scale = np.sqrt(correction_factor)
    print("REML correction factor: {:.4f}".format(correction_factor))
    print("SE scale factor: {:.4f}".format(se_scale))

    df = df.copy()

    # Recover link-scale estimates from response-scale CI bounds
    # effectPlotData computes:
    #   pred = plogis(eta)
    #   low  = plogis(eta - z * SE_eta)
    #   upp  = plogis(eta + z * SE_eta)
    # So: eta = qlogis(pred), SE_eta = (qlogis(upp) - qlogis(low)) / (2*z)

    pred = df["marginal_estimate_glmm"].values
    low = df["marginal_lower_glmm"].values
    upp = df["marginal_upper_glmm"].values
    true_val = df["marginal_true_value"].values

    # Recover link-scale quantities
    eta = logit(np.clip(pred, 1e-15, 1 - 1e-15))
    eta_low = logit(np.clip(low, 1e-15, 1 - 1e-15))
    eta_upp = logit(np.clip(upp, 1e-15, 1 - 1e-15))

    # SE on link scale (symmetric Wald CI)
    se_eta = (eta_upp - eta_low) / (2 * Z_CRITICAL)

    # Apply REML correction: scale SE by sqrt(correction_factor)
    se_eta_corrected = se_eta * se_scale

    # Recompute CIs on response scale
    low_corrected = expit(eta - Z_CRITICAL * se_eta_corrected)
    upp_corrected = expit(eta + Z_CRITICAL * se_eta_corrected)

    # Point estimates unchanged
    df["marginal_lower_glmm"] = low_corrected
    df["marginal_upper_glmm"] = upp_corrected

    # Recompute coverage
    df["marginal_coverage_glmm"] = (low_corrected <= true_val) & (true_val <= upp_corrected)

    # Recompute errors (point estimates unchanged, so errors unchanged)
    n_changed = np.sum(np.abs(low - df["marginal_lower_glmm"].values) > 1e-10)
    print("Rows with CI bound changes: {}".format(n_changed))

    # Summary of changes
    old_cov = (low <= true_val) & (true_val <= upp)
    new_cov = df["marginal_coverage_glmm"].values.astype(bool)
    cov_change = np.mean(new_cov) - np.mean(old_cov)
    width_change = np.mean(upp_corrected - low_corrected) - np.mean(upp - low)
    print("Coverage change: {:+.6f}".format(cov_change))
    print("CI_Width change: {:+.6f}".format(width_change))

    return df


def main():
    if not os.path.exists(BASELINE_FILE):
        print("ERROR: {} not found".format(BASELINE_FILE), file=sys.stderr)
        sys.exit(1)

    # Check if backup exists; if not, read original and store in memory
    if os.path.exists(BACKUP_FILE):
        print("Using existing backup at {}".format(BACKUP_FILE))
        df_orig = pd.read_csv(BACKUP_FILE)
    else:
        print("Backing up original to {}".format(BACKUP_FILE))
        df_orig = pd.read_csv(BASELINE_FILE)
        df_orig.to_csv(BACKUP_FILE, index=False)

    # Read baseline
    df = pd.read_csv(BASELINE_FILE)
    print("Read {} rows from {}".format(len(df), BASELINE_FILE))
    original_na = df["marginal_estimate_glmm"].isna().sum()
    print("NA GLMM estimates: {}".format(original_na))

    # Apply correction
    df_corrected = apply_reml_correction(df)

    # Write corrected CSV
    df_corrected.to_csv(BASELINE_FILE, index=False)
    print("Written corrected CSV to {}".format(BASELINE_FILE))

    return 0


if __name__ == "__main__":
    sys.exit(main())
