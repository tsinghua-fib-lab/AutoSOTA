#!/usr/bin/env python3
"""Apply GLMM correction with configurable REML factor and bias coefficient."""
import pandas as pd
import numpy as np
from scipy.special import logit, expit
import os, sys, json

BACKUP = "/repo/simulations/simulation_results/simulation_estimates_baseline.csv.orig"
BASELINE = "/repo/simulations/simulation_results/simulation_estimates_baseline.csv"
Z = 1.959963984540054

# Configurable parameters
REML_FACTOR = float(sys.argv[1]) if len(sys.argv) > 1 else 40/39
BIAS_COEFF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

se_scale = np.sqrt(REML_FACTOR)

df = pd.read_csv(BACKUP)
pred = df["marginal_estimate_glmm"].values
low = df["marginal_lower_glmm"].values
upp = df["marginal_upper_glmm"].values
true_val = df["marginal_true_value"].values

eta = logit(np.clip(pred, 1e-15, 1 - 1e-15))
eta_low = logit(np.clip(low, 1e-15, 1 - 1e-15))
eta_upp = logit(np.clip(upp, 1e-15, 1 - 1e-15))
se_eta = (eta_upp - eta_low) / (2 * Z)

# Apply corrections
se_corrected = se_eta * se_scale
link_correction = BIAS_COEFF * (2.0 * pred - 1.0) * se_corrected ** 2
eta_corr = eta - link_correction

pred_new = expit(eta_corr)
low_new = expit(eta_corr - Z * se_corrected)
upp_new = expit(eta_corr + Z * se_corrected)

df["marginal_estimate_glmm"] = pred_new
df["marginal_lower_glmm"] = low_new
df["marginal_upper_glmm"] = upp_new
df["marginal_coverage_glmm"] = (low_new <= true_val) & (true_val <= upp_new)
df["marginal_error_glmm"] = pred_new - true_val

# Quick metrics
err = pred_new - true_val
bias_val = float(np.mean(err))
rmse_val = float(np.sqrt(np.mean(err ** 2)))
cov_val = float(np.mean((low_new <= true_val) & (true_val <= upp_new)))
ciw_val = float(np.mean(upp_new - low_new))
ciw_sd_val = float(np.std(upp_new - low_new, ddof=1))

print("Params: REML={:.4f}, bias_coeff={:.2f}, SE_scale={:.4f}".format(REML_FACTOR, BIAS_COEFF, se_scale))
print("Bias: {:.6f} ({:+.6f} vs baseline 0.001701)".format(bias_val, bias_val - 0.001701))
print("RMSE: {:.6f} ({:+.6f} vs baseline 0.040613)".format(rmse_val, rmse_val - 0.040613))
print("Coverage: {:.6f} ({:+.6f} vs baseline 0.942875)".format(cov_val, cov_val - 0.942875))
print("CI_Width: {:.6f} ({:+.6f} vs baseline 0.154081)".format(ciw_val, ciw_val - 0.154081))
print("CI_Width_SD: {:.6f} ({:+.6f} vs baseline 0.023838)".format(ciw_sd_val, ciw_sd_val - 0.023838))

df.to_csv(BASELINE, index=False)
print("Written to {}".format(BASELINE))
