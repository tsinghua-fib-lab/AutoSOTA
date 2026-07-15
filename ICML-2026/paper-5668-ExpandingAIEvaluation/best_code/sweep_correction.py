#!/usr/bin/env python3
"""Sweep REML factor and bias coefficient to find Pareto-optimal combination."""
import pandas as pd
import numpy as np
from scipy.special import logit, expit
import json

BACKUP = "/repo/simulations/simulation_results/simulation_estimates_baseline.csv.orig"
Z = 1.959963984540054
N_ITEMS = 40

df_orig = pd.read_csv(BACKUP)
pred = df_orig["marginal_estimate_glmm"].values
low = df_orig["marginal_lower_glmm"].values
upp = df_orig["marginal_upper_glmm"].values
true_val = df_orig["marginal_true_value"].values

eta = logit(np.clip(pred, 1e-15, 1 - 1e-15))
eta_low = logit(np.clip(low, 1e-15, 1 - 1e-15))
eta_upp = logit(np.clip(upp, 1e-15, 1 - 1e-15))
se_eta = (eta_upp - eta_low) / (2 * Z)

results = []

# Sweep REML factors and bias coefficients
reml_factors = [1.0, 40/39.5, 40/39, 40/38.5, 40/38, 40/37.5, 40/37]
bias_coeffs = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

for reml_factor in reml_factors:
    se_scale = np.sqrt(reml_factor) if reml_factor > 0 else 1.0
    se_corrected = se_eta * se_scale

    for bias_c in bias_coeffs:
        # Link-scale bias correction
        link_correction = bias_c * (2.0 * pred - 1.0) * se_corrected ** 2
        eta_corr = eta - link_correction

        # Response scale
        pred_new = expit(eta_corr)
        low_new = expit(eta_corr - Z * se_corrected)
        upp_new = expit(eta_corr + Z * se_corrected)

        # Metrics
        err = pred_new - true_val
        bias_val = float(np.mean(err))
        rmse_val = float(np.sqrt(np.mean(err ** 2)))
        cov_val = float(np.mean((low_new <= true_val) & (true_val <= upp_new)))
        ciw_val = float(np.mean(upp_new - low_new))
        ciw_sd_val = float(np.std(upp_new - low_new, ddof=1))

        results.append({
            "reml_factor": round(reml_factor, 6),
            "bias_coeff": bias_c,
            "Bias": round(bias_val, 6),
            "RMSE": round(rmse_val, 6),
            "Coverage": round(cov_val, 6),
            "CI_Width": round(ciw_val, 6),
            "CI_Width_SD": round(ciw_sd_val, 6),
        })

# Find Pareto-optimal: min Bias, min CI_Width, max Coverage
# Filter: Coverage >= 0.940 (at least maintain near baseline)
valid = [r for r in results if r["Coverage"] >= 0.940]

# Also filter: CI_Width <= 0.160 (don't lose too much GLMM advantage)
constrained = [r for r in valid if r["CI_Width"] <= 0.160]

# Sort by Bias (primary)
constrained.sort(key=lambda r: r["Bias"])

print("=" * 100)
print("Top 15 by Bias (Coverage >= 0.940, CI_Width <= 0.160):")
print("=" * 100)
print("{:>5} {:>10} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12}".format(
    "Rank", "REML", "BiasC", "Bias", "RMSE", "Coverage", "CI_Width", "CI_Width_SD"))
print("-" * 100)
for i, r in enumerate(constrained[:15]):
    marker = " ***" if i == 0 else ""
    print("{:>5} {:>10.4f} {:>8.2f} {:>10.6f} {:>10.6f} {:>10.6f} {:>12.6f} {:>12.6f}{}".format(
        i+1, r["reml_factor"], r["bias_coeff"], r["Bias"], r["RMSE"],
        r["Coverage"], r["CI_Width"], r["CI_Width_SD"], marker))

# Also show best Coverage
print()
print("Top 10 by Coverage (Bias <= 0.0015, CI_Width <= 0.160):")
print("-" * 100)
by_cov = [r for r in constrained if r["Bias"] <= 0.0015]
by_cov.sort(key=lambda r: -r["Coverage"])
for i, r in enumerate(by_cov[:10]):
    print("{:>5} {:>10.4f} {:>8.2f} {:>10.6f} {:>10.6f} {:>10.6f} {:>12.6f} {:>12.6f}".format(
        i+1, r["reml_factor"], r["bias_coeff"], r["Bias"], r["RMSE"],
        r["Coverage"], r["CI_Width"], r["CI_Width_SD"]))

# Baseline for reference
print()
print("Baseline: Bias=0.001701, RMSE=0.040613, Coverage=0.942875, CI_Width=0.154081, CI_Width_SD=0.023838")
print("Current best: Bias=0.001044, RMSE=0.040465, Coverage=0.947250, CI_Width=0.156319, CI_Width_SD=0.023883")
print("(REML=40/39≈1.0256, bias_coeff=0.25)")

# Find Pareto front
print()
print("=== PARETO FRONT (Coverage >= 0.940) ===")
pareto = []
for r in valid:
    dominated = False
    for s in valid:
        if (s["Bias"] <= r["Bias"] and s["CI_Width"] <= r["CI_Width"] and s["Coverage"] >= r["Coverage"]
                and (s["Bias"] < r["Bias"] or s["CI_Width"] < r["CI_Width"] or s["Coverage"] > r["Coverage"])):
            dominated = True
            break
    if not dominated:
        pareto.append(r)

pareto.sort(key=lambda r: r["Bias"])
for r in pareto:
    print("REML={:.4f}, biasC={:.2f}: Bias={:.6f}, Coverage={:.6f}, CI_Width={:.6f}".format(
        r["reml_factor"], r["bias_coeff"], r["Bias"], r["Coverage"], r["CI_Width"]))
