#!/usr/bin/env python3
"""Diagnose bias patterns in GLMM estimates."""
import pandas as pd
import numpy as np
from scipy.special import logit

df = pd.read_csv("/repo/simulations/simulation_results/simulation_estimates_baseline.csv")
print("=== ERROR DISTRIBUTION ===")
err = df["marginal_error_glmm"]
print("Mean error: {:.6f}".format(err.mean()))
print("Median error: {:.6f}".format(err.median()))
print("Std error: {:.6f}".format(err.std()))
print("P5: {:.6f}, P95: {:.6f}".format(err.quantile(0.05), err.quantile(0.95)))

print()
print("=== ERROR BY LLM ===")
for llm in sorted(df["LLM_id"].unique()):
    e = df[df["LLM_id"] == llm]["marginal_error_glmm"]
    print("  {}: mean={:.6f}, std={:.6f}".format(llm, e.mean(), e.std()))

print()
print("=== ERROR BY TRUE VALUE QUINTILE ===")
df["true_quintile"] = pd.qcut(df["marginal_true_value"], 5)
for q, g in df.groupby("true_quintile", observed=True):
    e = g["marginal_error_glmm"]
    tv = g["marginal_true_value"].mean()
    print("  True={:.3f}: mean_err={:.6f}, std_err={:.6f}".format(tv, e.mean(), e.std()))

print()
print("=== CORRELATION: ESTIMATE VS TRUE ===")
r = df["marginal_estimate_glmm"].corr(df["marginal_true_value"])
print("  Pearson r: {:.4f}".format(r))

print()
print("=== COVERAGE BY TRUE VALUE ===")
for q, g in df.groupby("true_quintile", observed=True):
    cov = g["marginal_coverage_glmm"].astype(bool).mean()
    tv = g["marginal_true_value"].mean()
    print("  True={:.3f}: coverage={:.4f}".format(tv, cov))

print()
print("=== ESTIMATE VS TRUE DIRECTION ===")
for llm in sorted(df["LLM_id"].unique()):
    sub = df[df["LLM_id"] == llm]
    above = (sub["marginal_estimate_glmm"] > sub["marginal_true_value"]).mean()
    print("  {}: estimate > true: {:.3f}".format(llm, above))

print()
print("=== LINK-SCALE ANALYSIS ===")
eta_est = logit(np.clip(df["marginal_estimate_glmm"].values, 1e-15, 1-1e-15))
eta_true = logit(np.clip(df["marginal_true_value"].values, 1e-15, 1-1e-15))
eta_err = eta_est - eta_true
print("Link-scale mean error: {:.6f}".format(eta_err.mean()))
print("Link-scale median error: {:.6f}".format(np.median(eta_err)))

print()
print("=== BIAS DECOMPOSITION ===")
# Is the response-scale bias explained by link-scale bias + transformation?
eta_err_mean = eta_err.mean()
eta_err_std = eta_err.std()
pred_mean = df["marginal_estimate_glmm"].mean()
true_mean = df["marginal_true_value"].mean()
print("Response-scale bias: {:.6f}".format(pred_mean - true_mean))
print("Link-scale bias: {:.6f}".format(eta_err_mean))
print("Link-scale error std: {:.6f}".format(eta_err_std))
print("Approx Jensen bias (for eta~N(0, sigma^2)): {:.6f}".format(
    0.5 * eta_err_std**2 * df["marginal_true_value"].mean() * (1 - df["marginal_true_value"].mean()) * (1 - 2*df["marginal_true_value"].mean())))
