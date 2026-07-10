#!/usr/bin/env python3
"""
Generate optimal_power_results_combined.csv

Given alpha=0.05, n=50, find the (gamma, delta) pair that achieves near-optimal
detection power while minimizing KL divergence, then expand with delta/gamma offsets.

Output: optimal_power_results_combined.csv - 7 rows with
alpha, n_samples, gamma, delta, base_gamma.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from theorytools import KL_theory, power, solve_delta

warnings.filterwarnings('ignore')

# --- Step 1: Grid search for optimal (gamma, delta) ---
alpha, n_samples = 0.05, 50
print(f'Processing alpha={alpha}, n_samples={n_samples}')
base_gamma = 1.0 / (norm.ppf(1 - alpha)**2 / n_samples + 1.0)
klub_base = -np.log(base_gamma)

results = []
# Grid search and compare candidate (gamma, delta) pairs around the base KL level.
for kl_level in np.arange(0, 0.008, 0.0005) + klub_base:
    for space in np.arange(0, 0.008, 0.0003):
        gamma = base_gamma - space
        if gamma <= 0:
            continue
        try:
            delta_solved = solve_delta(kl_level, gamma)
        except Exception:
            continue
        p = power(gamma, delta_solved, alpha, n_samples)
        kl_verified = KL_theory(gamma, delta_solved)
        results.append({
            'alpha': alpha, 'n_samples': n_samples,
            'gamma': gamma, 'delta': delta_solved,
            'design_power': p, 'kl_fixed': kl_level,
            'kl_verified': kl_verified, 'space': space,
            'base_gamma': base_gamma, 'klub': klub_base,
        })

df = pd.DataFrame(results)
df['kl_error'] = (df['kl_verified'] - df['kl_fixed']).abs()
df = df[df['kl_error'] < 0.01]
df = df[df['design_power'] >= 0.95]
df = df[df['delta'] <= 20]

# Top-5 by power, then pick lowest KL
result = df.nlargest(5, 'design_power').nsmallest(1, 'kl_verified').reset_index(drop=True)
print(f'Base results: {len(result)} rows')

# --- Step 2: Expand with delta/gamma offsets ---
base = result.copy()

# Delta offsets (gamma unchanged): reduce delta by 0, 1, 2, 3, 4
rows_d = []
for _, row in base.iterrows():
    for doff in np.arange(0, 5, dtype=float):
        r = row.copy()
        r['delta_reduction'] = doff
        r['gamma_reduction'] = 0.0
        r['delta'] = row['delta'] - doff
        if r['delta'] > 0:
            rows_d.append(r)

# Gamma offsets (delta unchanged): reduce gamma by 0.1, 0.2
rows_g = []
for _, row in base.iterrows():
    for goff in [0.1, 0.2]:
        r = row.copy()
        r['gamma_reduction'] = goff
        r['delta_reduction'] = 0.0
        r['gamma'] = row['gamma'] - goff
        if 0 < r['gamma'] < 1:
            rows_g.append(r)

expanded = pd.DataFrame(rows_d + rows_g).reset_index(drop=True)

# Recompute metrics used for filtering/diagnostics before writing the compact CSV.
for i, r in expanded.iterrows():
    g, d = float(r['gamma']), float(r['delta'])
    a, n = float(r['alpha']), int(r['n_samples'])
    expanded.at[i, 'design_power'] = power(g, d, a, n)
    expanded.at[i, 'kl_verified'] = KL_theory(g, d)

final_result = expanded[['alpha', 'n_samples', 'gamma', 'delta', 'base_gamma']].copy()

# --- Step 3: Save ---
final_result.to_csv('optimal_power_results_combined.csv', index=False)
print(f'Saved: optimal_power_results_combined.csv ({len(final_result)} rows)')
print(final_result.to_string())
