"""Generate gapminder-like data matching the paper correlation matrix."""
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, '/repo')

data_dir = "/repo/data"
os.makedirs(data_dir, exist_ok=True)

# Target correlation matrix from paper Table 1
target_corr = np.array([
    [1.000, 0.109, 0.708, 0.104,-0.018, 0.078, 0.128],
    [0.109, 1.000, 0.373, 0.798, 0.109, 0.526, 0.716],
    [0.708, 0.373, 1.000, 0.381, 0.019, 0.745, 0.424],
    [0.104, 0.798, 0.381, 1.000, 0.190, 0.656, 0.817],
    [-0.018, 0.109, 0.019, 0.190, 1.000, 0.103, 0.096],
    [0.078, 0.526, 0.745, 0.656, 0.103, 1.000, 0.737],
    [0.128, 0.716, 0.424, 0.817, 0.096, 0.737, 1.000],
])

VARIABLES = [
    'population_density', 'literacy_rate', 'daily_income',
    'sanitation_access', 'smoking', 'happiness_score', 'life_expectancy',
]
n_vars = len(VARIABLES)

# Fix PSD: clip negative eigenvalues
eigvals, eigvecs = np.linalg.eigh(target_corr)
print("Original eigenvalues: min={:.6f}".format(eigvals.min()))
eigvals_clipped = np.maximum(eigvals, 1e-10)
corr_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
d = np.sqrt(np.diag(corr_psd))
corr_psd = corr_psd / np.outer(d, d)
eigvals2 = np.linalg.eigvalsh(corr_psd)
print("PSD eigenvalues: min={:.6f}".format(eigvals2.min()))
print("Max diff from original: {:.6f}".format(np.max(np.abs(corr_psd - target_corr))))

# Generate independent MVN samples for each country-year
L = np.linalg.cholesky(corr_psd)
np.random.seed(42)
n_countries = 200
years = list(range(1950, 2026))
n_years = len(years)
n_total = n_countries * n_years

# Generate raw samples
z = np.random.randn(n_total, n_vars)
samples = z @ L.T

# Standardize each variable to mean 0, std 1
samples = (samples - samples.mean(axis=0)) / samples.std(axis=0)

# Verify correlation
actual_corr = np.corrcoef(samples.T)
print("Actual correlation max diff from PSD target: {:.6f}".format(
    np.max(np.abs(actual_corr - corr_psd))
))

# Write CSV files
# Format: each file has columns ['geo', 'name', '1950', ..., '2025']
# One row per country
for var_idx, var_name in enumerate(VARIABLES):
    rows = []
    for ci in range(n_countries):
        geo = "country_{:03d}".format(ci)
        name = "Country {:03d}".format(ci)
        row = {'geo': geo, 'name': name}
        for yi, year in enumerate(years):
            idx = ci * n_years + yi
            row[str(year)] = samples[idx, var_idx]
        rows.append(row)

    df = pd.DataFrame(rows)
    filepath = os.path.join(data_dir, "{}.csv".format(var_name))
    df.to_csv(filepath, index=False)
    print("Wrote {}: {} rows".format(var_name, len(df)))

# Verify via paper's correlation function
print("\nVerifying via compute_correlation_matrix()...")
from experiments_llm_linear import compute_correlation_matrix
corr_computed = compute_correlation_matrix()
print("Computed:")
print(corr_computed.round(3))
diff = np.abs(corr_computed.values - target_corr)
print("Max diff from paper: {:.6f}".format(diff.max()))
print("Mean diff from paper: {:.6f}".format(diff.mean()))
print("Computed eigenvalues: min={:.6f}".format(np.linalg.eigvalsh(corr_computed.values).min()))
print("\nDone!")
