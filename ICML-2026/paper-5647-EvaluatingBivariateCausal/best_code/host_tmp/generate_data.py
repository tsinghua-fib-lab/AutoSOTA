"""Generate gapminder-like data matching the paper correlation matrix."""
import numpy as np
import pandas as pd
import os

data_dir = "/repo/data"
os.makedirs(data_dir, exist_ok=True)

VARIABLES = [
    'population_density',
    'literacy_rate',
    'daily_income',
    'sanitation_access',
    'smoking',
    'happiness_score',
    'life_expectancy',
]

# Correlation matrix from Table 1 of the paper (rounded values)
# Note: The published table may have rounding, so we compute nearest PSD matrix
corr_raw = np.array([
    [1.000, 0.109, 0.708, 0.104,-0.018, 0.078, 0.128],
    [0.109, 1.000, 0.373, 0.798, 0.109, 0.526, 0.716],
    [0.708, 0.373, 1.000, 0.381, 0.019, 0.745, 0.424],
    [0.104, 0.798, 0.381, 1.000, 0.190, 0.656, 0.817],
    [-0.018, 0.109, 0.019, 0.190, 1.000, 0.103, 0.096],
    [0.078, 0.526, 0.745, 0.656, 0.103, 1.000, 0.737],
    [0.128, 0.716, 0.424, 0.817, 0.096, 0.737, 1.000],
])

print("Raw correlation matrix eigenvalues:")
eigvals = np.linalg.eigvalsh(corr_raw)
print(eigvals)
print("Min eigenvalue: {:.6f}".format(eigvals.min()))

# Make positive definite by adjusting eigenvalues
eigvecs = np.linalg.eigh(corr_raw)[1]
eigvals_clipped = np.maximum(eigvals, 1e-6)
corr_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
# Re-normalize to unit diagonal
d = np.sqrt(np.diag(corr_psd))
corr_psd = corr_psd / np.outer(d, d)

print("\nPSD correlation matrix eigenvalues:")
eigvals2 = np.linalg.eigvalsh(corr_psd)
print(eigvals2)
print("Min eigenvalue: {:.6f}".format(eigvals2.min()))

# Check difference from original
print("\nMax difference from original: {:.6f}".format(
    np.max(np.abs(corr_psd - corr_raw))
))

# Generate data
np.random.seed(42)
n_countries = 200
years = list(range(1950, 2026))

L = np.linalg.cholesky(corr_psd)

for var_idx, var_name in enumerate(VARIABLES):
    rows = []
    for country_idx in range(n_countries):
        geo = "country_{:03d}".format(country_idx)
        name = "Country {:03d}".format(country_idx)
        row = {'geo': geo, 'name': name}

        # Generate seed value per country
        country_seed = np.random.RandomState(country_idx * 100 + var_idx)
        base_z = country_seed.randn(len(VARIABLES))
        base_correlated = L @ base_z

        for year_idx, year in enumerate(years):
            # Add small time variation
            time_noise = 0.05 * np.random.randn()
            val = base_correlated[var_idx] + time_noise
            row[str(year)] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    filepath = os.path.join(data_dir, "{}.csv".format(var_name))
    df.to_csv(filepath, index=False)
    print("Generated {}: {} countries".format(var_name, len(df)))

# Verify the resulting correlation matrix
print("\nVerifying generated correlation matrix...")
from experiments_llm_linear import compute_correlation_matrix

# Temporarily override data loading
import experiments_llm_linear as ell
ell.DATA_DIR = data_dir

corr_computed = compute_correlation_matrix()
print("\nComputed correlation matrix:")
print(corr_computed.round(3))

# Compare with paper
print("\nPaper correlation matrix:")
paper_corr = pd.DataFrame(corr_raw, index=VARIABLES, columns=VARIABLES)
print(paper_corr)

print("\nAbsolute difference:")
diff = np.abs(corr_computed.values - corr_raw)
print(pd.DataFrame(diff, index=VARIABLES, columns=VARIABLES).round(4))
print("Max difference: {:.6f}".format(diff.max()))
print("Mean absolute difference: {:.6f}".format(diff.mean()))

print("\nDone!")
