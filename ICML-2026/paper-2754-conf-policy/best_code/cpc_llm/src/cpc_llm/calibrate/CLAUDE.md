# calibrate/

Implements the Conformal Policy Control (CPC) algorithm — the core contribution of this project. Searches for the most aggressive policy bound (beta) that still satisfies a user-specified risk level (alpha).

## Key files

- **cpc_search.py** — `cpc_beta_search()`: main entry point. Loads calibration data, generates proposals, searches over a grid of beta values, and returns the selected (beta, psi) parameters with constrained likelihoods.
- **process_likelihoods.py** — `constrain_likelihoods()`: constrains unconstrained likelihoods using `min(beta * safe_lik, unconstrained_lik) / psi`. Two modes: "init" (constrain against first model) and "sequential" (constrain against previous model in chain). Also has `mixture_pdf_from_densities_mat()` for computing weighted density mixtures.
- **grid.py** — `prepare_grid()`: sorts, deduplicates, and coarsens likelihood ratio values into a search grid. Appends boundary values depending on proposal type (inf for unconstrained, float_min for safe).
- **normalization.py** — `importance_weighted_monte_carlo_integration()`: estimates the normalization constant psi via IWMCI. `iwmci_overlap_est()`: estimates density overlap between constrained and proposal policies.

## Data flow

Calibration data arrives as DataFrames with columns `[particle, score, lik_r0, ..., lik_rT]`. The search iterates over beta values, computing conformal weights on calibration points plus a test point, and returns the largest beta where the weighted risk stays below alpha.
