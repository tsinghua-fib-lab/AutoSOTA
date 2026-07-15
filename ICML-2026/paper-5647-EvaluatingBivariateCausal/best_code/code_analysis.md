# Code Analysis: Paper 5647 — Compatibility Scores

## Evaluation Path
- Entry: eval_compatibility.py compute_random_baseline()
- Core: synthetic_experiments_linear.py compatibility_score(A, cov)
- Correlation: experiments_llm_linear.py compute_correlation_matrix()
- Output: JSON with compatibility_score_random_baseline_mean and _std

## Key Functions
1. compatibility_score(A, cov) — objective to maximize
2. _estimate_bivariate_matrix(cov) in sensitivity_experiment.py — estimates A from chol(cov)
3. sample_sparse_causal_model(n, p, num_hidden) — synthetic models (not for eval)

## Variables (7)
population_density, literacy_rate, daily_income, sanitation_access, smoking, happiness_score, life_expectancy

## Safe Modification Targets
- eval_compatibility.py: modify A matrix generation
- synthetic_experiments_linear.py: read-only (score functions)

## Risky Files (DO NOT MODIFY)
- synthetic_experiments_linear.py compatibility_score(), bivariate_confounding(), multivariate_confounding()
- experiments_llm_linear.py compute_correlation_matrix()
- /repo/data/ — test data

## Known Capabilities
- _estimate_bivariate_matrix(cov) from sensitivity_experiment.py not used in eval
- Correlation matrix is PSD
- Score differentiable w.r.t. lower-triangular A entries
