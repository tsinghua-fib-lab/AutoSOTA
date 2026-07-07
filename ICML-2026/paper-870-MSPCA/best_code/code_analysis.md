# Code Analysis for Paper 870: Mean-Shift PCA by Knockoff Mean

## Evaluation Path
- **Entry point**: `/repo/eval.py` (created for reproduction)
- **Command**: `python3 eval.py`
- **Output format**: Parses stdout for `ALIGNMENT=<float>` and `RUNTIME_MS=<float>` in FINAL SUMMARY section
- **Config**: Synthetic data, seed=233, n=1000, d=900, c=0.9, n_trials=200
- **Primary metric**: Alignment (cosine similarity between recovered and true PC, percentage 0-100)
- **Secondary metric**: Runtime (milliseconds via perf_counter)

## Key Files
- **eval.py** (7188 bytes): Canonical evaluation script. Contains `estimate_theta_square`, `ms_pca`, `run_experiment`. This is the file to optimize.
- **main.py** (5518 bytes): Original research script with broader parameter grid. Contains both `ms_pca` (TruncatedSVD) and `_ms_pca` (full SVD) variants.
- **rebuttal.py, rebuttal2.py**: Extended experiments, not used for evaluation.
- **quick_test.py, reproduce_alignment.py, verify_env.py**: Test/verification scripts.

## Core Algorithm (`ms_pca` in eval.py, lines 33-78)
1. SVD of contaminated data X_tilde → U_tilde, S_tilde, Vh_tilde
2. Estimate theta from leading singular value: `estimate_theta_square(S_tilde[0], c)`
3. Generate knockoff perturbation A_prime with random direction and estimated noise
4. SVD of knockoff-perturbed data X_prime → U_prime, S_prime
5. Invariance check: component i is "stable" if ∃j s.t. |S_tilde[i] - S_prime[j]| < C/sqrt(n)
6. Return stable eigenvalues and components

## Safe Modification Targets
- `estimate_theta_square` (line 21-27): Numerical fix for negative discriminant (ALGO-01)
- `ms_pca` SVD call (line 38): Switch TruncatedSVD → LA.svd (CODE-01)
- `C` parameter (line 60): Adaptive thresholding (ALGO-02)
- `ms_pca` knockoff generation (lines 47-53): Ensemble/multiple directions (ALGO-03)
- Between SVD and theta estimation (after line 40): Singular value shrinkage (ALGO-04)
- `estimate_theta_square` call (line 42): Two-spectrum averaging (ALGO-07)
- Whole `ms_pca` call: Two-pass refinement (ALGO-06)

## Risky Files (DO NOT MODIFY)
- eval.py `__main__` block: Metric parsing, output format, experiment parameters
- eval.py `run_experiment`: Data generation, trial loop, alignment computation
- All CSV output paths and formats
- Seed, n_trials, noise_proportions, dimension parameters in eval.py

## Data
- All data is synthetically generated in-code (numpy, seed=233)
- No external datasets, models, or checkpoints needed
- `/paper_data` not mounted

## Repository State
- Git: main branch, 2 commits ahead of origin
- Tags: `_baseline` (commit 5f5503d, iter-0 baseline), `_best` (same as baseline)
- Container: `autosota_repro_paper_870`, Docker image `autosota/paper-870:reproduced`
- GPUs: 4,5
