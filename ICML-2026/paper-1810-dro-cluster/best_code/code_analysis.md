# Code Analysis for SOTA Optimization — Paper 1810

## Evaluation Path

1. scripts/run_synthetic_experiments.py — CLI entry point
2. Calls run_synthetic_suite() -> run_method() for each (scenario, seed, method)
3. For DRO: compute_dro_coefficients() -> spectral_clustering() -> clustering_score()
4. Output: results/synthetic_results/synthetic_experiments.csv
5. Metric parsing: mean of ami column filtered by scenario=main, method=dro

## Key Functions and Files

### dro.py — Core DRO optimization
- dro_sqrt_delta(): Monte Carlo calibration of Wasserstein radius penalty
- spectral_least_squares_svd(): Spectral-norm proximal operator
- gradient_descent(): Nesterov-accelerated GD for ADMM subproblem
- spectral_regression_admm(): Main ADMM solver with adaptive rho
- compute_dro_coefficients(): Top-level entry point

### metrics.py — Clustering evaluation
- spectral_clustering(): Affinity construction -> Laplacian -> SVD -> KMeans
- clustering_score(): Hungarian matching + AMI computation

### experiments.py — Experiment orchestration
- ExperimentConfig: Holds all experiment parameters
- generate_trial(): Creates synthetic data + labels
- compute_trial_scores(): End-to-end DRO clustering pipeline

### synthetic.py — Data generation
- sample_random_subspace(): Generates synthetic subspace-structured data

## Safe Modification Targets
1. metrics.py:spectral_clustering() — affinity construction, embedding scaling
2. dro.py:gradient_descent() — restart strategy
3. dro.py:spectral_regression_admm() — rho schedule
4. dro.py:compute_dro_coefficients() — parameter wiring

## Risky Files (do NOT modify)
- metrics.py:clustering_score() — metric computation
- synthetic.py — data generation
- experiments.py:ExperimentConfig — evaluation protocol

## Baseline
- AMI: 0.9135, Accuracy: 0.8746
- 10 trials, seeds 2021-2030
- Commit: 9933303 (iter-0 baseline)
