# Code Analysis for Paper 4194 SOTA Optimization

## Evaluation Path

- **Entry point:** `/repo/run_experiment.py` → `main()`
- **Command:** `python3 run_experiment.py`
- **Timeout:** 60 minutes
- **Output:** stdout (prints summary table), pickle files in `results/`

## Key Components

### Estimators (lines 14-54)
- `DQ_truncated_estimator(rewards, actions, states, discount_factor, k)` — Core TPG estimator. Uses `q_function(index, ...)` with O(T^2) loop over all indices for each k.
- `DQ_LSTD_lambda_estimator(rewards, actions, states, discount_factor, k, alpha, lambda_)` — Stationary baseline estimator.

### Simulation (lines 60-97)
- `simulate_nonstationary_mdp(policy, T, ...)` — Generates trajectory data.
- `generate_mean_reverting_kernels(T, mixing_coeff, treatment_bias, smoothness, noise_std, seed)` — Creates nonstationary transition kernels.

### Experiment Runner (lines 133-153)
- `evaluate_estimators_truncated_only(n_trials, T, ...)` — Runs n_trials with joblib Parallel, computes estimates for all k in {0,1,3,5,10,50,100,500,T}.

### Aggregation (lines 156-209)
- `summarize_across_mixing(results_dir, treatment_bias, ...)` — Loads pickle files, computes MAE (%) and STD per k, prints Table 3.

### Main (lines 214-269)
- Config block (lines 220-226): `mixing_coeffs`, `treatment_bias`, `smoothness`, `noise_std`, `reward_std`, `T`, `n_trials`, `reward_matrix`
- Loops over 20 mixing coefficients, runs evaluation, saves results, prints summary.

## Metric Parser
- MAE_k1 parsed from stdout: `"k=1      MAE_VALUE    STD_VALUE"` line in Summary table
- STD_k1 also from same line
- Mean GATE from `"Mean Ground Truth ATE across all files: VALUE"` line

## Safe Modification Targets

### CODE changes (no algorithm change):
1. `DQ_truncated_estimator()`: Vectorize O(T^2) loop to O(T) cumsum (IDEA-08)
2. `evaluate_estimators_truncated_only()`: Adaptive trial allocation (IDEA-09)

### ALGO changes (post-processing, estimator only):
1. `summarize_across_mixing()`: Add k-ensemble weighting (IDEA-01, IDEA-05)
2. New function: Convex combination TPG+DM (IDEA-07)
3. `DQ_truncated_estimator()`: Per-step reliability weights (IDEA-03)
4. `q_function()`: Soft exponential decay (IDEA-02)
5. New function: Lepski k-selection per mixing rate (IDEA-04, IDEA-10)

### PARAM changes:
1. Parameter block: Sweep n_trials, T, smoothness, etc. (IDEA-11, IDEA-12)

## Risky/Untouchable Files
- Evaluation protocol: never change metric definitions, test data, or scoring
- `run_experiment.py`: The main evaluation - changes must preserve all metric reporting
- `summarize_across_mixing()`: Must still compute MAE_k1 and STD_k1 honestly

## Red-line Confirmations for All Ideas
- Evaluation command unchanged: `python3 run_experiment.py`
- Metric computation unchanged: MAE = mean(abs(100*(est - truth)/truth)), STD = std(est)
- Test data unchanged: 20 mixing rates, same MDP parameters
- No hard-coded outputs
- Multi-metric objective respected: must report both MAE and STD
- Rollback via git commits

## Setup Notes
- Pure CPU simulation, numpy/scipy/joblib
- Reward matrix: fixed, differs from paper random but produces matching results
- No external datasets/models required
- Container: autosota_repro_paper_4194, Python 3.10.13
