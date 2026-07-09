# Code Analysis for Paper 2614 SOTA Optimization

## Evaluation Path
- reproduce_linreg.py - Main evaluation script
  - Generates synthetic data (linear regression, k=1, sigma=10)
  - Creates bootstrap distributions
  - Runs bilevel optimization via optimize_transportation_matrix()
  - Computes e_oos with 10^7 OOS samples
  - Output: stdout REPRODUCTION RESULT block + JSON in /repo/results/

## Key Source Files
- src/trainable_ot_dro/bilevel_optimization.py - Core optimization loop
  - Adam optimizer beta1=0.9, beta2=0.999
  - Learning rate 1e-4
  - Gradient clipping +/-1e4
  - Key: lines 92-93 (lr), 100-101 (clip), 121-122 (Adam), 243-258 (Adam update), 263 (stabilize_L)

- src/trainable_ot_dro/conic_problem.py - Clarabel solver
  - New solver per iteration (no warm-start)
  - time_limit=10s

- src/trainable_ot_dro/utils/numerical_utilities.py - stabilize_L()
  - Eigenvalue clipping [1e-6, 1e6]

## Config (reproduce_linreg.py)
- n_iter_max=1e6, lr=1e-4, store_every=100
- penalization: lambda=10.0, eta=100.0
- J=20, nb=20, wasserstein_type=1

## Safe Targets
- bilevel_optimization.py: Adam params, LR schedule, gradient clipping, regularization
- reproduce_linreg.py: opt_params, penalization, L_init
- conic_problem.py: solver settings, warm-start

## Red Lines
- No changes to metric computation, data generation seeds, eval protocol
- No hard-coded outputs

## Baseline
- e_oos=8.3517, e_wc=8.2119, ~9800 iterations, ~389s wall time
