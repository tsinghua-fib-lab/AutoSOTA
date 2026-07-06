# Code Analysis for Paper 1255: Optimal Transport under Group Fairness Constraints

## Evaluation Path
- Script: exps/exp_gaussian/exp_penalized_ot.py
- Flow: Data generation -> PenalizedOT.solve() -> results saved to pickle
- Data: Synthetic Gaussian mixture (2D, 250 source, 25 target, 2 groups)
- Solver: GCG (Generalized Conditional Gradient) with Sinkhorn inner LP

## Key Files
- exps/exp_gaussian/exp_penalized_ot.py: Eval script, config (Safe to modify)
- src/penalized_ot.py: Main solver class, orchestrates grid solve
- src/solvers.py: GCG, FairSinkhorn, penalized_ot_solver
- src/loss_funcs.py: quota_loss, weighted_quota_loss (both implemented)
- src/datagen.py: Synthetic data generation

## Metric Parser
- Output: results/exp_gaussian/results_penalized.pkl (pandas DataFrame)
- Key columns: cost_diff, fairness_loss_value, penalty
- cost_diff = |fair_cost - true_cost| where true_cost uses vanilla Sinkhorn

## Baseline Metrics
- cost_diff: 1.313 (at penalty approx 33)
- fairness_loss_value: 0.0023
- Full curve: penalty in [1, 1000], 80 points logspace

## Safe Modification Targets
1. Loss function: quota_loss -> weighted_quota_loss (ALGO-03)
2. Solver parameters: eps, numItermax, stopThr, numInnerItermax
3. GCG initialization: Warm-starting, FairSinkhorn init
4. Solver internals: Annealing, gradient clipping, adaptive tolerance
5. Penalty grid: Density, range, 2D sweep with epsilon
6. Data generation: n_samples, scale, centers, seeds

## Red-Line Boundaries
- DO NOT change metric computation (cost_diff formula)
- DO NOT change test data labels or target fairness matrix F
- DO NOT hard-code outputs or metric values
- DO NOT change evaluation protocol (pickle output format)
