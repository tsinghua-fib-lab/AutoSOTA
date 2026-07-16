# Code Analysis: MiP-CRIM (Paper 6308)

## Evaluation Path
- **Eval script**: `eval_sk_model.py` - canonical evaluation
- **Entry point**: `main()` -> 100 trials on SK model (n=1000 spins)
- **Per trial**: `make_sk_matrix()` -> `MiP_CRIM()` -> `sync_ratio()`
- **Output parsing**: stdout lines: "Best Energy:", "Mean Energy:", "Best Sync:", "Mean Sync:", "Mean Runtime:"

## Core Algorithm
- **File**: `mip_crim.py` -> `MiP_CRIM()` function
- **Method**: Basin-hopping with Adam optimizer on quartic polynomial relaxation
- **Energy**: H(x) = (beta/4)||x||_4^4 - (1/2)x^T(J + alpha*I)x, minimized over [-lambda, lambda]^n
- **Thresholding**: x >= 0 -> +1, x < 0 -> -1
- **Restarts**: K epochs, each with T Adam steps, then Gaussian perturbation for next init

## Current Parameters (Table 2, fixed)
- T=10 (inner Adam steps), K=200 (epochs/restarts)
- alpha=1.4996e-5, beta=0.001, lambda=0.0707
- step=1.0, beta1=0.09, beta2=0.999, eps=1e-8
- sigma_noise=1e-3

## Admissible Range
- Condition: 3*beta*lambda^2 < alpha < beta*lambda^2 + gamma_0
- gamma_0 = 1e-5 (LSB from rounding J to 5 decimals)
- Current: 3*beta*lambda^2 = 1.499547e-5, beta*lambda^2 + gamma_0 = 1.499849e-5
- Range width: only ~3.02e-9! Extremely tight.

## Key Observations
1. Parameters tuned on n=100, evaluated on n=1000 - potential scale mismatch
2. Admissible range for alpha is razor-thin with current lambda/beta
3. Sync=1.0 is maintained by the landscape guarantee, not by explicit filtering in eval
4. Runtime is fast (~0.26s/trial), allowing parameter search
5. No external data needed - all SK matrices generated on-the-fly with deterministic seeds
6. Two regimes: Table 1 (adaptive tuning per problem class) and Table 2 (fixed params)
7. FEM competitor achieves -16814.75 (better than our -16767.25 baseline)

## Safe Modification Targets
- `eval_sk_model.py`: params dict (the hyperparameters)
- `mip_crim.py`: MiP_CRIM algorithm body (K, T, sigma_noise schedule, multi-trajectory)
- `iamp_sk_solver.py`: sync_ratio (do NOT modify - metric definition)

## Risky Files (DO NOT MODIFY)
- `iamp_sk_solver.py`: sync_ratio() - metric computation
- `eval_sk_model.py`: make_sk_matrix(), main() flow, metric printing format
- `benchmark_SK.py`: make_sk_matrix() - data generation (used by eval)

## Idea Library (initial)
- IDEA-001 (ALGO): Increase K (epochs) for more basin-hopping restarts
- IDEA-002 (ALGO): Annealing schedule on sigma_noise (hot start, cool down)
- IDEA-003 (ALGO): Multi-trajectory per restart (run M trajectories, pick best)
- IDEA-004 (CODE): Re-tune alpha/lambda/beta for n=1000 scale
- IDEA-005 (CODE): Adaptive restart: increase sigma when stuck in same energy
- IDEA-006 (PARAM): Increase T (inner Adam steps) for better convergence
- IDEA-007 (ALGO): Different Adam momentum (beta1) - current 0.09 is very low
- IDEA-008 (CODE): Best-of-N solution pool with final selection
