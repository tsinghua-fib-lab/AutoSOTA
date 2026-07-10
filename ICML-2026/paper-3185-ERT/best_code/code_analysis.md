# Code Analysis for Paper 3185 SOTA Optimization

## Evaluation Path
- reproduce_sec42.py::main() generates data, applies CP, computes ERT via covmetrics
- Output: METRIC:L1-ERT=float and METRIC:L2-ERT=float on stdout

## Key Components
1. generate_data(n, rng) - creates X~U([-1,1]^8), Y~N(0, sigma(X1))
2. standard_cp_cover() - CP with S(X,Y)=|Y| (naive, ignores X)
3. run_single_experiment() - runs 1 experiment, returns L1-ERT, L2-ERT
4. ERT class - uses CheapLGBMClassifier, 5-fold CV

## Safe Modification Targets
- standard_cp_cover() - modify score function or add new CP methods
- run_single_experiment() - add method selection, data splitting
- main() - add CLI args, method iteration

## Red-line boundaries
Do NOT modify: ERT class, L1_miscoverage/losses, data generating process,
CV protocol (5-fold), alpha, n_cal, n_test, n_runs

## Bottleneck
Naive score S(X,Y)=|Y| uses global quantile, cannot adapt to heteroskedasticity.
sigma(X1) varies 0.5 to 2.5, so same |Y| means different things depending on X1.
Solution: use X-aware nonconformity scores.
