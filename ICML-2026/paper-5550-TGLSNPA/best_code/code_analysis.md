# Code Analysis for Paper 5550: Torus Graphs for Large Scale Neural Phase Analysis

## Evaluation Path
- **Benchmark script:** `/repo/benchmark_runtime.py`
- **Evaluation command:** `python3 benchmark_runtime.py`
- **Metric parser:** Regex `/SSM Runtime: ([0-9.]+) seconds/` from stdout
- **Baseline Runtime:** 35.77s on A100 (paper: 137±3s on A5000)

## Key Files
1. `benchmark_runtime.py` — Main evaluation script. Generates synthetic data, runs SSM fitting, reports runtime and quality metrics.
2. `src/ssm.py` — Core SSM implementation. Contains `estimate_params_ssm()` function with training loop.
3. `src/stats.py` — Statistical computation. Contains `get_H_hat()`, `get_stats()`, `stats_jac`, `solve_tg_exact()`.
4. `src/sample.py` — HMC sampling for synthetic data generation.

## Metric Parsing
- **Runtime:** Parsed from stdout line: `SSM Runtime: X.XX seconds`
- **SSM_R2:** Available in stdout as: `SSM R^2: X.XXXXXX` — guardrail metric
- **SSM_MSE:** Available in stdout as: `SSM MSE: X.XXXXXX` — guardrail metric

## Config Path
- All configuration via function parameters in `benchmark_runtime.py` and `src/ssm.py`
- No external config files

## Reusable Resources
- No `/paper_data` mount — all data is synthetic (generated on-the-fly)
- `/autosota_cache`, `/datasets`, `/models` available but unused

## Safe Modification Targets
- `benchmark_runtime.py` — n_iter, batch_size, lr, l2_reg, l1_reg, optimizer mode
- `src/ssm.py` — optimizer selection (lines 119-126), training loop (lines 135-164), loss function (lines 28-36), JIT decorators (lines 17, 28, 39)
- Pure hyperparameter changes are minimal risk
- Code-level optimizations (JIT static_argnums, early stopping logic) are medium risk

## Risky Files
- `src/stats.py` — Foundational statistics; changes here affect correctness
- `src/sample.py` — HMC sampling; changes affect data distribution
- `benchmark_runtime.py` data generation (`get_random_phi`) — changes affect ground truth

## Dependencies
- JAX 0.4.30 with CUDA 12
- optax 0.2.4
- flax 0.10.4
- blackjax 1.3

## Known Levers
- `n_iter` (default 5000) — paper used 5000
- `batch_size` (default 128) — paper used 128
- `lr` (default 3e-3) — paper used 3e-3
- `mode` (adam/adamw/sgd) — paper used adam
- `l2_reg` (default 0.0) — paper used no regularization
- `l1_reg` (default 0.0)
- `replace` (default True) — batch sampling with replacement

## Guardrails
- SSM_R^2 >= 0.95 (baseline: 0.98)
- SSM_MSE <= 0.015 (baseline: 0.0104)
