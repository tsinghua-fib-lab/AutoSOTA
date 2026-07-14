# Code Analysis — Paper 5883 SOTA Optimization

## Evaluation Path

- **eval_command:** `python3 power_directional.py --R 1000 --B_perm 500 --seed 5086`
- **Timeout:** 20 minutes
- **Output parsing:** Parse stdout for `Power (Permutation, alpha=0.05): X.XXXX` and `Power (Asymptotic, alpha=0.05): X.XXXX`

## Key Files

| File | Role | Safe to Modify? |
|------|------|-----------------|
| `power_directional.py` | Evaluation script: vMF simulation, permutation/asymptotic inference | Yes — add transforms, parallelization, multi-p |
| `optimized_estimators.py` | Core test statistic T_p_mn and SPH kernel | Yes — JIT compilation, better variance estimator |
| `spherical_simulations.py` | Original paper simulations (not used for eval) | No — reference only |
| `compositional_simulations.py` | Original paper simulations (not used for eval) | No — reference only |

## Config Path

- Truncation parameter `p` is a CLI arg to `power_directional.py` (default 2)
- Sample sizes `m`, `n` are CLI args (default 25 each)
- vMF concentrations `kappa_X`, `kappa_Y` are CLI args (default 5.0, 8.75)

## Metric Parser

Metrics are parsed from stdout lines:
```
Power (Permutation, alpha=0.05):  X.XXXX +/- Y.YYYY
Power (Asymptotic,  alpha=0.05):  X.XXXX +/- Y.YYYY
```

## Baseline

- Power (Permutation): 0.203
- Power (Asymptotic): 0.251
- Commit: 835c0e5 (iter-0 baseline)

## Reusable Resources

- No external datasets — all vMF simulated
- Cache mounts: `/autosota_cache`, `/datasets`, `/models`

## Risky Files (Do Not Modify)

- `/tools/record_score.sh` — scoring infrastructure
- `spherical_simulations.py` — paper reference code
- `compositional_simulations.py` — paper reference code

## Safe Modification Targets

1. `optimized_estimators.py::SphericalTestConfig.compute_reproducing_kernel()` — replace scipy eval_gegenbauer with JIT recurrence
2. `optimized_estimators.py::OptimizedTestStatistic` — add jackknife variance, batched permutation stats
3. `power_directional.py::run_power_simulation()` — add parallel permutation loop, multi-p aggregation, dual-transform ensemble

## Dependencies Installed for SOTA

- `numba` 0.66.0 — JIT compilation for Gegenbauer recurrence
- `joblib` 1.5.3 — parallel permutation loop
- `scipy` 1.15.3 — already present
- `numpy` 1.26.2 — already present

## Red Line Constraints (Verified)

- Evaluation command: unchanged from manifest
- Metric computation: parsed from stdout (unchanged)
- Test data: simulated vMF with seed 5086 (unchanged)
- No hard-coded outputs
- Guardrail metric (Power (Permutation)) must stay above 0.193 (5% regression tolerance)
