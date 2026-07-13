# Code Analysis — Paper 4980 SOTA Optimization

## Evaluation Path
- `kernel_experiments.py` → `main()` → surrogate mode → CV GP fitting → results.csv
- Metrics: `surrogate_normalized_mse_mean` (NMSE), `surrogate_r2_mean` (R²)
- NMSE = mean over CV folds of normalized MSE (normalized by y_train mean/std)
- R² = mean over CV folds of sklearn r2_score on normalized targets

## Key Files
| File | Location | Role | Editable |
|------|----------|------|----------|
| `kernel_experiments.py` | `/repo` | Main eval entry point | **Yes** |
| `utils.py` | `/repo` | Pre-sample generation, loading | **Yes** |
| `damg_kernels.py` | `/opt/conda/lib/python3.11/site-packages/bayesian_optimization/examples/damg_nas/` | Kernel definitions | No (read-only site-packages) |
| `damg_targets.py` | same path | Target/search space definitions | No |
| `damg_repo.py` | same path | Repository/search space construction | No |

## Evaluation Flow
1. Parse args → resolve targets → generate pre-samples via `lazy_dpp_sample_optimized` (non-deterministic!)
2. Evaluate each architecture via objective function (training ODE nets)
3. Apply y-transform (log1p default)
4. 5-fold CV with GP (GaussianProcessRegressor, normalize_y=True, alpha=1e-3, n_restarts=10)
5. Compute NMSE, R², NLPD per fold → mean/std reported

## Reproducibility Issue
- `generate_pre_samples()` uses DPP sampling with no seed control → each run samples different architectures
- 4 runs with identical params: NMSE ranges 0.265–0.384, R² ranges 0.702–0.771
- Fix: add `--pre-sample-file` flag to load specific .pt file instead of generating fresh

## Metric Parser
- Parse CSV: filter rows where kernel contains damg, extract `surrogate_normalized_mse_mean` and `surrogate_r2_mean`
- Column mapping: col[7]=NMSE_mean, col[8]=NMSE_std, col[9]=R2_mean, col[10]=R2_std

## Safe Modification Targets
1. `kernel_experiments.py` → add `--pre-sample-file`, modify GP params, add new kernels inline
2. `build_default_kernels()` at line 326 → add/remove kernels
3. GP fitting at lines 722-728 → change alpha, normalize_y, n_restarts
4. Target transform at line 618 → change y_transform
5. CV setup at line 712 → change folds, shuffle

## Risky Modifications
- Changing metric computation (NMSE, R² formulas)
- Changing test data or dataset splits
- Modifying site-packages damg_kernels.py (requires root in container)
- Hard-coding predictions

## Reusable Resources
- Best baseline pre-sample: `results/kernel_experiments/20260712_131909_n100_targets1_seeds1/presample_target_len_3_refined_1_100.pt`
- Container has GPUs 2,3 available
- `/autosota_cache`, `/datasets`, `/models` mounted
