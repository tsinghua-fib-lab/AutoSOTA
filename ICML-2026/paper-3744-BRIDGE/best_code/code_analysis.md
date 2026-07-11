# Code Analysis for Paper 3744 (BRIDGE) SOTA Optimization

## Evaluation Path
- `reproduce_metrics.py` → loads precomputed IRT params → fits linear regression → predicts time buckets → computes metrics
- Output parsing: stdout lines containing "Overall Accuracy:", "Weighted Macro F1:", "Weighted Kappa:"

## Key Files
| File | Role | Safe to Modify |
|------|------|----------------|
| `reproduce_metrics.py` | Evaluation script | Yes - regression, bucket boundaries |
| `fit_irt.py` | IRT model fitting (2PL via py-irt) | Yes - epochs, device, model type |
| `two_param_logistic.py` | 2PL IRT model (Pyro) | Yes - priors, model params |
| `prepare_irt.py` | Build IRT input from benchmark results | Yes - task selection |
| `data/all_a_pyirt.jsonl` | IRT training data (all benchmarks) | No (read-only input) |
| `params/all_a_pyirt.csv` | Precomputed IRT parameters | No (read-only input) |
| `data/combined_human_minutes.jsonl` | Ground-truth human time annotations | No (read-only labels) |

## Pipeline Flow
1. `prepare_irt.py` → `data/swe_a_pyirt.jsonl` (SWE-bench IRT input)
2. `prepare_sparse_pyirt.py` → `data/all_a_pyirt.jsonl` (combined benchmarks)
3. `fit_irt.py` → `params/all_a_pyirt.csv` (2PL IRT params: a=discrimination, b=difficulty)
4. `reproduce_metrics.py` → metrics (loads step 3 params, fits regression, predicts buckets)

## Key Levers
1. **Regression model** (cheapest, no IRT re-fitting needed):
   - Current: OLS linear: log(m) = slope * b + intercept
   - Options: polynomial, spline, robust, weighted, Ridge
   - Fitted on 170 METR tasks (hcast, rebench, swaa)
   - Could include more benchmarks or use different task filtering

2. **IRT model** (requires re-running fit_irt.py):
   - Current: 2PL with hierarchical priors, 1000 epochs
   - Options: more epochs, different priors, MCMC variant
   - The 2PL model is the only one implemented in this repo

3. **Time bucket boundaries**:
   - Current: [0, 15, 60, 240, inf]
   - Could be optimized based on data distribution

4. **Feature augmentation**:
   - Currently only uses b (difficulty) for prediction
   - Could use both a (discrimination) and b, or task metadata

## Risky Files (DO NOT MODIFY)
- Test data labels: `data/combined_human_minutes.jsonl`
- Benchmark results: `data/*_normalized_results.jsonl`
- Time estimates: `data/*_time_estimations_*.jsonl`
- Metric computation: `compute_bucket_metrics()` in reproduce_metrics.py
- The 4 time bucket labels: ["<15 min", "15-60 min", "1-4 hrs", ">4 hrs"]

## Safe Modification Targets
- Regression formula in reproduce_metrics.py (line: `reg_irt = stats.linregress(x_b, y_log_minutes)`)
- Task filtering for regression (line: `metr_fit_df = metr_fit_df[metr_fit_df["task_source"].isin(METR_SOURCES)]`)
- Add new regression variants alongside the existing one
- Time bucket boundaries (BINS constant)
- IRT fitting hyperparameters in fit_irt.py (epochs, seed, priors)

## Config Path
- No external config file for evaluation
- Container: `autosota_repro_paper_3744`
- Repo: `/repo`
