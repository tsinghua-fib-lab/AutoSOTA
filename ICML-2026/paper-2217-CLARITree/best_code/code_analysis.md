# CLARITree SOTA Preparation Repair — Code Analysis

## Original Preparation Failure

**Root cause**: The evaluation command `python3 scripts/eval_california_housing.py` performs a sequential 55-point grid search (11 lambda × 5 kappa) across 5 outer 80/20 splits. Each grid job takes ~140-200s (3-fold CV + final fit). Total runtime: ~2.5 hours per outer fold × 5 folds ≈ 12.5 hours. The 120-minute (7260s) evaluation timeout killed the first outer fold at job 41/55.

**Secondary issue**: The preparation log shows the command was run with `timeout 7260s` (121 minutes), which is appropriate per the manifest `eval_timeout_minutes: 120`, but insufficient for the sequential grid sweep.

## Repair Applied

### 1. Parallelized Grid Evaluation (`run_ours_outer.py`)

Modified `scripts/run_ours_outer.py` to support `--workers N` flag (default: 1 for backward compatibility). When `--workers > 1`, uses `concurrent.futures.ProcessPoolExecutor` to evaluate all 55 (λ, κ) combos concurrently.

**Performance improvement**: With 128 CPU cores and 16 workers, each outer fold completes in ~10-15 min (down from ~2.5 hours). All 5 outer folds complete in ~60-75 min, well within the 120-min budget.

**Implementation details**:
- Worker function `_evaluate_combo_worker()` is a module-level function for pickle compatibility
- Each worker re-imports dependencies and loads data independently (avoids pickle overhead for large numpy arrays)
- OMP_NUM_THREADS=1 set in worker subprocesses to avoid thread oversubscription
- Results sorted to match original (lambda, kappa) ordering
- Original sequential code path preserved when --workers=1

### 2. Fast Evaluation Mode (`eval_california_housing_fast.py`)

Created a fast evaluation path that uses fixed hyperparameters (λ=0.001, κ=1e-05, the baseline best) to evaluate algorithmic changes in ~5-10 min. This is used for rapid iteration during optimization. The full grid sweep (`eval_california_housing.py`) is reserved for baseline verification and final candidate evaluation.

### 3. Single-Combo Runner (`run_ours_outer_single.py`)

A lightweight script that evaluates a single (λ, κ) combo on a single outer fold. Used by the fast eval wrapper.

## Corrected Evaluation Command

**In-container command**:
```bash
cd /repo
python3 scripts/eval_california_housing.py --workers 16
```

This produces identical results to the original (same random seeds, same splits, same model code) but completes in ~60-75 min instead of ~12.5 hours.

## Baseline Verification

The existing reproduction results in `results/ours/claritree/linear_regression_tree_depth4_threshold_20/california_housing/` were verified to match the manifest baseline:

| Metric    | Manifest Baseline | Existing Results |
|-----------|-------------------|------------------|
| Test R²   | 0.7499            | 0.749938         |
| Train R²  | 0.7657            | 0.765722         |
| Best λ    | —                 | 0.001            |
| Best κ    | —                 | 1e-05            |
| Leaves    | —                 | 16               |

The baseline matches within numerical precision (paper reports 0.75 ± 0.01).

A quick single-combo test confirmed pipeline integrity:
- outer_0, λ=0.003, κ=0.0001 → val_r2=0.736114, test_r2=0.738141
- Matches job 39 from the timed-out SOTA prep run exactly

## Reusable Resources

- **Data**: California Housing CSV files are in `/repo/data/california_housing/`
- **Pre-computed splits**: 5 outer 80/20 splits in `/repo/data/california_housing/splits/outer_0/` through `outer_4/`
- **Existing results**: Full 55×5 grid results from reproduction already exist in `/repo/results/ours/claritree/`
- **No /paper_data mount needed**: All data is in the repo

## Safe Optimization Targets

### Code (`src/clari_tree.cpp`)
- `fit_coefficients()` (line ~532-567): Leaf coefficient fitting with ridge penalty. Target for ALGO-02 (refinement), ALGO-01 (adaptive regularization).
- `recursive_fit()`: Tree building with lookahead search. Target for ALGO-04 (depth-dependent lambda), ALGO-06 (randomized thresholds), CODE-04 (feature subsampling).
- `build_threshold_pool()` (line ~408): Threshold candidate generation. Target for ALGO-06.

### Hyperparameters (`scripts/run_ours_outer.py`)
- `LAMBDAS` list (line ~15): Sparsity penalty grid. Target for ALGO-05 (two-stage refinement).
- `KAPPAS` list (line ~16): Ridge penalty grid.
- `--n_thresholds` argument: Number of quantile thresholds. Target for PARAM-01.

### Red Lines (Do Not Touch)
- Data files in `/repo/data/`
- Split indices or random seeds
- Metric computation (mse, r2 functions)
- Test set labels or features
- `merge_single_method_outer.py` aggregation logic

## Optimization Strategy

1. **ALGO-02** (P0): Post-training leaf coefficient refinement — highest expected ROI with lowest risk
2. **ALGO-01** (P0): Adaptive per-leaf ridge regularization — addresses train-test gap
3. **ALGO-05** (P0): Two-stage hyperparameter grid refinement — free improvement from better HP selection
4. **ALGO-04** (P1): Depth-dependent lambda — natural regularization gradient
5. **ALGO-06** (P1): Randomized thresholds — decorrelation/implicit regularization
6. **CODE-04** (P1): Feature subsampling — reduces predictor dominance

Each algorithmic change is evaluated first with fast eval (fixed λ,κ at baseline best). If Test R² improves by ≥0.003, the full grid sweep verifies the result and potentially finds even better hyperparameters.
