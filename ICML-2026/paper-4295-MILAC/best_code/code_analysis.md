# SOTA Preparation Repair — Paper 4295 (MILCCI)

## Preparation Failure

The evaluation script `run_full_v2.py` crashed with:
```
UnboundLocalError: local variable 'N' referenced before assignment
```
at line 254 when constructing the results dict.

## Root Cause

`N` (number of pages) was only defined in the `else` branch (line 136: `N = len(PAGES)`) that handles fresh data download. When loading cached data from `/datasets/milcci_wikipedia/wiki_full_v2.npz`, the `if os.path.exists(CACHE_FULL)` branch was taken but `N` was never set.

## Fix

Added `N = Y.shape[0]` in the cached-data branch (after loading Y from .npz), before the print statements.

## Corrected Evaluation Command

```bash
cd /repo && python3 run_full_v2.py
```

## Baseline Verification

| Metric  | Manifest | Repaired |
|---------|----------|----------|
| R²      | 0.8315   | 0.8315   |
| Runtime | 86.65s   | 77.31s   |

Runtime variance is normal CPU noise. Results saved to `/datasets/milcci_wikipedia/full_v2_results.json`.

## Safe Optimization Targets

- `core.py`: Main MILCCI solver — component initialization, outer loop, decorrelation penalty, normalization
- `solvers.py`: Regularized least squares solver
- `phi_inference.py`: Phi inference modes (LS, dynamic_prior)
- `run_full_v2.py`: Experiment script — hyperparameters, preprocessing

All changes are hyperparameter/algorithm modifications. No evaluation protocol changes.
