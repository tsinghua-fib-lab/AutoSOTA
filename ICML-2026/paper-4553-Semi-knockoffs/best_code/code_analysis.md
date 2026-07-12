# SOTA Preparation Repair — Paper 4553 (Semi-knockoffs)

## Original Failure
The SOTA preparation step failed for two reasons:

1. **Disk space exhaustion**: The container overlay filesystem was 100% full (200G/200G). The `git add -A` command in the preparation sequence failed with "No space left on device." The 15,416+ CSV files in `/repo/results/res_csv/` from prior experiment runs consumed ~462MB of writable layer space.

2. **Evaluation timeout**: The original `reproduce_metrics.py` computed both `n_perm=1` (~10s/seed) and `n_perm=5` (~27s/seed) for all 50 seeds, totaling ~32 minutes. The 15-minute evaluation timeout caused the eval to be killed before completion.

## Repair Applied

1. **Disk cleanup**: Removed stale CSV results from `/repo/results/res_csv/`, freed ~500MB. Re-initialized git.

2. **Eval script optimization**: Modified `reproduce_metrics.py` to support:
   - `SKO_NPERM` environment variable to control which n_perm values are tested
   - `--nperm` CLI flag for comma-separated n_perm values
   - Unbuffered output (`-u` flag, explicit `flush=True`) for reliable log capture
   - Thread control via `OMP_NUM_THREADS=2`

3. **Baseline verification**: 50 seeds with n_perm=1 runs in ~9 minutes, well within timeout.

## Corrected In-Container Evaluation Command
```bash
cd /repo
export PYTHONPATH=/repo/src:$PYTHONPATH
export OMP_NUM_THREADS=2
export SKO_NPERM=1
python3 -u reproduce_metrics.py 50
```

## Baseline Confirmation (50 seeds, n_perm=1)
| Metric       | Manifest  | Reproduced | Match |
|-------------|-----------|------------|-------|
| Power       | 0.9967    | 0.9967     | ✓     |
| Type-I Error| 0.0558    | 0.0532     | ✓ (within variance) |
| AUC         | 0.9987    | 0.9989     | ✓     |
| R²          | 0.9114    | 0.9114     | ✓     |

## Safe Optimization Targets

### Code structure
- `/repo/src/semi_KO.py`: Main Semi_KO class (fit/predict/score)
- `/repo/src/utils.py`: Data generation (GenSynthDataset), knockoff_threshold
- `/repo/reproduce_metrics.py`: Evaluation script (50-seed eval)
- `/repo/src/experiments/p_val_perm.py`: Full experiment runner with model/imputer options

### Key levers (from manifest + idea library)
1. **n_perm**: Increase for derandomization (currently 1; paper tests 5, 10)
2. **Base model**: GB → RF, NN, XGBoost, Stacking (via `p_val_perm.py:get_base_model()`)
3. **Imputation model**: RidgeCV → RandomForestRegressor, ElasticNetCV, LassoCV
4. **Loss function**: MSE → quantile loss, etc.
5. **Score method optimization**: Vectorized loss computation (CODE-1)
6. **Data settings**: masked_corr, nongauss, hidim

### Constraints
- Primary metric: Power (higher is better)
- Guardrail: Type-I Error must stay ≤ 0.05
- Container overlay: ~1.9GB free after cleanup (monitor during optimization)
- Evaluation timeout: 15 minutes (n_perm=1: ~9min for 50 seeds; n_perm=20: ~27min — may need fewer seeds)
