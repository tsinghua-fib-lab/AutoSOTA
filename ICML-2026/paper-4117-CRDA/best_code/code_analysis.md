# Code Analysis - Paper 4117: CRDA (Counterfactual Residual Data Augmentation)

## Evaluation Path
- **Script**: `/repo/eval_energy_efficiency_xgb.py`
- **Parse mode**: `--parse-only` reads most recent `results.csv` in `experiments/EnergyEfficiency/*/`
- **Full mode**: Runs experiment with Config (xgboost, EnergyEfficiency.csv, hyperparam_tune=True, ignore_filter=True, 15 seeds)
- **Key output lines**: `FINAL: baseline_mse=<value>`, `FINAL: crda_mse=<value>`
- **Metrics in results.csv**: mse, aug_mse, delta_mse, p_wilcoxon, should_proceed

## Train/Inference Path
- `Experiment.run()` → `_run_crda()` for each dataset and seed
- Flow per seed: split → train_baseline → add_residuals → filter → tune_aug_params → cv_eval → final_eval
- CRDA pipeline: `_data_augmentation()` → `_intervention()` → `_counterfactuals()` → `_get_combined_aug_training_set()`

## Config Path
- `/repo/src/utils/config.py` - Config class with all experiment parameters
- Defaults: aug_data_size_factor=1.0, max_n_features_to_perturb=5, max_perturb_percent=0.1, aug_data_weight=0.9
- Model defaults in `/repo/src/utils/const.py`

## Metric Parser
- `experiment.run()` returns pd.DataFrame with aggregated metrics
- Results saved to `results.csv` with columns: dataset, metric, mean, std
- Primary metric: `aug_mse` (mean column where metric=="aug_mse")
- Secondary: `mse`, `delta_mse` (percent change)

## Reusable Resources
- Dataset: `/repo/data/EnergyEfficiency.csv` (768 rows, 9 features + 2 targets)
- Pre-downloaded paper data: None mounted
- Cache mounts: `/autosota_cache`, `/datasets`, `/models`

## Risky Files (do not modify)
- `/repo/eval_energy_efficiency_xgb.py` - eval script (but may modify for faster iteration)
- `/repo/src/utils/const.py` - model defaults (can modify for optimization)
- Test data, splits, labels: defined in dataset.py, must not change

## Safe Modification Targets
- `/repo/src/experiment.py` - main experiment logic, augmentation pipeline
- `/repo/src/baseline.py` - model training and tuning
- `/repo/src/causal.py` - causal graph learning
- `/repo/src/filter.py` - feature filtering
- `/repo/src/dataset.py` - dataset handling
- `/repo/src/utils/config.py` - add new config parameters

## Key Observations
1. `ignore_filter=True` is needed because causal filter often returns None on small dataset (n=765)
2. `n_jobs=-1` in RandomizedSearchCV and cross_val_score (patched to n_jobs=8 in reproduction)
3. Residual tiling in `_counterfactuals()`: `np.tile(train_residuals, n)` - limited diversity
4. Fixed perturbation for all samples in `_intervention()` - no per-sample adaptation
5. `aug_data_weight=0.9` applied uniformly to all augmented samples
