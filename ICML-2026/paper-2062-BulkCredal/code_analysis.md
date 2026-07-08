# Code Analysis - Paper 2062 Optimization

## Evaluation Path
- `bash /repo/run_california_housing.sh` runs the full pipeline:
  1. `credaldro setup-lv` - creates experiment config in agg_info
  2. `credaldro batch` - runs all 100 replications for 5 algorithms
  3. `credaldro csv` - aggregates results.csv
  4. `credaldro summary` - generates plots
- Metrics parsed from `$RUN_DIR/results.csv` columns: mae, rmse, p98_abs_error, cvar_abs_error
- Error units: raw values are in dollars; divide by 10000 for paper numbers (10^4 dollars)

## Train/Inference Path
- `credal_dro/main.py` run_replication() drives the LV-BAS-CH pipeline for california_housing
- Flow: geographic split → copula+ridge centre fit → DKW bulk calibration → rejection SAA sampling → CVXPY solve
- Geo-block CV (_ca_housing_geo_block_cv_select) selects epsilon within EAST region

## Config Path
- `credal_dro/constants.py` - all tunable constants (gamma, delta, MC samples, etc.)
- `results/california_housing/agg_info/.../experiment.json` - per-algorithm config

## Key Modifiable Targets

### High-ROI Bug Fixes
1. **main.py:1653** - `ridge_alpha=0` should be 0.005 (function default is 0.005, call site overrides to 0)
2. **main.py:2153** - same bug in geo-block CV inner loop

### Parameter Tuning
3. **constants.py:79** - CALIFORNIA_HOUSING_LV_MC_SAMPLES = 5000
4. **constants.py:81** - CALIFORNIA_HOUSING_STANDARDISE_Y = False
5. **constants.py:76** - CALIFORNIA_HOUSING_GAMMA_SET = [0.10]
6. **constants.py:77** - CALIFORNIA_HOUSING_DKW_DELTA = 0.05

### Algorithmic Improvements
7. **dataset.py:934** - GaussianCopula covariance (MLE → LedoitWolf shrinkage)
8. **main.py:1457** - bulk_shape selection (ellipsoid_x_interval_y)
9. **california_housing_lp.py** - problem formulations

## Safe Modification Targets
- constants.py (parameter values only, not structure)
- dataset.py (copula fitting internals)
- main.py (solver options, centre fitting, epsilon grid)
- california_housing_lp.py (new problem formulations)
