# Code Analysis — Paper 1029: Subgroup Discovery with the Cox Model

## Evaluation Path
- Script: `/repo/eval_aids_karnof.py`
- Steps: `run_experiments()` → runs all jobs from config → `compute_metrics()` → selects best regions → writes `/repo/results/aids_karnof/metrics.json`
- Config: `/repo/experiments/configs/aids_karnof.yaml`
  - Dataset: `aids` from sksurv (no external download)
  - Methods: `["ddgroup", "base", "random", "c_ind_ddgroup", "pl_ddgroup"]`
  - 10 seeds: [0-9]
  - subgp_cols: [[0]] (age), adjust_cols: [[2]] (karnofsky)
- Metric output: `ddgroup_test_epe`, `ddgroup_test_c_index`, `ddgroup_test_size`, `ddgroup_test_rej10`
- Current baseline: EPE=0.379, C-Index=0.839, Size=0.154, Rej10pct=0.045

## Train/Inference Path
- Data: `load_data("aids", [2], [0], seed)` → age as subgroup variable, karnofsky as adjusted covariate
- Algorithm: `ddgroup_job()` in `src/algs/ddgroup.py` → `core_group()` selects k-NN neighborhood with lowest EPE → `grow_region()` expands region → fits Cox on region
- Core Cox fiting: `fit_cox()` in `src/utils/subgroup.py` → uses unregularized `CoxPHSurvivalAnalysis`
- Core size: `[0.05, 0.1]` (only 2 values, k=46 or k=92 with n=920 train)
- Rejection thresholds: 50 values in [0.0, 0.49]

## Config Path
- `src/config/constants.py`: METHOD_DICT, core_sizes, rejection_thresholds, METHOD_HYPERS
- `experiments/configs/aids_karnof.yaml`: dataset configuration

## Metric Parser
- `eval_aids_karnof.py:compute_metrics()` → reads `selected.pkl` → computes means/SEMs → writes `metrics.json`

## Key Files
| File | Role | Safety |
|------|------|--------|
| `src/utils/subgroup.py` | fit_cox(), epe_metric(), core group metrics | PRIMARY TARGET — safe to modify |
| `src/utils/functions.py` | EPE() computation | Safe to fix numerical stability |
| `src/algs/ddgroup.py` | ddgroup_job(), core_group() | Safe — algorithm logic |
| `src/config/constants.py` | hyperparameter grids | Safe — expandable |
| `eval_aids_karnof.py` | evaluation orchestration | DO NOT MODIFY — contains SIZE_THRESHOLD, metric computation, selection logic |
| `src/utils/metrics.py` | run_eval(), EPE(), c_ind() | DO NOT MODIFY — metric definitions |
| `src/data/load.py` | data loading with train/test split | DO NOT MODIFY — affects evaluation protocol |

## Safe Modification Targets
1. `fit_cox()` — replace with CoxnetSurvivalAnalysis (elastic net)
2. `EPE()` — replace log(sigmoid) with -logaddexp
3. `core_sizes` — expand grid
4. `ddgroup_job()` — add alternative core selection logic

## Risky Files (DO NOT MODIFY)
- `eval_aids_karnof.py` — selection/best-region/metric computation
- `src/utils/metrics.py` — metric definitions
- `src/data/load.py` — train/test split
- `experiments/configs/aids_karnof.yaml` — evaluation protocol

## Available SKSURV Extensions
- `CoxnetSurvivalAnalysis` available in sksurv 0.25.0 (confirmed)
