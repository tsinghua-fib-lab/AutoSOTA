# Code Analysis — DistMatch (Paper 2010)

## Evaluation Path

- **Entry point**: `/repo/code/main.py` → `my_app()` hydra app
- **Config**: `configuration/default_config.yaml` with overrides
- **Key overrides in eval command**:
  - `config/model_fc=darts_forest` → RF point predictor with 50 lags
  - `config/model_uc=dist_match` → DistMatch uncertainty quantifier
  - `config/task=default_alpha01` → α=0.1
  - `config/dataset=enbPI_solar_atlanta` → Solar Atlanta dataset (8760 rows)
  - `config/evaluation=default.yaml`
  - `config.model_uc.match_method=ks_stat`
  - `config.model_uc.match_threshold=0.005` (NOTE: config default is 0.1!)

## Metric Extraction

Metrics are computed in `code/main_utils.py:353-368` and logged at line 410:
```
LOGGER.info(f"Evaluation metrics (alpha {alpha}: {eval_metrics}")
```
Format: Python dict with keys `mean_coverage`, `mean_coverage_eps`, `mean_pi_width`, `mean_pi_sd`, `winkler_score`, `winkler_score_norm`.

## Config Path

- **DistMatch config**: `/repo/configuration/config/model_uc/dist_match.yaml`
- **Dataset config**: `/repo/configuration/config/dataset/enbPI_solar_atlanta.yaml`
- **Task config**: `/repo/configuration/config/task/default_alpha01.yaml` (α=[0.1])
- **Eval config**: `/repo/configuration/config/evaluation/default.yaml`

## Key Source Files

### DistMatch Core (`code/models/uncertainty/dist_match/dist_match.py`)
- `__init__`: reads kwargs for match_threshold, past_window_len, qrf_param, match_method, etc.
- `calibrate_individual` → `_train_qrf_from_inputs`: calibration pipeline
- `_train_qrf`: builds sliding windows, computes match_mask, fits DistMatchQRF
- `_predict_step`: incremental prediction with optional QRF retraining
- `_match`: threshold comparison using configured matcher
- `_preprocess_inputs`: supports `normal`, `delta`, `residual` modes

### Tree/QRF (`code/models/uncertainty/dist_match/tree.py`)
- `DistMatchTree`: single tree with greedy search for distribution shift splits
- `DistMatchQRF`: ensemble of DistMatchTree with bagging
- `_compute_qrf_quantiles`: leaf-level quantile regression using QRF/linear/XGBoost
- `predict_from_trees`: mean aggregation over tree predictions (line 595)

### Matchers (`code/models/uncertainty/dist_match/utils.py`)
- `match_ks_stat`, `match_ks_p_val`, `match_mi`, `match_kl`, `match_wd`, `match_rand`

## Known Levers (from manifest)

| Parameter | Default | Description |
|-----------|---------|-------------|
| match_threshold (γ) | 0.005 (override) / 0.1 (default) | KS-based tree split threshold |
| past_window_len (w) | 100 | Sliding window size for residual patches |
| qrf_param.n_trees (B) | 10 | Bootstrap trees in QRF ensemble |
| qrf_param.bagging_ratio (θ) | 0.9 | Bootstrap sampling ratio |
| seed | 20 | Controls bootstrap randomness |
| qrf_upd_steps | null | Online retraining interval |
| input_mode | normal | normal/delta/residual preprocessing |
| match_method | ks_stat | ks_stat/ks/mi/wd/kl/rand |

## Reusable Resources

- **Dataset**: Solar_Atl_data_aligned.csv in `/repo/data/enbPI/` (8760 rows)
- **Models**: `models_save/fc/` and `models_save/uc/` directories
- **Cache**: `/autosota_cache`, `/datasets`, `/models`

## Safe Modification Targets

1. `dist_match.py:_predict_step` (lines 214-249) — residual processing, safe
2. `dist_match.py:_train_qrf` (lines 186-209) — calibration logic, safe
3. `dist_match.py:__init__` (lines 31-68) — parameter reading, safe
4. `tree.py:_compute_qrf_quantiles` (line 255) — leaf quantile estimation, safe
5. `tree.py:predict_from_trees` (line 595) — aggregation method, safe
6. `dist_match.yaml` — config defaults, safe
7. `utils.py` — matcher implementations, safe

## Risky Files (do not modify)

- `code/main_utils.py:353-368` — metric computation
- `code/main_utils.py:410` — metric logging format
- `code/loader/` — data loading and splitting
- `code/enbPI/` — base enbPI implementation
- `data/` — datasets
- `code/main.py` — evaluation entry point

## Red-Line Constraints

1. ✅ Eval command unchanged
2. ✅ Metric computation untouched
3. ✅ Test data/splits untouched
4. ✅ No hard-coded outputs
5. ✅ Optimization objective: minimize Win. while Cov. ≥ 0.90
6. ✅ All guardrail metrics reported

## Evaluation Time

~6-8 minutes per single-seed evaluation on A100 GPU.
