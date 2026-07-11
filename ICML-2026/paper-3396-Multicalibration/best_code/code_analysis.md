# Code Analysis — Paper 3396: Multicalibration Yields Better Matchings

## Evaluation Path
1. `main.py --cfg experiments/rubric_repro --redo` runs all experiments
   - Config: `wmcal/configs/experiments/rubric_repro.py`
   - 10 runs: 5 seeds (42-46) x 2 spreads (2, 4)
2. Each experiment:
   - Trains SimpleNet predictor on synthetic data
   - Runs GridBoostCalibrator for max 1024 iterations
   - Logs metrics to `.logs/<config_id>/metrics.jsonl`
3. `eval.py` parses `.logs/*/metrics.jsonl` and outputs METRICS_JSON

## Train/Inference Path
- Predictor: `wmcal/predictors/simple_net.py` — Linear(10->4) + Sigmoid, SGD
- Dataset: `wmcal/data/datasets/synthetic/top_k.py` — synthetic quadratic data
- Calibrator: `wmcal/calibrators/grid_boost.py` — GridBoost (check1 + check2)
- Grid: `wmcal/utils/grid_utils.py` — random sampling with early stop
- Experiment: `wmcal/experiments/base.py` — orchestrates fit sequence

## Config Path
- `wmcal/configs/experiments/rubric_repro.py`
- Key params: eps=0.25, output_dim=4, predictor_size=10000, grid_size=1024, max_iter=1024, batch_size=1024

## Metric Parser
- `eval.py` reads `.logs/*/metrics.jsonl`
- Extracts: Utility_Gap = best_grid - final_ew, Utility_Improvement = final_ew - pre_ew, MSE_Reduction = mse_init - mse_final
- Aggregates mean over all 10 runs
- Outputs METRICS_JSON line

## Safe Modification Targets
- `wmcal/utils/grid_utils.py` — grid generation (max_failures, sampling method)
- `wmcal/calibrators/grid_boost.py` — eps schedule, momentum, update magnitude, early stop patience, replay buffer, check2_prob schedule
- `wmcal/predictors/simple_net.py` — predictor architecture (hidden layers)
- `wmcal/configs/experiments/rubric_repro.py` — experiment parameters

## Risky Files (DO NOT MODIFY)
- `eval.py` — metric computation (red line)
- `wmcal/data/datasets/synthetic/top_k.py` — decision_function (red line)
- `wmcal/experiments/base.py` — experiment orchestration (red line)
- `wmcal/utils/functions.py` — xover utility (shared)

## Evaluation Command (in-container)
```bash
cd /repo && uv run main.py --cfg experiments/rubric_repro --redo && uv run python3 eval.py
```

## Baseline Metrics
- Utility_Improvement: 0.06715
- MSE_Reduction: 0.01502
- Utility_Gap: 0.00796

## Notes
- All data synthetic (quadratic features + Gaussian noise)
- GPU: 2xA100-80GB (only one active during training)
- Container: uv + Python 3.13, pytorch 2.1.0
- Grid early-stop produces ~300-320 unique points (out of 1024 requested)
