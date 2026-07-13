# Code Analysis — Paper 5412: The Cost of Learning Under Multiple Change Points

## Evaluation Path
- **Main eval script**: `/repo/eval_nab_regret.py`
- **Imports from**: `/repo/icml/atc_scaling_scenarios_dense_adver.py`
- **Data**: `/repo/NAB/data/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv`
- **Metric**: `ATC_cumulative_regret = sum((hatmu - mu_t)^2)` over T=4032 timesteps
- **Parsed from stdout**: `ATC_cumulative_regret=<float>`

## Key Functions (in atc_scaling_scenarios_dense_adver.py)
- `atc_run()` (line 455): Main ATC loop with alarm detection and reset
- `gamma_paper()` (line 313): Adaptive threshold formula
- `scan_statistic_max()` (line 423): Max scan statistic over candidate splits
- `alpha_r_from_restart()` (line 308): Per-restart alpha schedule
- `candidate_splits_all()` (line 362): Exact scan (all k in [r+1, t-1])
- `sliding_run()` (line 545): Sliding window baseline
- `discount_run()` (line 573): Discounted mean baseline
- `regret_squared()` (line 599): Cumulative squared error metric
- `variance_bias_terms()` (line 624): Regret decomposition (diagnostic)

## Config / Tunable Parameters
- `sigma` (line 24 of eval): Default 1.0 — noise std proxy / sub-Gaussian parameter
- `alpha` (line 25 of eval): Default 0.05 — detection sensitivity
- Scan strategy: `candidate_splits_all` (exact) — can switch to `candidate_splits_geom_ends` or `candidate_splits_uniform`
- Threshold type: `gamma_paper` (time-varying) vs `gamma_constant_factory` (fixed)

## Baseline Metrics
- ATC_cumulative_regret: 9061.12
- SlidingWindow_cumulative_regret: 64344.86
- DiscountedMean_cumulative_regret: 157939.87
- ATC alarms: [183, 380, 421, 593, 1522, 1680, 1849, 2897, 3566, 3568, 3576]
- Ground truth CPs: [377, 420, 592, 3575]

## Safe Modification Targets
1. `/repo/icml/atc_scaling_scenarios_dense_adver.py` — `atc_run()` (cooldown, hybrid reset, trimmed mean, shrinkage)
2. `/repo/icml/atc_scaling_scenarios_dense_adver.py` — `gamma_paper()` (online variance-aware threshold)
3. `/repo/icml/atc_scaling_scenarios_dense_adver.py` — `scan_statistic_max()` (min segment length constraint)
4. `/repo/eval_nab_regret.py` — sigma estimation from segments

## Risky Files (do not modify)
- NAB data files
- Metric computation in `regret_squared()`
- `variance_bias_terms()` (diagnostic only, not eval)

## Reusable Resources
- None — CPU-only algorithm, no model weights or large datasets
- NAB CSV is small (~33KB), included in repo

## Manifest Contract Recovery
- `eval_command`: `python3 eval_nab_regret.py` — runs directly inside container, no Docker wrapper needed
- Verified baseline matches: 9061.12
