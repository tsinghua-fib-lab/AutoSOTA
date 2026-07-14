# Code Analysis for Paper 5295 — ProEval SOTA Optimization

## Evaluation Path
- Entry: `experiment/exp_performance_estimation.py` → `main()`
- Covers all 14 methods, but we focus on BQ-SF (primary metric)

## Key Functions and Lines
1. **SF Feature Preparation**: `prepare_score_features()` — exp_performance_estimation.py:94-117
   - Computes `u = mean(pretrain_matrix, axis=1)` (equal-weight prior)
   - Normalizes `test_x` by centering and dividing by sqrt(n_pretrain-1)
   - Uses only GMM-selected pretrain indices (already correct for CODE-05 audit)

2. **BQ Active Sampling**: `_bq_active_sampling()` — bq.py:231-326
   - Uses noise_variance=0.3 throughout
   - Prior mean `u` is uniformly weighted
   - Initial sampling: filters to "good_indices" (0.2 < u[i] < 0.6), then random choice
   - n_init=0 by default (no warm-start)

3. **GP Posterior**: `_get_posterior()` — bq.py:146-167
   - Linear kernel: K = X @ X^T
   - Uses k_t_inv for efficient rank-1 updates

4. **Acquisition**: `_variance_improvement()` — bq.py:190-203
   - Maximizes integral variance reduction

5. **GMM Selection**: `select_pretrain_models_gmm()` — pretrain_selector.py:177-323
   - Uses covariance_type="diag", reg_covar=1e-4
   - BIC-based cluster count selection (2-10)

6. **Data Loading**: `setup_train_test_split()` — data.py:129-185

## Config Path
- experiment/exp_performance_estimation.py — all argparse params (lines 249-279)
- proeval/sampler/bq.py — BQPriorSampler defaults

## Metric Parser
- stdout: BQ-SF.*MAE = (number)
- CSV: /repo/results/summary_*.csv — Mean MAE column, row "BQ-SF"

## Safe Modification Targets
- proeval/sampler/bq.py: _bq_active_sampling, _get_posterior, _variance_improvement
- experiment/exp_performance_estimation.py: prepare_score_features, main
- proeval/sampler/pretrain_selector.py: select_pretrain_models_gmm
- proeval/sampler/data.py: setup_train_test_split

## Risky Files (do not modify)
- proeval/sampler/baselines.py
- Data files in /repo/data/
EOF
echo "code_analysis.md written"