# Code Analysis for Paper 4750 - SOTA Optimization

## Repository Structure

- `DLinear/run_predictability_test.py` — Evaluation entry point. Uses `--is_training 0` for eval-only.
- `DLinear/run_longExp.py` — Training entry point (run_linear_longExp.sh calls this).
- `DLinear/exp/exp_main.py` — Core: training loop, evaluate_predictability(), coherence computation.
- `DLinear/models/DLinear.py` — DLinear architecture (series decomposition + linear layers). Has `use_norm=False`.
- `DLinear/utils/tools.py` — EarlyStopping, adjust_learning_rate (type1: halving every epoch).
- `DLinear/data_provider/` — Data loading from CSV.

## Evaluation Pipeline

1. `run_predictability_test.py` → `Exp_Main.evaluate_predictability()`
2. Loads checkpoint from `checkpoints/<setting>/checkpoint.pth`
3. Iterates test_loader batches
4. For each batch: extracts x_tail (history tail) and y_head (future head) of Np timesteps
5. Computes Welch coherence → NMSE_lb via GPU-batched `coherence_nmse_lb_gpu_batched()`
6. Computes MSE_lb = NMSE_lb * Var(y) + (mean(y) - mean(x))^2
7. Computes MSE_model = (y_true - y_pred)^2 per channel per sample
8. Computes Pearson R between MSE_lb and MSE_model across all (sample, channel) pairs
9. Outputs: `[OK] overall r -> X.XXXXXX` in stdout, `correlation_overall.txt` file

## Key Parameters

- `welch_win_frac`: 0.25 (nperseg fraction for Welch), line ~713
- `welch_overlap`: 0.5 (segment overlap), line ~726
- `alpha_boundary`: 1.0 (Np multiplier), line ~712
- `tau`: 1e-2 (variance threshold for NMSE), line 740
- `train_epochs`: 10, `batch_size`: 16, `learning_rate`: 1e-3 (from manifest eval_command = the original training config)
- `use_amp`: False (default)

## Safe Modification Targets

1. `DLinear/exp/exp_main.py` — Welch window parameters (lines 712-713), tau threshold (line 740)
2. `DLinear/run_predictability_test.py` — argument defaults and grids
3. `DLinear/models/DLinear.py` — `use_norm` flag, RevIN implementation
4. `DLinear/exp/exp_main.py` training loop — LR schedule, epochs, AMP

## Risky Files (do not modify)

- `DLinear/utils/metrics.py` — metric computation
- Test data / dataset splits
- `correlation_overall.txt` parsing (fixed format)

## Baseline Metrics

- R: 0.880047
- MSE_model: 0.194631
- MAE_model: 0.277396
- NMSE_model: 0.369978
- P_lin: 0.751099

## Red-Line Constraints

- Do not modify metric definitions, test data, labels, splits, or scoring scripts
- Do not hard-code predictions or metric values
- Use `/tools/record_score.sh` for all score records
