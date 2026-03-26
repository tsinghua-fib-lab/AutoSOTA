# Paper 46 — NonStatTS

**Full title:** *Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation*

**Registered metric movement:** -2.27% (MSE: 0.4618 → 0.4513)

---

# Optimization Results: Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation

## Summary
- Total iterations: 16
- Best `test_mse`: **0.4513** (baseline: 0.4618, improvement: **-2.27%**)
- Target achieved: 0.4513 ≤ 0.4526 ✅
- Best commit: 9180b77953

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| test_mse (with TTA) | 0.4618 | 0.4513 | -0.0105 (-2.27%) |
| test_mae (with TTA) | 0.4611 | 0.4520 | -0.0091 (-1.97%) |
| test_mse (no TTA) | 0.4695 | ~0.4550 | ~-0.0145 |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| SOLVER.MAX_EPOCH 30→50 (config.py) | -1.76% MSE | Most impactful change: better base model → better TTA starting point |
| All-batch partial GT loss (IDEA-010) | -0.39% MSE | Use all batch samples instead of only pred[0][:period] |
| Sample-specific partial GT lengths (IDEA-024) | -0.04% MSE | Each sample uses only its available GT steps |
| Mixed MAE+MSE adaptation loss (IDEA-009) | -0.09% MSE | 0.5*MSE + 0.5*L1 loss for both adaptation steps |
| Fix graph reuse for multi-step (bug fix) | stability | Prevents backward() through freed graph when STEPS>1 |

---

## What Worked

1. **More training epochs (50 vs 30)**: By far the biggest individual improvement (-1.76%). The DLinear model converges more thoroughly with 50 epochs, giving TTA a much better starting point. The cosine LR decay schedule benefits from longer training.

2. **All-batch partial GT loss**: The original code used only `pred[0][:period]` for the PAAS adaptation signal — discarding all batch samples except the first. Using `pred[:, :period, :]` provides a proper gradient estimate over all available samples.

3. **Sample-specific partial GT lengths**: Each sample `i` in a batch of `period+1` samples has exactly `period-i` available future timesteps as ground truth. Using sample-specific lengths provides cleaner gradient signal.

4. **Mixed MAE+MSE loss**: Combining 0.5*MSE + 0.5*L1 in adaptation losses provides more balanced gradients — less dominated by large outliers.

## What Didn't Work

1. **STEPS=2 (more gradient steps per adaptation)**: Overfits to local windows. STEPS=1 is optimal. (+0.26%)
2. **Multi-frequency period detection (top-3 FFT)**: Single dominant frequency works better. (+0.17%)
3. **EMA smoothing of GCM params (momentum=0.9)**: Too aggressive, prevents adaptation entirely. (+1.95%)
4. **Higher TTA LR=0.003**: Destabilizes calibration. LR=0.001 is optimal. (+0.30%)
5. **Val-tail warm-start**: Biases GCM toward validation distribution, not helpful for test. (+0.35%)
6. **Bias correction from partial GT**: Mean error correction too noisy, overcorrects. (+8.9%)
7. **Gating-only adaptation**: Insufficient capacity — weight matrix essential. (+2.17%)
8. **Variable importance weighted loss**: Negligible effect (+0.02%)
9. **TTA weight decay=0.0001**: Restricts adaptation capacity (+0.28%)

## Top Remaining Ideas (for future runs)

1. **Try SOLVER.MAX_EPOCH=100 or 80**: Even more training might help further
2. **Tune SOLVER.BASE_LR for training**: Try 1e-4 vs 5e-5 for DLinear training
3. **AdamW for training**: Better optimizer regularization during training
4. **Period multiplier TAFAS.PERIOD_N=2**: Larger adaptation batches
5. **Multiple adaptation loss weights**: Try 0.3*MSE + 0.7*L1 instead of 50/50
