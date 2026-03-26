# Optimization Results: TimePFN - Effective Multivariate Time Series Forecasting

## Summary
- Total iterations: 7 (+ baseline)
- Best `mse`: **0.6798** (baseline: 0.7452, improvement: **-8.77%**)
- Target: ~0.73 (2% improvement) — **TARGET REACHED** ✓
- Best commit: d214e047 (iter-7: type3 LR decay from epoch 3, decay=0.8)

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| mse | 0.7452 | 0.6798 | -0.0654 (-8.77%) |
| mae | 0.467 | 0.4325 | -0.0345 (-7.39%) |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| type3 LR schedule (decay=0.8 from epoch 3) | 0.7452→0.6798 | Critical improvement from proper LR scheduling |
| Cosine annealing | 0.6798→0.6966 | Too aggressive, rolled back |
| Ordered data | 0.6811→0.9498 | Much worse, rolled back |
| Dropout reduction | 0.6811→0.7314 | Hurt performance, rolled back |
| EMA weight averaging | 0.6811→0.7731 | Degraded, rolled back |

## What Worked
- **type3 LR schedule with decay=0.8**: Changed from type1 (0.5^epoch halving) to type3 (constant for 2 epochs, then 0.8^epoch decay). This was the dominant improvement.
- **Starting decay from epoch 3 (instead of epoch 4)**: Marginal but consistent improvement.

## What Didn't Work
- Cosine annealing: Final decay too aggressive
- Ordered data: Random shuffling is better for traffic datasets
- Dropout=0.0: Model needs regularization
- EMA weight averaging: Averaging with pretrained weights hurts

## Top Remaining Ideas
- Try decay=0.85 or 0.75
- Experiment with warmup epochs
- Layer-wise learning rate decay
