# Optimization Results: Channel Normalization for Time Series Channel Identification

## Summary
- Total iterations: 12 (+ baseline)
- Best `mse`: **0.1615** (baseline: 0.1904, improvement: **-15.2%**)
- Target: ~0.186 (2% improvement) — **TARGET REACHED** ✓
- Best commit: 2f32945 (iter-12: seq_len=336 with cosine LR + dropout)

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| mse | 0.1904 | 0.1615 | -0.0289 (-15.2%) |
| mae | 0.2766 | 0.2519 | -0.0247 (-8.9%) |
| mse_96 | 0.1639 | 0.1291 | -0.0348 (-21.2%) |
| mse_192 | 0.1743 | 0.1490 | -0.0253 (-14.5%) |
| mse_336 | 0.1912 | 0.1671 | -0.0241 (-12.6%) |
| mse_720 | 0.2322 | 0.2007 | -0.0315 (-13.6%) |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| seq_len 96→336 | 0.1761→0.1615 (+7.7%) | Longer lookback captures more patterns |
| Cosine LR annealing | 0.1813→0.1790 | Better than exponential decay |
| Dropout in temporal block | 0.1790→0.1781 | Marginal improvement |
| Second MLP residual block | 0.1790→0.1761 | More capacity helped |
| GELU activation | 0.1790→0.1790 | No significant change |

## What Worked
- **Extended input sequence length (96→336)**: The most impactful change. Longer context (14 days vs 4 days) captures weekly seasonality in electricity data.
- **Cosine LR annealing**: Better final convergence than exponential decay.
- **Dropout regularization**: 0.1 dropout in temporal block.
- **Increased model capacity**: Second MLP residual block.

## What Didn't Work
- Larger batch_size (64): Hurt performance
- Weight decay regularization: Hurt significantly
- d_model reduction (2048→1024): No improvement, kept 2048

## Top Remaining Ideas
- Try seq_len=504 or longer
- Experiment with different dropout rates
- Explore other activation functions
