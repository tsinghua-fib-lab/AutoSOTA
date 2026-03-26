# Optimization Results: SEMPO - Lightweight Foundation Models for Time Series Forecasting

## Summary
- Total iterations: 7 (+ baseline)
- Best `mse`: **0.3409** (baseline: 0.3413, improvement: **-0.12%**)
- Target: ~0.334 (2% improvement) — **TARGET NOT REACHED**
- Best commit: b10dd228 (iter-4: multi-head stacking 336+192+192 for pl=720)

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| mse | 0.3413 | 0.3409 | -0.0004 (-0.12%) |
| mse_96 | 0.279 | 0.279 | 0 (0%) |
| mse_192 | 0.335 | 0.335 | 0 (0%) |
| mse_336 | 0.3579 | 0.3579 | 0 (0%) |
| mse_720 | 0.3934 | 0.3919 | -0.0015 (-0.38%) |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| Multi-head stacking 336+192+192 for pl=720 | 0.3413→0.3409 | Only pl=720 improved |
| Few-shot MoE fine-tuning | 0.3413→0.3413 | Failed/hurt performance |
| Output smoothing window=3 | 0.3413→0.3431 | Hurt, rolled back |
| Multi-head decomposition | - | Implementation failed |

## What Worked
- **Multi-head stacking for long prediction length**: Breaking pl=720 into 336+192+192 heads gave modest improvement on the longest horizon.

## What Didn't Work
- Few-shot MoE fine-tuning on 5% data: Hurt performance
- Output smoothing: Model predictions already smooth
- Multi-head decomposition: Implementation bugs

## Analysis
SEMPO appears to be well-optimized by default. The foundation model architecture has limited headroom for improvement through simple hyperparameter tuning. The marginal improvement suggests:
1. Model is near its capacity ceiling
2. Different approaches (architecture changes, different loss functions) may be needed for larger gains

## Top Remaining Ideas
- Explore different foundation model checkpoints
- Try longer input context windows
- Investigate dataset-specific fine-tuning strategies
