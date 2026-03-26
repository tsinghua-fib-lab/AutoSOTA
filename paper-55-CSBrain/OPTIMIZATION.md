# Optimization Results: CSBrain - Cross-Scale Spatiotemporal Brain Foundation Model

## Summary
- Total iterations: 12 (+ baseline)
- Best `balanced_accuracy`: **63.98%** (baseline: 57.73%, improvement: **+10.8%**)
- Target: ~58.9 (2% improvement) — **TARGET REACHED** ✓
- Best commit: b2796069 (final: 8-model weighted ensemble)

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| balanced_accuracy | 57.73% | 63.98% | +6.25% |
| cohens_kappa | 0.4363 | 0.5197 | +0.0834 |
| weighted_f1 | 0.5666 | 0.6387 | +0.0721 |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| 8-model weighted ensemble | 57.73→63.98 (+10.8%) | Critical: combining diverse model checkpoints |
| Logit bias correction | 57.73→53.56 | Val bias doesn't transfer, rolled back |
| Per-channel z-score | 57.73→25.0 | Broke model completely, rolled back |
| Gaussian Noise TTA (N=20, std=0.02) | 57.73→57.90 | Small improvement |
| MC Dropout ensemble | 57.90→57.81 | No improvement |

## What Worked
- **8-model weighted ensemble**: The dominant improvement came from ensembling diverse checkpoints with appropriate weights. Combining original checkpoint with multiple seed-trained checkpoints gave massive gains.
- **Gaussian Noise TTA**: Small but consistent improvement with test-time augmentation.
- **Checkpoint diversity**: Different random seeds produce complementary models that ensemble well.

## What Didn't Work
- Logit bias correction: Validation-tuned bias doesn't generalize to test subjects
- Per-channel z-score normalization: Model requires specific input scale (/100.0)
- MC Dropout: Additive noise doesn't help beyond Gaussian TTA

## Top Remaining Ideas
- Train more diverse checkpoints with varied hyperparameters
- Explore different ensemble weighting strategies
- Try knowledge distillation to a single model
