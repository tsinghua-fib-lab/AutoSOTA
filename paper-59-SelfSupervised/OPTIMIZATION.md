# Paper 59 — SelfSupervised

**Full title:** *Not All Data are Good Labels: On the Self-supervised Learning of Time Series*

**Registered metric movement:** -0.61% (avg_mse: 0.4592 → 0.4562)

---

# Optimization Results: Self-supervised Time Series Learning

## Summary
- Total iterations: 5
- Best `avg_mse`: **0.4562** (baseline: 0.4592, improvement: -0.65%)
- Target: 0.4499 (2% improvement) — **Not reached** (gap: +1.4%)

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| avg_mse | 0.4592 | 0.4562 | -0.65% |
| avg_mae | 0.4459 | 0.4427 | -0.72% |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| SNR weighting (if_snr=True) | +1.3% regression | Helps short horizons but hurts long horizons |
| **Increase ensemble size (num_models=16)** | -0.6% | New best |
| **CosineAnnealingLR** | -0.04% | Marginally improves over iter 2 |

---

## What Worked
1. **Increased ensemble size**: num_models=16, num_series=16 improved avg_mse by 0.6%
2. **CosineAnnealingLR**: Slightly improves over constant LR schedule

## What Didn't Work
- **SNR weighting**: Helps short horizons but hurts long horizons, net negative effect
- **Single-model approaches**: Ensemble is critical for this method

---

## Conclusion
The method benefits from larger ensembles and proper learning rate scheduling. However, reaching the 2% improvement target requires more substantial changes to the model architecture or training procedure.
