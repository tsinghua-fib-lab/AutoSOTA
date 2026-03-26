# Optimization Report: paper-98

## Summary

- **Paper**: X-Mahalanobis - Transformer Feature Mixing for Reliable OOD Detection
- **Total iterations**: 12
- **Best `auroc`**: 0.9833 (baseline: 0.9729, improvement: **+1.07%**)
- **Best `fpr95`**: 0.0790 (baseline: 0.1389, improvement: **-43.1%**)

## Key Results

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| auroc | 0.9729 | 0.9833 | +1.07% |
| fpr95 | 0.1389 | 0.0790 | -43.1% |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| Normalize features AFTER layer mixing | +0.0019 auroc | Most principled change for Mahalanobis embedding |
| Increase cosine classifier scale: 25→35 | +0.0025 auroc | Sharper softmax predictions improve feature learning |
| Increase AdaptFormer bottleneck dim: 4→16 | +0.0060 auroc | Auto-computed dim=4 was too small |

## What Worked

1. **Normalize after mixing (not per-layer before)**: The original code normalized each layer's CLS features individually before the weighted sum. Changing to mix first then normalize creates a more coherent embedding space for Mahalanobis distance computation.

2. **Cosine classifier scale=35**: A higher scale value creates a sharper softmax over class logits during training, which pushes features to be more tightly clustered around class prototypes.

3. **adapter_dim=16**: The auto-computation formula `2^floor(log2(100/24)) = 4` gives a tiny bottleneck. Increasing to 16 allows AdaptFormer to learn richer task-specific features.

## What Didn't Work

- **LedoitWolf covariance**: Slightly hurt performance
- **Higher CLIP similarity weight**: 0.1 weight is already well-tuned
- **CrossEntropy loss**: LA (Logit Adjusted) loss produces better OOD features
- **adapter_dim=32**: Overfitting - 16 is the sweet spot

## Optimization Trajectory

```
Iter 0:  auroc=0.9729 (baseline)
Iter 4:  auroc=0.9748 (normalize after mixing)
Iter 6:  auroc=0.9767 (scale=30)
Iter 7:  auroc=0.9773 (scale=35)
Iter 11: auroc=0.9833 (adapter_dim=16) ← FINAL BEST
```
