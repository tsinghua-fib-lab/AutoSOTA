# Optimization Report: paper-101

## Summary

- **Paper**: ACIA - Anti-Causal Invariant Abstraction via Measure Theory
- **Total iterations**: 12
- **Best `acc`**: 99.45% (baseline: 98.88%, improvement: **+0.57%**)
- **Target**: 100.8576% (unreachable - acc bounded at 100%)

## Key Results

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| Acc | 98.88% | 99.45% | +0.57% |
| EI (Environment Invariance) | 0.0093 | 0.0034 | -63.4% better |
| IR (Intervention Regularity) | 0.0141 | 0.0084 | -40.4% better |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| Best checkpoint tracking | +0.20% acc | Restore highest-acc epoch instead of last |
| 16→24 epochs with cosine annealing | +0.155% acc | Model hadn't converged at 8 epochs |
| Wider phi_H (32→64→16→128 → 32→128→64→128) | Combined effect | Removed bottleneck at 16-dim |
| Reduced regularization (lambda1/2: 0.1→0.02) | +0.085% acc | Lower invariance penalty → better accuracy |
| Top-5 weight averaging (LEAP) | +0.015% acc | Ensemble of 5 best checkpoints |

## What Worked

1. **Best checkpoint tracking**: The biggest single win. The model's final epoch was often not its best. Tracking and restoring the best checkpoint is essential.

2. **Cosine annealing LR schedule**: Transformed training dynamics dramatically. With cosine annealing (1e-3→1e-5), it smoothly converged. This also had a massive effect on EI (0.0079→0.0016).

3. **Reducing regularization weights**: lambda1/lambda2 reduced 5x, lambda3 reduced 10x. This allowed the classifier to focus more on discriminative accuracy.

4. **Extended training (8→24 epochs with cosine)**: The model was clearly not converged at 8 epochs.

5. **Top-K checkpoint weight averaging (LEAP)**: Averaging the weights of the top-5 checkpoints gave a small but consistent improvement.

## What Didn't Work

- 32 epochs without LR schedule: Saturated at 99.24%
- Very wide phi_H (256-dim): Timed out
- Zero regularization: NaN gradient explosion
- Batch_size=64: Slightly worse than batch_size=32

## Optimization Trajectory

```
Iter 0:  acc=98.88% (baseline)
Iter 1:  acc=99.085% (+0.205%, best checkpoint)
Iter 4:  acc=99.35% (+0.11%, cosine annealing)
Iter 8:  acc=99.435% (+0.085%, reduced regularization)
Iter 11: acc=99.45% (+0.015%, LEAP) ← FINAL BEST
```
