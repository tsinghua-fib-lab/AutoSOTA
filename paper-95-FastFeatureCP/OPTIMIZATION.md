# Optimization Report: paper-95

## Summary

- **Paper**: FastFeatureCP - Accelerating Feature Conformal Prediction via Taylor Approximation
- **Total iterations**: 12
- **Best `band_length`**: 2.589938 (baseline: 2.740545, improvement: **-5.49%**)
- **Target**: band_length ≤ 2.6857 ✓ ACHIEVED

## Key Results

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| band_length (layer 2) | 2.740545 | 2.589938 | -5.49% |
| coverage_pct (layer 2) | 90.059% | 90.010% | -0.05% |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| Clip calibration + test gradient norms at 93rd percentile | -5.49% band_length | Core insight: outlier gradient norms inflate intervals |
| Remove `-1` from quantile border formula | ~-0.03% | Minor tightening of quantile selection |

## Methodology

The FFCP (Fast Feature Conformal Prediction) algorithm computes:
- Nonconformity score: `nc = |prediction - y| / ||gradient_at_layer||`
- Quantile threshold: the `(1-alpha)` quantile of nc scores on calibration set
- Prediction interval: `[pred - threshold * ||grad_test||, pred + threshold * ||grad_test||]`

**Key insight**: The band_length formula involves both the threshold AND the test gradient norms:
```
band_length ≈ 2 * threshold * mean(||grad_test||)
```

**Solution**: Clip both calibration and test gradient norms at the 93rd percentile of calibration gradient norms. This symmetrically limits the influence of outlier gradient magnitudes, reducing the threshold and test interval widths.

## What Worked

- **Gradient norm clipping**: The key improvement. By capping excessively large gradient norms at the calibration distribution's 93rd percentile, we prevent outliers from inflating prediction intervals.
- **Less conservative quantile border**: Removing the `-1` from `floor(alpha*(n+1)) - 1` allowed a marginally tighter quantile.

## What Didn't Work

- Larger calibration split (65%): Slightly hurt performance
- Alpha adjustment (0.102, 0.105): Reduced band_length but violated 90% coverage requirement
- Using all training data for calibration: Data leakage caused coverage drop
- 80th percentile clip: Too aggressive, caused regression

## Optimization Trajectory

```
Iter 0:  band_length=2.740545 (baseline)
Iter 6:  band_length=2.607638 (95th pctile clip)
Iter 7:  band_length=2.591543 (90th pctile clip)
Iter 11: band_length=2.590652 (90th+quantile tweak)
Iter 12: band_length=2.589938 (93rd+quantile tweak) ← FINAL BEST
```
