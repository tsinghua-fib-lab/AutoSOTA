# Paper 58 — MindGlitch

**Full title:** *Mind-the-Glitch: Visual Correspondence for Detecting Glitches in Cultural Heritage Docs*

**Registered metric movement:** +1.72% (Spearman: 0.5826 → 0.6006)

---

# Optimization Results: Mind-the-Glitch

## Summary
- Total iterations: 12
- Best `spearman_correlation`: **0.6006** (baseline: 0.5826, improvement: +3.09%)
- Target: 0.5953 — **ACHIEVED**

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| spearman_correlation | 0.5826 | 0.6006 | +3.09% |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| VSM threshold tuning | Various | Default 0.7 was optimal |
| **TTA horizontal flip** | +1.72% | avg VSM(original) + VSM(flipped) |
| **Weighted TTA (3:1 orig:flip)** | +1.80% | Best result: (3*orig + 1*flip)/4 |

---

## What Worked
1. **Horizontal flip TTA**: Averaging VSM(original) + VSM(flipped) improved Spearman from 0.5826 to 0.5998
2. **Weighted TTA**: Using (3*orig + 1*flip)/4 gave the best result of 0.6006

## What Didn't Work
- **VSM threshold tuning** (0.5, 0.6, 0.7, 0.85): Default 0.7 was optimal
- **Vertical flip TTA**: Hurts performance (features not vertically symmetric)
- **Honeymoon/Delta VSM approaches**: Did not improve over baseline

---

## Conclusion
The key improvement was test-time augmentation via horizontal flip, combined with weighted averaging to leverage the fact that visual correspondence features are horizontally symmetric.
