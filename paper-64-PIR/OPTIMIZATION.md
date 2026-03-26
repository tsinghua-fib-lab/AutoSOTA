# Paper 64 — PIR

**Full title:** *Improving Time Series Forecasting via Instance-aware Post-hoc Revision*

**Registered metric movement:** -10.6% (MSE: 0.4635 → 0.4142)

---

# Optimization Results: PIR (Instance-aware Post-hoc Revision)

## Summary
- Total iterations: 12 + 1 final eval
- Best `mse`: 0.4142 (baseline: 0.4635, improvement: -10.6%)
- Target: mse ≤ 0.4547 — **ACHIEVED**

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| mse    | 0.4635   | 0.4142 | -10.6% |
| mae    | 0.312    | 0.2794 | -10.5% |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| `refine_epochs` 10→50 | Allows more training time | Model reaches ~35 epochs |
| `lradj='type1'` → `'type3'` | Gentle 0.9x LR decay vs 0.5x | **Critical change** |
| `quality_loss *= 0.1` | Focuses PIR on prediction accuracy | Reduces over-optimization |
| Learnable retrieval temperature | Minor MSE improvement | Better MAE |

---

## Root Cause Analysis

The original PIR code used `lradj='type1'` which HALVES the LR every epoch. With `refine_epochs=10`, the model was spending 80% of its training with a nearly-zero LR.

Switching to `lradj='type3'` (0.9x from epoch 4+) allows the model to continue learning throughout training.

---

## What Worked
1. **Slower LR decay (type3)** — THE key change. Transformed ~10.5% MSE improvement.
2. **Quality loss weighting (0.1x)** — Small but consistent improvement
3. **More training epochs (50)** — Combined with slow LR, model stops at ~35 epochs naturally

---

## Conclusion
The key insight was that the original learning rate schedule caused the model to converge too quickly to a suboptimal solution. Switching to a gentler LR decay schedule allowed the PIR refinement module to fully converge, resulting in a 10.6% MSE improvement.
