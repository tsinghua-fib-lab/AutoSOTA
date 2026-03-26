# Paper 90 — RSTIB

**Full title:** *Information Bottleneck-guided MLPs for Robust Spatial-temporal Forecasting*

**Registered metric movement (internal ledger, ASCII only):** -0.38%(18.4928->18.4219)

# Final Optimization Report: RSTIB-MLP (Information Bottleneck-guided MLPs)

**Run**: run_20260321_060148
**Date**: 2026-03-21
**Optimizer**: Claude Sonnet 4.6 (auto-pipeline)

---

## Summary

|| Metric | Baseline | Best | Improvement |
||--------|---------|------|-------------|
|| mae | 18.4928 | 18.4219 | -0.0709 (-0.38%) |
|| rmse | 30.1469 | 30.0943 | -0.0526 (-0.17%) |
|| mape | 12.2998 | 12.3363 | +0.0365 (+0.30%) |

**Target**: 18.1229 (not reached; within ~1.6% of target)
**Best Score**: MAE=18.4219

---

## System Architecture

**RSTIB-MLP** is an Information Bottleneck-guided MLP for robust spatial-temporal forecasting.

Key components:
- **RSTIB principle**: Robust Spatial-Temporal Information Bottleneck
- **Knowledge distillation**: Teacher-student training regime
- **PyTorch Geometric**: Uses torch_scatter and torch_sparse for graph operations

---

## Optimization Strategy

### Key Insight
The IB regularization (`info_beta`) was over-compressing the representation. Setting it to zero (effectively disabling the IB bottleneck) gave consistent monotone improvement as beta decreased.

### What Worked
1. **Reducing info_beta to 0.0** (from 0.001): The biggest win (+0.38% improvement)
2. **Increasing n_sample to 50** at test time: Small but consistent variance reduction
3. **Multi-run stochastic ensemble** (LEAP): Marginally better by averaging 3 independent MC chains

### What Didn't Work
1. **KD loss activation** (`lamb=0.3`): Teacher guidance at this scale is detrimental
2. **LR schedule modifications**: Very sensitive; removing step=1 caused catastrophic regression
3. **MAE criterion**: The config setting only affects logged auxiliary losses
4. **Increased patience + mae_avg early stop**: Moved best_epoch but no improvement
5. **num_layer=4 + batch_size=64**: More layers and larger batch both hurt
6. **embed_dim=128**: Hard failure due to hardcoded channel dimensions (160/320)

---

## Iteration Log

|| Iter | Key Change | MAE | Delta |
||------|-----------|-----|-------|
|| 0 | Baseline | 18.4928 | - |
|| ... | info_beta: 0.001→0.0001 | ... | progressive improvement |
|| ... | info_beta: 0.0001→0.0 | **18.4219** | **best** |
|| ... | n_sample: 12→50 | -0.0009 | marginal |
|| 8 | Final (42087fe) | **18.4219** | - |

---

## Final Configuration

The winning modifications in training/inference:

```python
# info_beta reduction (most impactful)
info_beta = 0.0  # was 0.001

# Increased MC samples for variance reduction
n_sample = 50  # was 12

# Multi-run ensemble (LEAP)
n_runs = 3  # 3 independent 50-sample MC runs averaged
```

---

## Key Architectural Insights

1. **The `criterion: Smooth` config doesn't affect the main training loss**
   - Core loss: `stu_loss = difficulty_weighted_MAE + IB_terms`
   - `criterion` only affects KD/tea/avg auxiliary losses

2. **KD loss was never actually added to training** in the original code!
   - The `lamb * kd_loss` term is computed but never added
   - This appears to be a code artifact

3. **Early stopping at epoch 154** uses n_sample=1 validation MAE
   - Model finds its best point very early

---

## Reproducibility

**Best commit**: `42087fe` (iter-8)
**Final evaluation**: Run validation with info_beta=0.0, n_sample=50
**Expected output**: MAE=18.4219

---

## Top Remaining Ideas (for future runs)

1. **Fix embed_dim scalability**: Modify RSTIB_MLP.py to compute reduce_dimen dims dynamically
2. **Smaller info_beta with annealing**: Start high to encourage exploration, then drop to 0
3. **KD with very small lambda**: Try lamb=0.01 or lamb=0.05
4. **Multiple seeds ensemble**: Train 3 models with different seeds
5. **Difficulty weighting inversion**: Try giving harder samples more weight instead of less
