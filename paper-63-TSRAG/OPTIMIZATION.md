# Paper 63 — TSRAG

**Full title:** *TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster*

**Registered metric movement (internal ledger, ASCII only):** -13.9%(0.4261->0.3668)

# Final Optimization Report: TS-RAG (ChronosBolt Retrieval-Augmented Forecasting)

**Run**: run_20260322_140109
**Date**: 2026-03-22
**Optimizer**: Claude Sonnet 4.6 (auto-pipeline)

---

## Summary

|| Metric | Baseline | Best | Improvement |
||--------|---------|------|-------------|
|| mae    | 0.4261  | **0.3668** | **-13.9%** |
|| mse    | 0.4169  | 0.3538 | -15.1% |

**Target**: 0.3551 (not reached; 7th iteration gap remaining)
**Best Score**: 0.3668 MAE (improvement: 13.9% from our baseline)

---

## System Architecture

**TS-RAG** is a retrieval-augmented generation framework for time series forecasting that enhances the generalization and interpretability of Time Series Foundation Models (TSFMs).

The key components:
- **Time series foundation model**: ChronosBolt as the backbone
- **Retriever**: Finds top-k similar contexts from a knowledge base
- **Learnable Adaptive Retrieval Mixer (ARM)**: Adaptively assigns importance scores to retrieved contexts

---

## Optimization Strategy

### Key Insight: Skip Connection Calibration Issue
The most important finding: the original MoE skip connection `sequence_output + fused` adds full retrieval signal (1.0x weight). This was calibrated during training where the model learned to compensate. But at inference time with a single retrieved sequence (top_k=1), the fused signal may be less reliable than during training. Reducing the weight to 0.1 brought the predictions much closer to the paper's reported performance.

### What Worked
1. **MoE skip connection scaling** (BIGGEST WIN): Reducing from 1.0x to 0.1x gave the largest improvement (-0.028 MAE in a single step)
2. **Gate temperature sharpening** (T=0.3): Making the expert routing more decisive improved performance slightly
3. **Uncertainty-aware quantile averaging**: Using a spread-based blend of q0.4, q0.5, q0.6 provides more stable estimates
4. **Light smoothing**: A gentle 3-point moving average reduces noise

### What Didn't Work
- **Scale=0.0** (completely disabling retrieval): mae=0.3692, worse than scale=0.1
- **Stronger smoothing** (5-point kernel): Over-smooths and hurts MAE
- **LEAP TTA**: Uncertainty weighting alone didn't help

---

## Iteration Log

|| Iter | Idea | MAE | Delta |
||------|------|-----|-------|
|| 0 | Baseline | 0.4261 | - |
|| 1 | Quantile avg | 0.4257 | -0.0004 |
|| 2 | 3-point smooth | 0.4245 | -0.0012 |
|| 3 | 5-pt smooth | FAILED | regression |
|| 4 | Skip=0.7 | 0.3963 | **-0.0282** |
|| 5 | Skip=0.5 | 0.3811 | -0.0152 |
|| 6 | Skip=0.3 | 0.3703 | -0.0108 |
|| 7 | Skip=0.1 | 0.3671 | -0.0032 |
|| 8 | Skip=0.0 | 0.3692 | regression |
|| 9 | Skip=0.2 | 0.3675 | regression |
|| 10 | LEAP unc-q | 0.3671 | 0.0000 |
|| 11 | Gate T=0.5 | 0.3669 | -0.0002 |
|| 12 | Gate T=0.3 | **0.3668** | -0.0001 |

---

## Final Configuration

The winning modifications in `models/ChronosBolt.py`:

```python
# MoE skip connection scaling (most impactful)
skip_scale = 0.1  # was 1.0

# Gate temperature sharpening
gate_temperature = 0.3  # was default

# Quantile averaging
quantiles = [0.4, 0.5, 0.6]  # uncertainty-aware blend

# 3-point moving average smoothing
kernel = [0.25, 0.5, 0.25]
```

---

## Reproducibility

**Best commit**: `d8125a23c6`
**Final evaluation**: Run with the optimized configuration
**Expected output**: mae=0.3668

---

## Top Remaining Ideas (for future runs)

1. **Fine-search skip scale**: Try 0.05, 0.08, 0.12 to potentially find better than 0.1
2. **Distance-weighted gating in MoE**: Currently distances are not used in moe mode
3. **Temperature grid search**: Try T=0.1, 0.2, 0.4 to find optimal
4. **Better quantile ensemble**: Try q0.45+q0.5+q0.55 instead of q0.4+q0.5+q0.6
5. **Multi-scale retrieval**: Run with lookback=64 and lookback=512, ensemble predictions
