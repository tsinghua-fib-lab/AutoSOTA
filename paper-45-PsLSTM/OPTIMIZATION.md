# Paper 45 — PsLSTM

**Full title:** *Unlocking the Power of LSTM for Long Term Time Series Forecasting*

**Registered metric movement:** -0.86% (MSE: 0.1492 → 0.1479)

---

# Final Optimization Report

## Results Summary

| Metric | Baseline | Best Achieved | Delta | Target |
|--------|----------|---------------|-------|--------|
| **MSE** | 0.14922569 | **0.14794841** | **-0.86%** | ≤0.1462 (-2.0%) |
| MAE | 0.20801531 | 0.20630378 | -0.82% | — |

**Target was NOT fully achieved** (target: -2.0%, achieved: -0.86%). Best result is **0.86% below baseline**, representing a real improvement but falling short of the 2% goal.

---

## Key Changes Applied

### 1. `models/P_sLSTM.py` — Bug Fix: Dropout propagation
```diff
+            dropout=configs.dropout,
```
The `dropout=0.1` CLI argument was not being passed to `xLSTMBlockStackConfig`, causing the xLSTM blocks to use `dropout=0.0` (no dropout). Adding this single line fixed the bug and enabled proper regularization.

### 2. `exp/exp_main.py` — MC Dropout + Input Perturbation Ensemble
- `model.train()` during inference (enables dropout stochasticity)
- **MC Dropout**: K=20 forward passes with different dropout masks
- **Input Perturbation**: 5 noise levels (σ=0.01), first pass uses clean input
- Total: 100 forward passes per test sample, averaged for variance reduction

---

## What Worked
1. **Dropout Bug Fix** (critical): Model was trained with `dropout=0.0` despite `--dropout 0.1` being specified.
2. **MC Dropout Ensemble**: Running K=10-30 stochastic forward passes with `model.train()` reduced test-time variance.
3. **Input Perturbation**: Adding small Gaussian noise (σ=0.01) to inputs across 5 passes further diversified the ensemble.

## What Didn't Work
- **RevIN (Reversible Instance Normalization)**: Caused distribution mismatch
- **Alternative LR schedules**: Type1's aggressive halving caused early plateau
- **Higher dropout (0.2)**: Overregularized the model

---

## Conclusion
The main limitation was architectural: the P-sLSTM processes only 6 patches (336/56) with no overlap — very coarse temporal granularity. The model reached a convergence plateau by epoch 8 regardless of training duration.
