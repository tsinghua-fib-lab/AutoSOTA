# Optimization Report: Inverse Methods for Missing Data Imputation

**Paper ID**: 100  
**Repository folder**: `paper-100-InvMissingData`  
**Source**: AutoSota optimizer run artifact (`final_report.md`).  
**Synced to AutoSota_list**: 2026-03-22  

---

# Optimization Results: Inverse Methods for Missing Data Imputation

## Summary
- Total iterations: 1
- Best `mae`: 0.14614 (baseline: 0.14991, improvement: -2.5%)
- Target achieved: 0.14614 ≤ 0.147 ✓
- Best commit: 2b93f31e18

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| MAE    | 0.14991  | 0.14614 | -2.5% |
| WASS   | 0.09233  | 0.09014 | -2.4% |
| RMSE   | 0.39511  | 0.38953 | -1.4% |
| MSE    | 0.15611  | 0.15173 | -2.8% |

## Key Changes Applied
| Change | File | Effect | Notes |
|--------|------|--------|-------|
| n_pairs: 2 → 8 | benchmark_kpi.py:74 | MAE -2.5% | Single-line change; more gradient signal per epoch |

## What Worked
- **Increasing n_pairs from 2 to 8**: This provides 4× more batch pairs sampled per epoch for the kernel regression training loop. Each epoch now has much richer gradient signal, covering more combinations of data samples and feature predictions. This leads to better convergence with the same early-stopping patience.

## What Didn't Work
- No failed experiments in this run; target was reached in iteration 1.

## Top Remaining Ideas (for future runs)
1. **IDEA-002**: Increase stop patience to 100 + epochs to 1000 — baseline output shows MAE still very slowly decreasing at epoch 500; more training may help further
2. **IDEA-003**: Reduce labda from 1.0 to 0.1 — lower kernel ridge regularization may allow tighter fit
3. **IDEA-005**: Tune sigma scales to data-appropriate [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0] for standardized data
4. **IDEA-010**: Combined n_pairs=8 + labda=0.1 + stop=100 compound gains
5. **IDEA-013**: Bias correction — compute systematic bias on validation set and subtract from predictions
