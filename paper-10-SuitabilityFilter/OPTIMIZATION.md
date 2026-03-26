# Paper 10 — SuitabilityFilter

**Full title:** *Suitability Filter: A Statistical Framework for Classifier Evaluation in Real-World Deployment Settings*

**Registered metric movement (internal ledger):** +1.56%(0.9687→0.9838)

## Summary

The best OOD suitability score improved from **0.9687** to **0.9838** by stacking three stability-focused changes: **isotonic calibration**, **10-fold multi-fold training**, and a compact **3-feature subset** (`conf`, `logit`, `loss`) combined with **Stouffer Z-score** aggregation. This combination outperformed larger feature bundles and gave more stable cross-split ranking.

## Key ideas

- **Calibrate first**: isotonic mapping reduced score distortion in high-confidence regions.
- **More folds, same objective**: 10-fold training improved robustness under distribution shift.
- **Feature discipline**: the smaller conf/logit/loss subset generalized better than wider handcrafted sets.

## Where to look

- `filter/suitability_filter.py`
- `run_fmow*.py` scripts in this repo snapshot
