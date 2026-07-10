# Final Report: paper-3185

- Title: Conditional Coverage Diagnostics for Conformal Prediction
- Primary metric: `L1-ERT` (lower)
- Records: 12
- Generated: 2026-07-09T17:12:02Z

## Best Result

- Iteration: 9
- Idea: ALGO-01b — Better Normalized Score: 80/20 split with larger LGBM (n=1000, leaves=50)
- Primary metric: 0.0075
- Commit: `c859d905f8b8b1f76ba407cce373189187e61a9c`
- Notes: Normalized score S=|Y-mu|/sigma with 80/20 cal split and larger LGBM (n_estimators=1000, num_leaves=50). Slightly better than 50/50 version (0.0075 vs 0.0099). L1-ERT now 91.7% below baseline 0.0903. Essentially at noise floor - near-perfect conditional coverage achieved. Negative L2-ERT is sampling noise near zero.
