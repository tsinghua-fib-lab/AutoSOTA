# Paper 48 — Shapiq

**Full title:** *shapiq: Shapley Interactions for Machine Learning*

**Registered metric movement (internal ledger):** +30.3%(0.76→0.99 precision_at_10)

## Summary

At a fixed Monte Carlo budget (~200 samples), **precision_at_10** rose from **0.76** to **0.99**. The largest step was switching to **coalition-size-only stratification** (`StratifiedBySize`: `stratify_coalition_size=True`, `stratify_intersection=False`) instead of plain SHAPIQ or full SVARMIQ-style stratification. **`pairing_trick`**, **`N_ENSEMBLE=100`**, and diverse **large-prime** random seeds reduced variance enough for stable top-k ranking.

## Key ideas

- **StratifiedBySize** allocates samples evenly across coalition sizes without tracking intersection strata—better variance than baseline SHAPIQ at the same budget.
- **Ensemble + pairing** and seed diversity avoid correlated duplicate runs that plateaued earlier configs.

## Where to look

- **`eval_shapiq.py`** (final evaluator entrypoint).
- Pipeline run **`run_20260324_211015`** under `optimizer/papers/paper-1593/runs/` on the source machine.
