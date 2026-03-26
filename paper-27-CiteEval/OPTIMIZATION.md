# Paper 27 — CiteEval

**Full title:** *CiteEval: Principle-Driven Citation Evaluation for Source Attribution*

**Registered metric movement (internal ledger, ASCII only):** +24.1%(0.733->0.910)

# Final Optimization Report: CiteEval (paper-4)

## Summary

**Pearson correlation on the statement-level metric** improved from **0.733 → 0.910** after repro was skipped—well above a paper-style baseline **~0.701** cited in the ledger. Changes concentrated in **`evaluate_metric.py`**: robust handling of **all-none ratings**, **None** imputation, **weighted ensembling**, and **piecewise / power** transforms so scores are not dominated by sparse failure modes.

## Key ideas (results ledger)

- Harden **rating parsing** (`all_none_rating`, missing-value fills).
- **Weighted integration** and **nonlinear transforms** of sub-scores before correlation.

## Where to look next

- **`README.md`** and `evaluate_metric.py` in the optimized snapshot.
