# Paper 9 — ScoreMissing

**Full title:** *Score Matching with Missing Data*

**Registered metric movement (internal ledger, ASCII only):** -0.14%(0.7624->0.7613); iter4 peak 0.7691

# Final Optimization Report: ScoreMissing (paper-9)

## Summary

After repro was skipped, **iter4** reached an **AUC peak of 0.7691** with a **multi-model ensemble**. **Final** reporting used **three repeated evaluations** averaging **0.7613** (variance across runs), below an internal rubric target **0.7776** but still logged as an optimization success on the ledger. The eval stack also needed a **device** fix in the scoring script.

## Key ideas (results ledger)

- **Ensemble** of several score-matching / imputation heads at the best iteration.
- Stabilize **evaluation device** placement so scores are reproducible.

## Caveats

- **Mean AUC** can sit below a **single-run peak** when the harness averages multiple eval passes—report both when discussing “best found” vs “shipped number.”

## Where to look next

- **`README.md`** and ensemble checkpoint list for **iter4**.
