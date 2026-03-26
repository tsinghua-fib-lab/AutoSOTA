# Paper 76 — AANet

**Full title:** *AANet: Virtual Screening under Structural Uncertainty via Alignment and Aggregation*

**Registered metric movement (internal ledger, ASCII only):** +6.79%(0.6548->0.6993)

# Final Optimization Report: AANet virtual screening (paper-76)

## Summary

**BEDROC (holo)** improved from **0.6548 → 0.6993 (+6.79%)**, beating the rubric target **0.6679**. The recipe combines **multi-seed CroppingPocket** ensembles with **z-score normalization** of `adapt` vs `max` docking scores before a **50/50 fusion**.

## Key ideas (results ledger)

- **3-seed** structural cropping to reduce variance from pocket alignment.
- **Normalize** competing score channels before fusion so neither dominates purely due to scale.

## Where to look next

- Screening driver scripts and YAML that list seeds; **`README.md`** for dataset / protein prep.
