# Paper 66 — TropicalAttention

**Full title:** *Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms*

**Registered metric movement (internal ledger, ASCII only):** +15.55%(70.69->81.68)

# Final Optimization Report: Tropical attention (paper-66)

## Summary

Over **12** SOTA iterations, **Quickselect length-OOD binary F1** rose from **70.69 → 81.68 (+15.55%)**, clearing the internal target derived from the rubric. The winning recipe leaned on **length-16 auxiliary batches** during training with mixture probability **p ≈ 0.3** so the model sees harder length skew without collapsing in-distribution accuracy.

## Key ideas (results ledger)

- **OOD length augmentation** (length=16 helper batch, stochastic mixing) to stress the tropical attention pathway on longer prefixes.
- Standard LR / regularization nudges in later iterations for stability.

## Where to look next

- Training scripts under `experiments/` (layout varies); **`README.md`** for the exact OOD split definition.
