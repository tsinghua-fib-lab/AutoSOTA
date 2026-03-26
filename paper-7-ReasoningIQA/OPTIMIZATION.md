# Paper 7 — ReasoningIQA

**Full title:** *Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment*

**Registered metric movement (internal ledger, ASCII only):** +2.68%(0.7803->0.8012)

# Final Optimization Report: ReasoningIQA (paper-7)

## Summary

**PLCC** improved from **0.7803 → 0.8012 (+2.68%)** by changing how CLS and patch tokens are fused for the quality head instead of relying on a single pooling path.

## Key ideas (results ledger)

- Blend **CLS** and patch statistics: tune **alpha_cls** with **patch_max** and a three-way mix (CLS + patch_mean + patch_max) so global and local evidence both contribute.

## Where to look next

- **`README.md`** and eval config for fusion weights checked in with the winning run.
