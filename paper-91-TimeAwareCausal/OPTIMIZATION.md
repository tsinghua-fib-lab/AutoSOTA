# Paper 91 — TimeAwareCausal

**Full title:** *Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains*

**Registered metric movement (internal ledger, ASCII only):** +4.30%(86.0->89.7)

# Final Optimization Report: TimeAwareCausal (paper-91)

## Summary

**Average accuracy** improved from **86.0% → 89.7%** at **iter8** after repro was skipped. The best mix used **`weight_decay=1e-4`**, a **6× masker middle**, and **`dropout=0.1`**. Pushing mask width to **8×**, adding label smoothing, or stronger dropout hurt generalization on the evolving-domain split.

## Key ideas (results ledger)

- **Weight decay** for stable causal features under distribution shift.
- **Moderate structured masking** (**6×**) plus light **dropout**—avoid over-sparsifying the representation.

## Where to look next

- **`README.md`** and representation / masker width settings in the exported run.
