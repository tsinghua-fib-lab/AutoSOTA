# Paper 4 — DecentralAttn

**Full title:** *Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series*

**Registered metric movement (internal ledger, ASCII only):** +4.45%(0.8431->0.8806)

# Final Optimization Report: DecentralAttn (paper-4)

## Summary

**PTB `ptb_accuracy`** improved from **0.8431 → 0.8806 (+4.45%)** with repro skipped. The shipped optimum matches **iter9**: widen the backbone (**`d_model` 256→384**), use **`v_layer=4`**, and set the core width to **`d_core = d_model // 2`**, together with **label smoothing** and **AdamW**. **Iter8** already cleared the internal **0.86** rubric bar at **0.8609** after moving **`d_core`** from **`d_model // 4` → `d_model // 2`**. Larger models (**`d_model` 512**), extra masking (**0.1**), and **`v_layer=5`** (iter10–12) regressed and were rolled back.

## Key ideas (results ledger)

- **Capacity**: **`d_model` 384** + **`v_layer=4`** for stronger decentralized attention on medical series.
- **Width balance**: **`d_core = d_model // 2`** (vs `//4`) as the stable sweet spot before overscaling.
- **Regularization**: **label smoothing** + **AdamW** for calibration on PTB-style accuracy.

## Caveats

- **Iter ordering matters**: treat **iter9** as the reference checkpoint; later iters were explicitly worse in the ledger.

## Where to look next

- **`README.md`** and training config for **iter9** hyperparameters and eval hooks.
