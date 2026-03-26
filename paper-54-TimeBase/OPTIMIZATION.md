# Paper 54 — TimeBase

**Full title:** *TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting*

**Registered metric movement (internal ledger, ASCII only):** -2.12%(0.1684->0.16485) avg_mse lower is better

# Final Optimization Report: TimeBase (paper-54)

## Summary

**avg_mse** improved from **0.1684 → 0.16485** (lower is better), beating an internal target **0.165**. **GELU** was wired through **ts2basis / basis2ts**, **`basis_num` grew (8→12→16→30)**, and on the long **pred_720** horizon **`use_orthogonal=0`** with **lr=4e-2** and uniform **ow=0.02** gave the best trade-off.

## Key ideas (results ledger)

- **Nonlinearity**: **GELU** in the minimal basis pathway.
- **Capacity**: increase **basis count** to **30** where the signal supports it.
- **720-step split**: disable orthogonal regularizer and use **higher LR** with small **output weights**.

## Where to look next

- **`README.md`** and the basis / horizon-specific config blocks from **iter0–12**.
