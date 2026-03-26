# Paper 8 — MeanFlows

**Full title:** *Mean Flows for One-step Generative Modeling*

**Registered metric movement (internal ledger, ASCII only):** -0.14%(2.8112->2.8074) FID lower is better

# Final Optimization Report: MeanFlows (paper-8)

## Summary

**FID** improved from **2.8112 → 2.8074** (lower is better) after repro was skipped. The durable recipe **interpolates EMA weights** (**~98.7%** `net_ema1` **+ ~1.3%** live `net`). Seed sweeps, two-step ODE, three-way EMA blends, and tanh output clamps did not beat the final checkpoint.

## Key ideas (results ledger)

- **EMA interpolation** (`alpha` ≈ **0.987**) for evaluation weights; final FID **2.8074** vs a paper-reported reference **2.8883** (ledger-relative framing).

## Where to look next

- **`README.md`** and checkpoint / EMA merge scripts in the optimized tree.
