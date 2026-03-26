# Paper 82 — OnlineLLMRouting

**Full title:** *Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving*

**Registered metric movement (internal ledger, ASCII only):** +1.61%(2748.6->2792.8)

# Final Optimization Report: Multi-LLM routing (paper-82)

## Summary

**PORT quality (performance aggregate)** rose from **2748.6 → 2792.8 (+1.61%)** after deterministic seeding, better gradient parsing for the surrogate, **distance-weighted ANN** queries (exp kernel ×4), and tighter **L-BFGS-B** tolerances for the small routing QP.

## Key ideas (results ledger)

- **Fixed RNG seeds** so A/B comparisons are not swamped by stochastic load replay.
- **Parse gradients** from the logged telemetry instead of finite-difference hacks where analytic forms exist.
- **ANN retrieval** with exponential distance weights to emphasize near neighbors in embedding space.

## Where to look next

- Serving simulator configs; **`README.md`** for reproducing the PORT benchmark bundle.
