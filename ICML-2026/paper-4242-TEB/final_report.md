# Final Report: paper-4242

- Title: Threshold-Based Exclusive Batching for LLM Inference
- Primary metric: `RPS` (higher)
- Records: 14
- Generated: 2026-07-11T23:06:30Z

## Best Result

- Iteration: 6
- Idea: IDEA-12 — CFR + output_margin=0.2 (aggressive KV cache)
- Primary metric: 22.78
- Commit: `f1e56de073117e36e1a03834851115db77d3ceb3`
- Notes: CFR mode with output_margin=0.2 (most aggressive). RPS +4.9% over IFR baseline (22.78 vs 21.72). All 3980 requests successful. Best RPS so far. TPOT slightly higher (266.45 vs 256.48) but within tolerance.
