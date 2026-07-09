# Final Report: paper-1108

- Title: InfoFlow KV: Information-Flow-Aware KV Recomputation for Long Context
- Primary metric: `F1_guided_recompute_norm_15pct` (higher)
- Records: 18
- Generated: 2026-07-06T19:13:43Z

## Best Result

- Iteration: 11
- Idea: CODE-05 — Use inference-consistent instruction prompt for scoring query
- Primary metric: 0.45
- Commit: `63f3d40abd30891c07aef8d52c25a6a9b5e76ec5`
- Notes: Scoring query tokens now use full instruction prompt (Answer the question...Question: X\nAnswer:) instead of raw question only. This aligns scoring attention with inference attention. F1 improved from 0.4441 to 0.4500 (+0.0059). 91/200 correct vs 89 baseline.
