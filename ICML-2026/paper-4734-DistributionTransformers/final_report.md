# Final Report: paper-4734

- Title: Distribution Transformers: Fast Approximate Bayesian Inference With On-The-Fly Prior Adaptation
- Primary metric: `KL-Divergence` (lower)
- Records: 13
- Generated: 2026-07-12T09:08:00Z

## Best Result

- Iteration: 10
- Idea: PARAM-01 — DT-5 lr=0.002 epochs=60
- Primary metric: 0.000257
- Commit: `3263a9c2cb0932945b979ed99481f368f94b8e00`
- Notes: Lower lr=0.002 with epochs=60 gives slight improvement over lr=0.005 (0.000257 vs 0.000265). Cumulative: baseline 0.005746 -> 0.000257 (22.4x better).
