# Final Report: paper-2023

- Title: Global Directional Priors with Local Statistical Validation for Scalable Causal Discovery
- Primary metric: `SHD` (lower)
- Records: 10
- Generated: 2026-07-07T17:55:36Z

## Best Result

- Iteration: 8
- Idea: PARAM-tau-090 — Lower tau=0.90 with bandwidth-tuned scores
- Primary metric: 63.3
- Commit: `24731841cef82bfc55daf64993fc1dd2dbcbaad6`
- Notes: tau=0.90 with CODE-03+CODE-04 (bandwidth tuning + adaptive spouse). Lower tau compensates for compressed scores, improving candidate recall. Best so far: SHD -1.1, F1 +0.013.
