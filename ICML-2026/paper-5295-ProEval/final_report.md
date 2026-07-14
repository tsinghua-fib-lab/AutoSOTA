# Final Report: paper-5295

- Title: ProEval: Proactive Failure Discovery and Efficient Performance Estimation for Generative AI Evaluation
- Primary metric: `BQ-SF_MAE` (lower)
- Records: 12
- Generated: 2026-07-13T01:38:05Z

## Best Result

- Iteration: 8
- Idea: PARAM-12 — n_init=1 stratified init with noise=0.04
- Primary metric: 0.002152
- Commit: `61b53c6bc9987f4e50c4eed22d12dbdc85fd4734`
- Notes: n_init=1 with stratified init (1 quantile spanning full dataset) + noise=0.04 gives BQ-SF MAE=0.002152 — 87.1% improvement from baseline 0.016707! Only 1/13 budget for init, 12 for active acquisition. Remarkable: std=0.0 across 5 seeds (deterministic BQ active path after initial sample). BQ-RPF stable at 0.008167. n_init=1 is the sweet spot: minimal init budget, maximum active learning headroom.
