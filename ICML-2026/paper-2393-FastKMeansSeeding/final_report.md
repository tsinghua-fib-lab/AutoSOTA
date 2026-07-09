# Final Report: paper-2393

- Title: Fast k-means Seeding Under The Manifold Hypothesis
- Primary metric: `Cost` (lower)
- Records: 13
- Generated: 2026-07-08T13:12:13Z

## Best Result

- Iteration: 12
- Idea: ALGO-05 — Final: Max-norm subset=100 + ef=35 (20 runs)
- Primary metric: 2.75
- Commit: `90e9bda2b636598e90723f9540e27573f332056d`
- Notes: FINAL OPTIMIZED RESULT: Max-norm first center (subset_size=100) + ef=35. Cost=2.7480 (2.75 paper units) vs baseline 2.80 — 1.8% improvement. Time=14.62ms (essentially unchanged from 14.31ms baseline). QKMEANS now MATCHES AFKMC2 on Cost (2.7480 vs 2.7485) while being 2.1x FASTER (14.62ms vs 30.53ms). Only 0.8% worse than k-means++ on Cost but 10.7x faster (14.62ms vs 155.93ms). The max-norm first center finds a moderately high-norm representative point, creating a kappa distribution that enables better exploration through slightly more uniform sampling — a counterintuitive but effective improvement confirmed across 20-run statistical averaging.
