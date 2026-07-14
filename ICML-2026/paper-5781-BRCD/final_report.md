# Final Report: paper-5781

- Title: Root Cause Analysis of Failures in Microservices via Bayesian Root Cause Discovery
- Primary metric: `Top-1` (higher)
- Records: 12
- Generated: 2026-07-13T23:49:11Z

## Best Result

- Iteration: 5
- Idea: PARAM-num-candidates-3 — num_root_causes_candidates=3
- Primary metric: 0.8382
- Commit: `c3d419c1a7416268a98347251998c9700bbf6b31`
- Notes: num_root_causes_candidates=3. Further improvement over n=2: Top-1 +78.1% vs baseline (0.4706→0.8382), Top-3 +14.0% (0.8382→0.9559), MRR +33.9% (0.6652→0.8910). Top-5 at ceiling (0.9559). 7140 combos @ 15.5min. Diminishing returns begin — MRR increased but Top-5 saturated.
