# Final Report: paper-3315

- Title: Anytime Detection of Strategic Deviations in Multi-Agent Systems
- Primary metric: `Empirical FWER` (lower)
- Records: 7
- Generated: 2026-07-10T03:40:12Z

## Best Result

- Iteration: 6
- Idea: ALGO-01 — Optimal 2-comp mixture {0.05, 1.50} (0.2, 0.8)
- Primary metric: 0.103
- Commit: `c6320de7d6e4d9e51ac60c2dabbf32febcc12405`
- Notes: Best result: 2-component mixture of lambda in {0.05, 1.50} with weights (0.2, 0.8). Detection time 66.1 rounds (94.6% reduction from baseline 1228.4). FWER=0.103 within alpha=0.2. All alpha levels pass. This beats pure lambda=1.50 on both FWER (0.103 vs 0.127) and detection (66.1 vs 68.7).
