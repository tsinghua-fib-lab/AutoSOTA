# Final Report: paper-3042

- Title: Constrained Meta Reinforcement Learning with Provable Test-Time Safety
- Primary metric: `reward_regret` (lower)
- Records: 7
- Generated: 2026-07-09T18:23:32Z

## Best Result

- Iteration: 5
- Idea: FALLBACK-01 — Safe Policy Fallback — median-noise with max-noise fallback
- Primary metric: 8309.0
- Commit: `c45308f8e9c86a18c5cd064f3adde5e07d030f48`
- Notes: When median-noise safe policy (0.3) is infeasible, fall back to max-noise (0.5). 5/10 tasks used fallback. Regret improved to 8.3K with constraint 1.41. Fallback+ALGO-02 alpha acceleration pushes alpha too high on fallback tasks. 4/10 valid runs. Marginal improvement over ALGO-02 alone (8.5K, 5/10).
