# Final Report: paper-3909

- Title: Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity
- Primary metric: `RougeL` (lower)
- Records: 10
- Generated: 2026-07-10T18:00:02Z

## Best Result

- Iteration: 8
- Idea: ALGO-04 — Maximally aggressive diversity prompt for VS-CoT
- Primary metric: 16.57
- Commit: `9b72ec3d9a1c973aaa55d8bae21bb864502de49b`
- Notes: Maximally aggressive diversity prompt: requires brainstorming 15+ directions, selecting most different, explicit NO OVERLAP constraint. VS_COT: Diversity 37.43% (+1.66pp over iter-4 best), RougeL 16.57% (-0.53pp). Now matches iter-0 baseline RougeL (16.67) while exceeding diversity by +10.3pp. Strong Pareto improvement. Total improvement over v4-flash baseline: Diversity +7.0pp (+23%), RougeL -2.55pp (-13.3%).
