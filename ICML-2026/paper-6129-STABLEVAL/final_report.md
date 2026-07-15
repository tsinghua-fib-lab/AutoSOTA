# Final Report: paper-6129

- Title: STABLEVAL: Disagreement-Aware and Stable Evaluation of AI Systems
- Primary metric: `Agent_Score_PEC_llama-13b` (higher)
- Records: 7
- Generated: 2026-07-14T11:20:33Z

## Best Result

- Iteration: 6
- Idea: IDEA-02+11+06 — Consistency alpha=0.1 + Adaptive prior C=2.0 + Temperature T=1.15
- Primary metric: 0.151254
- Commit: `b5ff015a0fe55b967e23765e76ad42cc9c3e93c9`
- Notes: Best result: combination of all three algorithmic improvements. Consistency-weighted E-step (alpha=0.1) + adaptive Dirichlet prior (C=2.0) + temperature scaling (T=1.15). llama-13b PEC improved from baseline 0.130641 to 0.151254 (+15.8%). DS improved to 0.121875 (+8.3%). Guardrails: gpt-4 PEC -0.72%, claude-v1 -0.65%, both within 3% tolerance. Ranking stability unchanged. Added --pec-consistency-weight, --dirichlet-adaptive, --pec-temperature flags to pipeline.
