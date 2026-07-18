# Final Report: paper-3573

- Title: Adaptive Generation of Bias-Eliciting Questions for LLMs
- Primary metric: `Average_Fitness_Overall` (lower)
- Records: 8
- Generated: 2026-07-10T13:30:11Z

## Best Result

- Iteration: 3
- Idea: ALGO-07 — Constitutional Self-Critique with ALGO-01 debiasing prompt
- Primary metric: 0.226
- Commit: `06fb7677e8607cedaf55eccad36d6c39ba7cdb85`
- Notes: ALGO-07: Constitutional self-critique (pre-response check instruction) + ALGO-01 debiasing prompt. Overall 0.226 vs 0.232 (ALGO-01). Gender slightly worse (0.220 vs 0.208), Race better (0.257 vs 0.263), Religion at theoretical minimum (0.203 vs 0.225). Overall best so far.
