# Final Report: paper-5253

- Title: Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills
- Primary metric: `Accuracy` (higher)
- Records: 7
- Generated: 2026-07-14T01:21:56Z

## Best Result

- Iteration: 4
- Idea: PROMPT-01 — Enhanced Aggregation Prompt with Model Reliability Info
- Primary metric: 55.56
- Commit: `3909d43809a32f10d726541e7a0303684de32820`
- Notes: BEST RESULT: Added model reliability scores to aggregation prompt (QwenR1: 51.41% HIGH, Qwen: 35.34% MODERATE). +1.52pp over baseline (54.04% -> 55.56%). QwenR1 as aggregator benefits from knowing which source models are more reliable.
