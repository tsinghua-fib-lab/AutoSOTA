# Final Report: paper-907

- Title: Beyond Point-wise Neural Collapse: A Topology-Aware Hierarchical Classifier for Class-Incremental Learning
- Primary metric: `A_Avg` (higher)
- Records: 4
- Generated: 2026-07-06T20:46:19Z

## Best Result

- Iteration: 2
- Idea: ALGO-02 — Adaptive Prototype Allocation per Class
- Primary metric: 90.21
- Commit: `7cbf50fd6857daf0704a5b2ab7c92bfec7a95f1d`
- Notes: Redistributed prototype budget (6000 total) proportional to intra-class feature dispersion. A_Avg +0.512% (90.21 vs 89.698 baseline), A_Last +2.34% (87.49 vs 85.15). HC-SOINN gain over FC: +2.80%. All guardrails satisfied. Adaptive allocation consistently outperforms uniform across all tasks.
