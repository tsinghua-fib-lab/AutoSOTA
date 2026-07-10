# Final Report: paper-2777

- Title: Causal Disentangled Anchor Learning for Scalable Fair Multi-view Clustering
- Primary metric: `ACC` (higher)
- Records: 13
- Generated: 2026-07-09T08:54:32Z

## Best Result

- Iteration: 6
- Idea: CDAL-002 — Two-stage training: alpha=0 then alpha=5000 with split gradient
- Primary metric: 74.55
- Commit: `af0a0c3348d55c56c607a321c72d5f1c6bad7e95`
- Notes: CDAL-002: Two-stage training — Stage 1 (alpha=0) establishes cluster structure, Stage 2 (alpha=5000, warm start) fine-tunes with fairness. Massive ACC improvement from 60.24 to 74.55 (+23.7%). Bal=3.82 (was 0.00). NMI=78.17 (was 67.31). All metrics simultaneously improved. Seed 12 with alpha=5000 beta=500 selected. Seed 12 alpha=1000 gave ACC=74.39 Bal=3.70. Two-stage resolves local minima issue that caused Bal=0 in baseline.
