# Final Report: paper-2075

- Title: Data Reconstruction: Identifiability and Optimization with Sample Splitting
- Primary metric: `L2_Distance` (lower)
- Records: 8
- Generated: 2026-07-15T17:31:21Z

## Best Result

- Iteration: 5
- Idea: CODE-02-scale20 — KKT loss scaling factor=20.0 for Loo method
- Primary metric: 2.8499
- Commit: `2252bc412da400cf29488620fbd1e78c153088ff`
- Notes: KKT scale=20.0 delivers further improvement over scale=10.0. Final L2=2.8499 at T=99000, 39.6% improvement over baseline 4.72. Score dropped sharply at T~86000 as LR decay reduced effective LR to ~6e-4.
