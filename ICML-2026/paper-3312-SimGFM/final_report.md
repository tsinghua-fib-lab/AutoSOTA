# Final Report: paper-3312

- Title: SimGFM: Simplifying Discrete Flow Matching for Graph Generation
- Primary metric: `Relaxed Validity` (higher)
- Records: 11
- Generated: 2026-07-10T08:25:36Z

## Best Result

- Iteration: 8
- Idea: CODE-5e — temp=0.97 + 20 steps [BEST]
- Primary metric: 99.71
- Commit: `9aa59797c9893680f93761392b9d81c2faee8bd9`
- Notes: Temperature=0.97 + sample_steps=20: Relaxed Validity 99.71 (+0.19 from baseline 99.52). ALL metrics improved: Validity 99.57 (+0.38), Uniqueness 95.67 (+0.05 above baseline!), FCD 0.600 (better). Clear Pareto improvement over baseline on every metric.
