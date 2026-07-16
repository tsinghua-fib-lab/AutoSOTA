# Final Report: paper-6249

- Title: Active Timepoint Selection for Learning Measure-Valued Trajectories
- Primary metric: `Mean W2` (lower)
- Records: 13
- Generated: 2026-07-16T05:18:13Z

## Best Result

- Iteration: 9
- Idea: IDEA-06c — Two-phase: uncertainty (1-20) + 1 step velocity-weighted, switch_step=20
- Primary metric: 0.0157
- Commit: `150f82399f632c8ba8ba95b1aca953dfe80b96ca`
- Notes: Optimal: 20 steps pure uncertainty + 1 final step velocity-weighted. Mean W2=0.0157 (28.3% better than baseline 0.0219), w-W2=0.0167 (24.4% better than baseline 0.0221). Best result so far. Pattern: maximal exploration with minimal velocity bias at the very end.
