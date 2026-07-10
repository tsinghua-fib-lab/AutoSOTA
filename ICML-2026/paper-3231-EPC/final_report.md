# Final Report: paper-3231

- Title: ePC: Fast and Deep Predictive Coding in Digital Simulation
- Primary metric: `Test_Accuracy_CE` (higher)
- Records: 7
- Generated: 2026-07-10T05:14:06Z

## Best Result

- Iteration: 3
- Idea: 3231-ALGO-003 — Label smoothing (eps=0.1) on cross-entropy loss
- Primary metric: 87.64
- Commit: `1d2ebca0fd38fade161ea46c948ad7b3069fa9f8`
- Notes: First improvement over baseline! +0.06% (87.58->87.64). Label smoothing reduces overconfidence and improves generalization. Consistent gains in 4/5 seeds. Keeping this change for subsequent iterations.
