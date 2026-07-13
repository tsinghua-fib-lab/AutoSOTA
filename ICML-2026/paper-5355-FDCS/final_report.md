# Final Report: paper-5355

- Title: Fair Decisions from Calibrated Scores: Achieving Optimal Classification While Satisfying Sufficiency
- Primary metric: `Accuracy` (higher)
- Records: 8
- Generated: 2026-07-13T01:30:25Z

## Best Result

- Iteration: 4
- Idea: SOFT-02 — Multi-epsilon soft-sufficiency with finer resolution
- Primary metric: 0.8707
- Commit: `0f9746a51b2a7258d1f985069ff5a54f642e4e36`
- Notes: Multi-epsilon soft-sufficiency with res=1e-4. Best: eps=0.005, accuracy=0.8707 (White: p=0.9235,q=0.2413; Black: p=0.9186,q=0.2462). Also tested eps=0.004 (accuracy=0.8697). +0.0031 vs baseline (0.8676). All guardrails satisfied. Finer resolution (1e-4 vs 5e-4) found +0.0003 additional improvement.
