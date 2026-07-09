# Final Report: paper-2203

- Title: The Hippocampal Place Field Gradient: A Bio-inspired Framework Building Multiscale Representation for Better Sample Efficiency
- Primary metric: `PPL` (lower)
- Records: 13
- Generated: 2026-07-08T05:54:36Z

## Best Result

- Iteration: 12
- Idea: PARAM-02 — WSD + lr=2e-3 + graduated sigma (FINAL BEST)
- Primary metric: 33.57
- Commit: `ecdb5037576d468d51bafcfd7c5cc397b67af120`
- Notes: WSD schedule with lr=2e-3 and sigma gradient [200,400,600,800]. FINAL BEST: PPL 33.57 vs baseline 50.99. Total improvement: 17.42 PPL (34.2% reduction). Higher LR with WSD continues to improve. Last VAL PPLs: 40.01, 39.23, 37.77, 36.68, 35.59, 34.37, 33.57 — still rapidly improving at end. The WSD schedule + higher LR accounts for most of the improvement; HIPE sigma gradient contributes modestly.
