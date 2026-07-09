# Final Report: paper-2529

- Title: Addressing Instrument-Outcome Confounding in Mendelian Randomization through Representation Learning
- Primary metric: `Mean_Bias` (lower)
- Records: 5
- Generated: 2026-07-08T19:50:52Z

## Best Result

- Iteration: 1
- Idea: CODE-02 — LR Scheduler (ReduceLROnPlateau, patience=20, factor=0.5)
- Primary metric: 0.00425
- Commit: `c496259000838901dabcc0a39794f6307b718017`
- Notes: CODE-02: Enabled ReduceLROnPlateau scheduler (factor=0.5, patience=20, monitor=val/tot_loss). 3 seeds (42, 43, 44). Mean_Bias=0.00425 (63.6% reduction from baseline 0.01166). SD=0.01773 (45.4% reduction from baseline 0.03248). This is the best configuration found.
