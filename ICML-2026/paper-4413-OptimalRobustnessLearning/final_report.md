# Final Report: paper-4413

- Title: Towards Optimal Robustness in Learning-Augmented Paging
- Primary metric: `Cost Ratio` (lower)
- Records: 7
- Generated: 2026-07-11T08:44:01Z

## Best Result

- Iteration: 1
- Idea: PARAM-01 — tau=4 pred_budget (uniform, from reproduction eval)
- Primary metric: 1.2123
- Commit: `e51b42c45b11060aa15fe3c7285efa4b751fa3e2`
- Notes: Increased pred_budget (tau) from 1 to 4. Cost Ratio improves 0.7% (1.2208->1.2123), Hit Ratio improves 0.9% (24.33%->24.56%). Both metrics improve simultaneously. Results from reproduction benchmark --full-sweep output (stat/ directory, all 13 traces). The primary improvement is driven by sphinx3 trace where tau=8 is optimal (CR 1.357->1.266).
