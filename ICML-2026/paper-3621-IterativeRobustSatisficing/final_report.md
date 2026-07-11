# Final Report: paper-3621

- Title: Iterative Robust Satisficing: Minimizing Performance Degradation Under Distribution Shift
- Primary metric: `Avg` (higher)
- Records: 7
- Generated: 2026-07-10T17:34:26Z

## Best Result

- Iteration: 5
- Idea: IDEA-11 — Add weight_decay=5e-5
- Primary metric: 0.801
- Commit: `4b115ced6b7804109422117ed9be4c2c34ff1110`
- Notes: Added weight_decay=5e-5 on top of IDEA-01 (M=7 aug) + IDEA-03 (grad clip) + IDEA-12 (target_tau=0.2). All metrics improved: Avg=0.801, Tail=0.6773, Worst=0.651. Metrics from classwise CSV. Compared to baseline: Avg +26.5%, Tail +56.1%, Worst +68.2%.
