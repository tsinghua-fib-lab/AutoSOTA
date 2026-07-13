# Final Report: paper-5090

- Title: Constrained Multi-Objective Reinforcement Learning with Max-Min Criterion
- Primary metric: `optimal_value_error` (lower)
- Records: 13
- Generated: 2026-07-12T18:33:52Z

## Best Result

- Iteration: 10
- Idea: IDEA-12 — Sweep: l_w annealing range to 0.008->0.0002
- Primary metric: 0.003545
- Commit: `cdc0d88393f5b513f7768b3f233e8a8dd8de8b02`
- Notes: Sweep over l_w_max/l_w_min. Best: l_w_max=0.008, l_w_min=0.0002. Error 0.003545 — beats paper's 0.004! Constraint satisfaction 2/3.
