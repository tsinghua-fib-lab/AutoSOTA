# Final Report: paper-1519

- Title: Performative Policy Gradient: Optimality in Performative Reinforcement Learning
- Primary metric: `Expected_Average_Return_Vpipi` (higher)
- Records: 12
- Generated: 2026-07-08T04:02:03Z

## Best Result

- Iteration: 5
- Idea: PARAM-nf5 — num_followers=5 with no regularization
- Primary metric: 9.261366
- Commit: `1808bae518a6eedbbf36888800a583512ac2a9f0`
- Notes: Set num_followers=5 (middle ground between 2 and 50). Vpipi=9.26 (53x baseline, 5.2x better than nf=2). Very high variance (std=40.6). d_diff=0.00153 < 0.011. No regularization applied.
