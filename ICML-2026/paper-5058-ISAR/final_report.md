# Final Report: paper-5058

- Title: Instance-Specific Approximation Ratios for Correlation Clustering and Max-Cut
- Primary metric: `Instance-Specific_Approximation_Ratio` (lower)
- Records: 7
- Generated: 2026-07-12T20:27:24Z

## Best Result

- Iteration: 4
- Idea: IDEA-12 — MWU rho=2 tuning + eps=0.002
- Primary metric: 1.6357
- Commit: `e611db80334992457f9e6480496940eeaa8ef488`
- Notes: IDEA-12: MWU rho parameter tuning. rho=2 outperforms rho=3 at same epsilon. Best: eps=0.002, rho=2 → 689. Ratio 1127/689=1.6357 (-2.47% from baseline). Rho controls oracle packing density vs iterations (T ∝ rho/ε²). Smaller rho = faster convergence for this instance.
