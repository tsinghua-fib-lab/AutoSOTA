# Final Report: paper-5883

- Title: A Studentized Spherical Harmonics–Based Nonparametric Two-Sample Test for Compositional and Directional Data
- Primary metric: `Power (Asymptotic)` (higher)
- Records: 9
- Generated: 2026-07-14T00:46:27Z

## Best Result

- Iteration: 7
- Idea: ALGO-08 — Null-calibrated asymptotic p-values (B_cal=50)
- Primary metric: 0.261
- Commit: `4545d92882e5fbcb83e2a84f5f44d9411bfabf30`
- Notes: Per-replicate null calibration: estimate null mean mu_0 and std sigma_0 from B_cal=50 within-replicate permutations, then T_calibrated = (T - mu_0)/sigma_0. Corrects N(0,1) approximation at n=25 (empirical null std ~0.96). Asymptotic power +4.0% (0.251→0.261). Permutation power stable at 0.203. First improvement over baseline.
