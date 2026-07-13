# Final Report: paper-5237

- Title: LieStoNet: Learning Lie Symmetries from Spatiotemporal Data for Stochastic Dynamical Systems
- Primary metric: `Maximum Principal Angle` (lower)
- Records: 8
- Generated: 2026-07-13T06:27:43Z

## Best Result

- Iteration: 7
- Idea: ALGO-05 — 12000 steps + gentler cosine decay (alpha=0.3)
- Primary metric: 1.9514 (RAW stacking)
- Commit: `e5839345fa4f6c349d8ddf1143d9e70869b0bb45`
- Notes: |
  **Result**: RAW Max Principal Angle = 1.9514° (angles=[0.10, 0.55, 1.95]°). BALANCED mode: not reported (was higher in prior runs).
  **Context — seed sensitivity**: Baseline (seed=0) gave 13.72°, but identical code with seed=42 gave 10.81°, seed=123 gave 5.38°. The 85.8% reduction claim uses the worst seed as reference; against the 3-seed mean (~10.0°) the reduction is ~80%, and the BALANCED mode showed much weaker alignment across all seeds (24–29°).
  **Context — loss terms**: Of 7 loss terms, L3 (skew-symmetry), L7 (pushforward), L2 (Jacobi), and L4 (bilinearity) showed negligible contribution during training (L3=0, L7 flat at ~6.98, L2/L4 at machine epsilon). Only L1, L5, L6 drove optimization.
  **Caution**: Single-seed result; no multi-seed statistics for the final config. The evaluation reports RAW stacking of (tau, xi) components without per-generator normalization; BALANCED mode results are higher. These factors likely overstate the true improvement.
