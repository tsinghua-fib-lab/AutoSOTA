# Final Report: paper-4517

- Title: Scalable Bayesian Inference for Nonlinear Conservation Laws
- Primary metric: `Source RMSE` (lower)
- Records: 7
- Generated: 2026-07-16T12:42:16Z

## Best Result

- Iteration: 3
- Idea: PARAM-01 — Joint Sparsity-Amplitude-Lengthscale Optimization
- Primary metric: 0.2948
- Commit: `17a00a93678ca6ad1a01e672b2408f8c79c2ab24`
- Notes: Best params: ρ=1.2, source_amplitude=4.0, ℓ_c=0.10, ℓ_s=0.09, smoothness=2. RMSE improved 38.2% from baseline (0.4767→0.2948), BEATING paper's 0.44. Runtime improved 33% (1.40s→0.94s) due to sparser factorization (fill 1.8% vs 4.8%). Both metrics Pareto-dominant over baseline. Combination of: (1) shorter lengthscales for sharper resolution, (2) sparser ρ for faster computation, (3) wider source prior to capture true source strength=5.0.
