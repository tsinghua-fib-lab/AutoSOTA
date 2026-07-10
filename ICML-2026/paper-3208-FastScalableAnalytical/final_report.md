# Final Report: paper-3208

- Title: Fast and Scalable Analytical Diffusion
- Primary metric: `MSE` (lower)
- Records: 11
- Generated: 2026-07-09T22:28:20Z

## Best Result

- Iteration: 7
- Idea: PARAM-mask_threshold — mask_threshold=0.035 optimal - MSE 0.001444 (-78.1%)
- Primary metric: 0.00144418
- Commit: `ce9d21287e9c457d09c9412e074fc696477bf55b`
- Notes: mask_threshold sweep 0.01-0.06 with 3-step cosine + k_max=5000 + temp=1.3. Optimal mask_threshold=0.035: MSE=0.001444 (-78.1% vs baseline 0.00658), r2=0.828 (+16.3%), Time=0.224s (-10.0% vs baseline 0.249s). mask_threshold too low (0.01) causes r2 collapse to 0.657. mask_threshold=0.04 gives similar MSE (0.001442) but slightly lower r2. The 0.035 threshold balances projection sparsity and quality.
