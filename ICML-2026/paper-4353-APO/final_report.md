# Final Report: paper-4353

- Title: Active Policy Optimization for Individualized Dosing via Gradient Variance Minimization
- Primary metric: `policy_suboptimality` (lower)
- Records: 11
- Generated: 2026-07-11T11:30:16Z

## Best Result

- Iteration: 8
- Idea: IDEA-07v2 — normalize_x=True with fixed sampler model() routing
- Primary metric: 0.1129
- Commit: `e5c73e15abe3791c576342178bd7e240d7367c76`
- Notes: CODE: Fixed IDEA-07 by changing posterior=base_model(inputs) to posterior=model(inputs) in all samplers, then enabled normalize_x=True. Input normalization significantly improves GP predictions (initial MSE 2.39 vs 14.2). Combined with target_sample_size=64, cand_t_grid_size=20, and adaptive beta. Result: 0.1129 vs baseline 0.1382 (18.3% improvement). NEW BEST.
