# Final Report: paper-5149

- Title: Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models
- Primary metric: `Acc` (higher)
- Records: 7
- Generated: 2026-07-13T20:41:51Z

## Best Result

- Iteration: 5
- Idea: IDEA-2 — IDEA-2: state_noise_mixing=0.2 (LIMIT=200)
- Primary metric: 42.5
- Commit: `a944d42b5542ebfd809c1de89e4be8f0db18a75e`
- Notes: Mild noise injection (beta=0.2) improves Acc by +4pp over baseline (42.5%% vs 38.5%%). Faster (10.7s vs 11.5s/sample). Paper-validated mechanism. Strict-match also improved (32.5%% vs 30%%).
