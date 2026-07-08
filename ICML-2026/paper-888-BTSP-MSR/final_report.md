# Final Report: paper-888

- Title: Bottleneck Communication Delay Minimization for Communication-Efficient Decentralized Learning
- Primary metric: `test_accuracy` (higher)
- Records: 7
- Generated: 2026-07-06T19:17:07Z

## Best Result

- Iteration: 6
- Idea: COMBINED-MCSS-500 — M+C+S+Stratified at 500 rounds
- Primary metric: 0.4455
- Commit: `5de9eb8f68a5d42e6f7225305128c8e772a0e602`
- Notes: Best configuration at 500 rounds: 0.4455 (3.1x baseline 0.1438). Already 65% above the 5000-round baseline of 0.2707. Convergence trajectory: 200r:0.365, 300r:0.403, 400r:0.430, 500r:0.446. Improvement rate decaying but still significant (+0.009 per 50 rounds at round 450).
