# Final Report: paper-307

- Title: Test-Time Detoxification without Training or Learning Anything
- Primary metric: `Max_Toxicity` (lower)
- Records: 7
- Generated: 2026-07-04T22:25:25Z

## Best Result

- Iteration: 4
- Idea: ALGO-02+ALGO-04 — Momentum + Adaptive Stepsize on 50-prompts
- Primary metric: 0.07742
- Commit: `88dfa42db522b4752d71d5f2a8c4a48114e183ea`
- Notes: ALGO-02+ALGO-04: Adam momentum (beta1=0.9, beta2=0.999) + adaptive stepsize. 50 prompts. Max Toxicity -53.3% (0.077 vs baseline 0.166), Mean Toxicity -35.9% (0.057 vs 0.089), Toxic Rate 0%. Perplexity 4.704 (preserved vs baseline 4.706). Convergence 14.5x faster (0.21 vs 3.04 avg iter). Better than momentum-only: lower toxicity + better perplexity!
