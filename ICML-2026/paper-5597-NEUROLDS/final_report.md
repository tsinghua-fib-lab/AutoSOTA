# Final Report: paper-5597

- Title: Neural Low-Discrepancy Sequences
- Primary metric: `D2_star` (lower)
- Records: 8
- Generated: 2026-07-14T20:30:04Z

## Best Result

- Iteration: 6
- Idea: 5597-1+5597-8+5597-5+5597-12 — Gradient Post-Processing on Best Training Model
- Primary metric: 0.00018795
- Commit: `b3a84d9ead152e0c9db7ef9811f51f294029ed96`
- Notes: Idea #1 applied to Iter 4 model (Adaptive Clip + Curriculum + Extended WSD). 200-step projected gradient descent refined points from D2_star=0.00021295 to 0.00018795 (-11.74% from training output, -15.11% vs original baseline 0.00022140). This is the best result so far. The gradient post-processing consistently delivers ~11.7% improvement regardless of base model quality, but better base models yield better final results.
