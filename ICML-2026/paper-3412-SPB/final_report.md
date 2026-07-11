# Final Report: paper-3412

- Title: Symmetries in PAC-Bayesian Learning
- Primary metric: `PAC-Bayesian Bound` (lower)
- Records: 13
- Generated: 2026-07-11T03:19:33Z

## Best Result

- Iteration: 9
- Idea: IDEA-05+IDEA-10+IDEA-01 — KL-Reg lambda=0.1 + Cosine + sigma=0.04
- Primary metric: 0.347
- Commit: `67d55f0450b906e6d0ffdc36a6a5396009ea9de7`
- Notes: Combined KL regularization (lambda=0.1) + cosine annealing (lr=1e-3) + optimal sigma=0.04. KL -61.6pct (2830 vs 7369), bound -32.7pct (0.347 vs 0.515), test risk -25.9pct (0.159 vs 0.215). ALL metrics substantially improved over baseline. Sigma=0.04 is the sweet spot: lower KL than sigma=0.05 while Gibbs risk stays low due to smaller weight perturbation.
