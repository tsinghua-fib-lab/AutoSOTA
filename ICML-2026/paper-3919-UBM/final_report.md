# Final Report: paper-3919

- Title: Uncovering Bias Mechanisms in Observational Studies
- Primary metric: `rho_b1_A_detection` (higher)
- Records: 6
- Generated: 2026-07-11T02:01:40Z

## Best Result

- Iteration: 1
- Idea: CODE-05+CODE-04 — max_iter=5000 + tol=1e-6 + random sampling for correlation
- Primary metric: 0.955
- Commit: `84791c9d252c2fd24b3c85eb6a7c2a9f70d1d20f`
- Notes: IDEA-05: LogisticRegression(max_iter=5000, tol=1e-6). IDEA-04: random sampling instead of iloc[:n] for correlation. Y improved +4.5pp (0.38->0.425). A maintained. S improved slightly.
