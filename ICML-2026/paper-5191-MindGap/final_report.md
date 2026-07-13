# Final Report: paper-5191

- Title: Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy
- Primary metric: `MG_in_sample_error` (lower)
- Records: 13
- Generated: 2026-07-13T04:40:14Z

## Best Result

- Iteration: 3
- Idea: ALGO-04 — Iteration-dependent noise annealing with cosine schedule
- Primary metric: 13.08
- Commit: `a3bcc2717a79de806a895b4da797f5010d9feb86`
- Notes: Cosine noise annealing from 0.0385 to 0.030 over T=100 iterations. MG improved massively: 15.34->13.08% (-2.26pp vs baseline). AG: 15.44->13.40%. QG: 15.64->13.39%. PCD stable at 6.76%. All mechanisms benefited from reduced late-iteration noise.
