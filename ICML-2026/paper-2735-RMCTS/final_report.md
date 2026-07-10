# Final Report: paper-2735

- Title: Recursive Monte-Carlo Tree Search
- Primary metric: `Mean Score` (higher)
- Records: 10
- Generated: 2026-07-09T17:00:04Z

## Best Result

- Iteration: 5
- Idea: ALGO-13 — Dirichlet noise counterproductive at C=0.75; no_noise achieves Score 8.31
- Primary metric: 8.31
- Commit: `7a1fb2754075622a6086d6f61b10cc34041ab19f`
- Notes: Tested Dirichlet noise at root (AlphaZero technique) at C=0.75. Results: no_noise→8.31★, alpha=0.3_frac=0.25→3.77 (severe degradation), alpha=0.03_frac=0.25→partial. Dirichlet noise adds exploration on top of already well-calibrated C=0.75 prior, disrupting the exploitation-exploration balance. Lower C already reduces exploration; adding Dirichlet noise reintroduces it counterproductively. The 8.31 score at C=0.75, tau=0.20 is the best observed so far (3 runs averaging ~8.0).
