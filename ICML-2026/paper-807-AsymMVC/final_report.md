# Final Report: paper-807

- Title: Asymmetric Multi-View Clustering with Hyperbolic Uncertainty Modeling
- Primary metric: `ACC` (higher)
- Records: 13
- Generated: 2026-07-05T06:57:37Z

## Best Result

- Iteration: 4
- Idea: ALGO-04 — Quality-Weighted Alignment (alpha0*alpha1 gating)
- Primary metric: 88.17
- Commit: `1122220af83e1b6bbd666c8c619b4b4fbd0a4420`
- Notes: ALGO-04 + CODE-02: Weighted alignment loss by α0*α1 (quality gating weights). Down-weights uncertain views during alignment. ACC=88.17 (+1.34 vs baseline 86.83), NMI=82.30 (+1.90), ARI=76.75 (+2.48). All three metrics improved significantly. Exceeds paper reported ACC of 87.5.
