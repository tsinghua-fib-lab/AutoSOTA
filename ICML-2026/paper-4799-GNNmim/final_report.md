# Final Report: paper-4799

- Title: Rethinking GNNs and Missing Features: Challenges, Evaluation and a Robust Solution
- Primary metric: `F1_Score` (higher)
- Records: 13
- Generated: 2026-07-12T06:37:44Z

## Best Result

- Iteration: 12
- Idea: IDEA-08 — Per-node loss weighting + residual + focal loss
- Primary metric: 0.7929
- Commit: `c6f378f87e83396e3a33cfe0a30ec1f4ad1d1c62`
- Notes: Added per-node loss weighting based on feature completeness (weight = 0.5 + 0.5 * completeness) combined with residual connections, focal loss (gamma=1.5), and extended training (2000 epochs). F1=0.7929 — best result! +5.6% over baseline (0.7508), +5.6% relative improvement. Nodes with more observed features get higher loss weight (up to 1.0x), while nodes with mostly missing features get reduced weight (min 0.5x). This biases training toward reliable examples while maintaining signal from challenging ones. This is the new best candidate.
