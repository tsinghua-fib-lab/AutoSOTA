# Final Report: paper-3301

- Title: Manifold-Aware Perturbations for Constrained Generative Modeling
- Primary metric: `COV` (higher)
- Records: 7
- Generated: 2026-07-10T02:36:09Z

## Best Result

- Iteration: 4
- Idea: A6 — Sinusoidal time embedding (dim=32) replacing raw scalar time_concat
- Primary metric: 0.9281
- Commit: `545aefb02e37c25ddffd17da71ce0646aa32945d`
- Notes: Sinusoidal positional encoding (dim=32) replaces raw scalar time_concat. Combined with A3 (multi-step projection), A5 (EMA), A1 (cosine schedule). COV +0.0176 (+1.9%), JSD -39%, TVD -29% vs baseline. Slight TVD increase vs cosine-only (0.12570->0.12750) within tolerance. Sinusoidal provides fixed, frequency-rich temporal signal without learnable parameters.
