# Interactive per-token explorer is static and pre-computed

The project page is served by GitHub Pages, which is static hosting with no
backend and no GPU at serve time. We therefore ship the per-token explorer as a
client-side widget that reads **pre-exported JSON** (token strings, per-token
surprisal, per-token steps-to-converge, and the converged horizon for a handful of
sample/model pairs), rather than running any model in the browser or calling a
service. The trade-off: the explorer can only show the trajectories we exported,
not arbitrary user input — accepted because a live model is infeasible on Pages and
the export is small and reproducible.

## Consequences

- A one-off **export step** is required before/at build: steps-to-converge comes
  free from each run's `progressive_prefixes` dataset, but **per-token surprisal is
  not cached** and needs one frozen forward pass per featured sample (forward-only,
  small GPU job — see `scripts/analyze_surprisal_vs_steps.py`).
- The committed JSON payload must stay small; cap featured samples/models
  accordingly.
