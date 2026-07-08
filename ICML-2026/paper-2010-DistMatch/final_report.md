# Final Report: paper-2010

- Title: DistMatch: Adaptive Binning  via Distribution Matching for Robust Sequential Conformal Prediction
- Primary metric: `winkler_score_norm` (lower)
- Records: 15
- Generated: 2026-07-07T18:27:48Z

## Best Result

- Iteration: 8
- Idea: ALGO-02 — Winkler CV adaptive window: selected w=150
- Primary metric: 1.6288
- Commit: `a30c877f46154a85d93aa974974d581a64830e4c`
- Notes: ALGO-02 SUCCESS: Adaptive window selection via in-sample Winkler CV chose w=150 (vs default w=100). Win.=1.6288 (-2.6% vs baseline 1.6725), Cov.=0.9256 (above 0.90 guardrail), Width=65.61 (-5.2% vs baseline 69.23). w=150 provides longer context for residual distribution matching, producing better tree splits. 6 candidates tested: w=50(0.59), 75(0.64), 100(0.62), 125(0.67), 150(0.48), 200(0.66).
