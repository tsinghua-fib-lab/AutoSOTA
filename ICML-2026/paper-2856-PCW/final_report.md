# Final Report: paper-2856

- Title: Beyond Heuristic Tuning: Power-Calibrated LLM Watermarking
- Primary metric: `-log(DKL)` (higher)
- Records: 11
- Generated: 2026-07-09T06:50:34Z

## Best Result

- Iteration: 9
- Idea: ALGO-2-entropy-0.1-delta-4.83 — Entropy-gated (0.1) with reduced delta=4.83
- Primary metric: 3.033
- Commit: `c3e5e1ef6842f1d613dffeac73d352e237d84edd`
- Notes: ALGO-2 Entropy-Gated with delta lowered from 5.83 to 4.83. -log(DKL)=3.033 (+11.6% vs baseline, +1.1% vs delta=5.83 with same entropy threshold). TPR=0.98 maintained. Lower delta means less distortion per watermarked token; entropy gating compensates for detection by focusing signal on high-entropy positions.
