# Final Report: paper-4552

- Title: LORD-GoF: A Robust Online Detection Approach for LLM Watermarks in Sparse and Mixed Streams
- Primary metric: `FDR` (lower)
- Records: 9
- Generated: 2026-07-11T14:25:04Z

## Best Result

- Iteration: 6
- Idea: IDEALIB-4552-012 — Hyperparameter tuning: W0=0.03 gamma=1.05
- Primary metric: 0.0
- Commit: `1650c8811db853ddad0209f929f57e171bda097c`
- Notes: Sweep found W0=0.03, GAMMA_EXP=1.05 gives FDR=0.0, Power=1.0 at tau=0.5. Improved ALL GoF stats: Lord-And 0.000/1.0, Lord-Cra 0.000/0.963, Lord-Chi 0.000/0.944, Lord-Ney 0.000/0.704. Also zero FDR at tau=0.7 and 0.9.
