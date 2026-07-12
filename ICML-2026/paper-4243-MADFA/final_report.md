# Final Report: paper-4243

- Title: Mechanistic Anomaly Detection via Functional Attribution
- Primary metric: `CLC_AUROC` (higher)
- Records: 7
- Generated: 2026-07-12T02:06:49Z

## Best Result

- Iteration: 2
- Idea: ALGO-02 — Cosine LR schedule for SGLD (1e-5 -> 1e-7)
- Primary metric: 1.0
- Commit: `4623c27a3dbce51cc845dc1502ad959b3623536e`
- Notes: Cosine LR schedule from 1e-5 to 1e-7 over 2000 draws. Dramatic improvement in Mean_Corr (+0.0579) and Mean_CCC (+0.0337). CLC maintained at ceiling 1.0. Metrics parsed from analyze.py stdout.
