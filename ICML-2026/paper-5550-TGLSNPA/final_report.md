# Final Report: paper-5550

- Title: Torus Graphs for Large Scale Neural Phase Analysis
- Primary metric: `Runtime` (lower)
- Records: 13
- Generated: 2026-07-13T13:31:07Z

## Best Result

- Iteration: 12
- Idea: IDEA-01+IDEA-05 — Batch 1024 + SGD Momentum + n_iter 250
- Primary metric: 1.67
- Commit: `cffe23423371cbcbd70fd22a9c71084046a30d02`
- Notes: IDEA-01 + IDEA-05 + IDEA-12 combined: batch_size=1024, n_iter=250, lr=0.024, sgd_momentum. Runtime 1.67s (-95% vs baseline 35.77s). Total samples: 250*1024=256K vs baseline 5000*128=640K. R2=0.974 above 0.95 guardrail. MSE=0.0131 within 0.015 guardrail. Massive speedup from combining large batch training, SGD momentum, and aggressive iteration reduction.
