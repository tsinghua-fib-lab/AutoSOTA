# Final Report: paper-5623

- Title: Cheap2Rich: A Multi-Fidelity Framework for Data Assimilation and System Identification of Multiscale Physics - Rotating Detonation Engines
- Primary metric: `RMSE` (lower)
- Records: 12
- Generated: 2026-07-14T04:00:21Z

## Best Result

- Iteration: 10
- Idea: ALGO-002+PARAM-001 — SSIM loss + HF lr=3e-4
- Primary metric: 0.098086
- Commit: `d2d6955cfc100609cbd43a695eb956b29f9059e7`
- Notes: SSIM λ=0.01/0.005 + HF lr=3e-4. Best RMSE yet: 0.0981 (-4.8% vs baseline 0.103). SSIM 0.376 (-0.2% vs iter-9 best). Lower HF lr further reduces overfitting.
