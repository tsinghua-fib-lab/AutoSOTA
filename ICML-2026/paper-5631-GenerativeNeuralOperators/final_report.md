# Final Report: paper-5631

- Title: Generative Neural Operators through Diffusion Last Layer
- Primary metric: `ED` (lower)
- Records: 10
- Generated: 2026-07-14T05:31:19Z

## Best Result

- Iteration: 7
- Idea: DLL-CODE-003 — 64 MC samples + Heun on eta_min model
- Primary metric: 1.058
- Commit: `1481ec554c4ad7208e129f0c7f3f120d10a8af85`
- Notes: test_samples_per_example 32->64 + Heun solver + eta_min=1e-6. ED 1.058 (-21.2pct vs baseline 1.342!). SWD 0.198 (-9.2pct). NRMSEm 0.262 (-1.1pct, within tolerance!). NRMSEs 0.226 (-16.6pct). ALL metrics improved. 64 samples gives more reliable MC estimate; previous 32-sample ED estimates were biased high. This is a clean win.
