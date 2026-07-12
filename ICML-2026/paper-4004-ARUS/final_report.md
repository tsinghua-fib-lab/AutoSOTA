# Final Report: paper-4004

- Title: Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision-Language Models
- Primary metric: `CHAIRs` (lower)
- Records: 6
- Generated: 2026-07-11T13:18:13Z

## Best Result

- Iteration: 5
- Idea: PARAM-P2-12 — alpha_max sweep: A=25 optimal, CHAIRs=30.4 (-9.2 pts)
- Primary metric: 30.4
- Commit: `a95ab9b52e697fbfac093e00c56dbda2d947f3b2`
- Notes: Swept alpha_max=[15,25]. A=15: CHAIRs=46.8 (worse, too weak). A=25: CHAIRs=30.4, CHAIRi=18.6 — both improved vs baseline (39.6/22.0). Stronger steering (25 vs default 20) reduces hallucinations significantly. Downside: 2.3x longer generation time due to longer captions. Best result so far.
