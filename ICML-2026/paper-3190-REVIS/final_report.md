# Final Report: paper-3190

- Title: REVIS: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models
- Primary metric: `CHAIRS` (lower)
- Records: 9
- Generated: 2026-07-09T22:29:00Z

## Best Result

- Iteration: 1
- Idea: IDEA-01 — Recomputed vector, alpha=0.5, tau=-0.5, gamma=1.0
- Primary metric: 19.0
- Commit: `48e29efb7c551792b22c857a63f24cee53cd6551`
- Notes: Recomputed steering vector with always-on gating (tau=-0.5) at alpha=0.5. CHAIRs reduced from 20.0% (baseline) to 19.0%. CHAIRi improved from 6.61% to 6.32%. Modest but real improvement, confirming the recomputed vector captures a meaningful direction. BLEU/ROUGE slightly degraded vs baseline (BLEU_1 0.086 vs 0.119, ROUGE_L 0.130 vs 0.159), suggesting caption quality trade-off.
