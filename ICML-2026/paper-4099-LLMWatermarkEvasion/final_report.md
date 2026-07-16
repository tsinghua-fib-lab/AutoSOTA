# Final Report: paper-4099

- Title: LLM Watermark Evasion via Bias Inversion
- Primary metric: `Attack Success Rate` (higher)
- Records: 4
- Generated: 2026-07-16T05:28:49Z

## Best Result

- Iteration: 2
- Idea: ALGO-1+CODE-3+CODE-1 — ALGO-1 two-pass re-attack + CODE-3 rep_penalty + beta=-5.5
- Primary metric: 100.0
- Commit: `72eb7ba08ad6adffe7fe6463f9e87a6ea4982fd3`
- Notes: ALGO-1 two-pass re-attack triggered on 4 samples, succeeded on 3. Combined with CODE-3 repetition_penalty=1.1 and CODE-1 beta=-5.5. ASR=100% (perfect evasion), TPR@1%=0.006, TPR@10%=0.022, Best F1=0.666. All guardrails improved vs baseline. Runtime: ~90 min.
