# Final Report: paper-2417

- Title: Semantic-level Backdoor Attack against Text-to-Image Diffusion Models
- Primary metric: `ASR` (higher)
- Records: 13
- Generated: 2026-07-08T19:52:44Z

## Best Result

- Iteration: 9
- Idea: PARAM-05 — Constraint weight 0.05 (near-zero) + 1600 iters + EMA — BEST
- Primary metric: 98.0
- Commit: `72bc3c2c3e9a16be9bb717bcb007cd01ecba038e`
- Notes: Reduced constraint_loss_weight to 0.05. ASR=98.0% (maintains breakthrough). CLIPp improved to 25.21 (+1.38 from baseline 23.83, best yet). LPIPS=0.340 (within 0.35 guardrail). Near-zero constraint maximizes KV alignment while LPIPS stays within bounds.
