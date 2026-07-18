# Final Report: paper-3612

- Title: SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs
- Primary metric: `time-to-train/T0` (lower)
- Records: 13
- Generated: 2026-07-18T05:46:42Z

## Best Result

- Iteration: 12
- Idea: IDEA-01 — Grid sweep: r=9 ckpt=14
- Primary metric: 2.9074
- Commit: `83428d706daac99fddc2358215b3674880fe6db0`
- Notes: IDEA-01. r=9, ckpt=14. Best result overall. time-to-train -1.3% vs baseline (2.946->2.907), Availability slightly improved (87.22->87.41). Better trade-off than paper optimum at ckpt=15. More frequent checkpointing at ckpt=14 reduces rework without excessive overhead. NEW BEST.
