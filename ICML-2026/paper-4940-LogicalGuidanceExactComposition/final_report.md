# Final Report: paper-4940

- Title: Logical Guidance for the Exact Composition of Diffusion Models
- Primary metric: `Conformity_Score_N2` (higher)
- Records: 9
- Generated: 2026-07-12T21:52:36Z

## Best Result

- Iteration: 1
- Idea: ALGO-01 — Guidance Annealing α=0.5
- Primary metric: 96.01
- Commit: `a851528964361928788f5e36268f1c998da575de`
- Notes: 时间依赖的引导退火策略（Guidance Annealing α=0.5），在去噪过程中动态调整引导强度。Conformity_Score_N2 从基线 92.04% 提升至 96.01%（+3.97pp, +4.31%），Joint_Entropy 保持在 3.5 警戒线上方（3.57），logdiff_avg_batch_time 无明显回归（+0.7%）。基线完全复现论文指标（92.04% vs 论文 93.8%，在 CI 范围内）。iter-2（α=0.25, +2.60pp）、iter-6（负引导 + 退火, +3.55pp）、iter-7（激进退火 0.3-2.0x, +3.19pp）均有提升但不如 iter-1。iter-8（激进退火 + 负引导）触发 guardrail 失败被排除。
