# Final Report: paper-3078

- Title: DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention
- Primary metric: `Accuracy` (higher)
- Records: 3
- Generated: 2026-07-10T07:30:57Z

## Best Result

- Iteration: 2
- Idea: PARAM-01 — Higher cosine similarity threshold (tau=0.999)
- Primary metric: 0.7938
- Commit: `a7abf438a227263cae6b349e5c3de46f3f0ab89c`
- Notes: Raised threshold from 0.9975 to 0.999. More tokens get full FFN recomputation per step, improving accuracy. Accuracy 79.38% (+1.44% vs baseline 77.94%, beats paper 79.30%). Throughput 46.69 t/s (-6.3% vs baseline 49.85, within 10% guardrail). Pareto-improved accuracy at acceptable throughput cost.
