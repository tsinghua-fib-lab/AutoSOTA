# Final Report: paper-2428

- Title: AdaMeZO: Adam-style Zeroth-Order Optimizer for LLM Fine-tuning Without Maintaining the Moments
- Primary metric: `Accuracy` (higher)
- Records: 3
- Generated: 2026-07-09T01:26:29Z

## Best Result

- Iteration: 1
- Idea: IDEA-01 — R-AdaZO variance-reduced second moment
- Primary metric: 59.39
- Commit: `2a1dc585530f9a965d1efc1f2739bb245686232b`
- Notes: R-AdaZO: replaced temp_hess (proj^2 * z^2 accumulation) with squared EMA first moment (temp_grad^2). Sign-based update on EMA first moment. Mean 59.39% (+5.89 vs baseline). Seeds: 13=57.76%, 21=63.18%, 42=56.32%, 87=60.29%. Best seed 21 exceeds paper 63.1%.
