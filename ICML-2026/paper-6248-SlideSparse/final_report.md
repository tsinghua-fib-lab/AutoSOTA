# Final Report: paper-6248

- Title: SlideSparse: Fast and Flexible (2N-2):2N Structured Sparsity
- Primary metric: `Accuracy` (higher)
- Records: 6
- Generated: 2026-07-16T13:48:49Z

## Best Result

- Iteration: 3
- Idea: PARAM-01 — Wanda 6:8 with 256 calibration samples
- Primary metric: 71.29
- Commit: `0174a8ccbfaeef08366d5e6f8912c76fd5b62410`
- Notes: PARAM-01: Doubled calibration samples from 128 to 256. PPL essentially unchanged (8.12 vs 8.13 baseline) but MMLU improved +0.22pp (71.29% vs 71.07%). Required fixing hardcoded nsamples=128 in prepare_calibration_input. BEST result so far.
