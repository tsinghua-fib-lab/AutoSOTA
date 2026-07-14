# Final Report: paper-5914

- Title: Finding Most Influential Sets
- Primary metric: `Wall-clock time (ms) median n=1e6 k=1e5` (lower)
- Records: 9
- Generated: 2026-07-14T00:38:49Z

## Best Result

- Iteration: 8
- Idea: CODE-08 — Micro-optimizations: restrict pointers, loop unrolling, software prefetch
- Primary metric: 108.02
- Commit: `53e03c7e47d7ce9483fdca30d54e5697f1c4fdd3`
- Notes: Added __restrict__ pointer qualifiers, #pragma GCC ivdep hints, -funroll-loops flag, and __builtin_prefetch in subset sum loop. Marginal improvement (108.02 vs 108.19 ms, -0.2%) — compiler already does excellent auto-vectorization with -O3 -march=native. Median: 108.02 ms (baseline: 278.04, -61.2% total). Algorithm correctness verified.
