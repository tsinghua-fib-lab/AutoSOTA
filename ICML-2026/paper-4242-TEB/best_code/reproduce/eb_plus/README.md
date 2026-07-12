# §4.4 — Adaptive mode selection (EB⁺)

The hybrid EB⁺ scheduler switches online between mixed batching and EB
based on the closed-form crossover criterion. Two sub-experiments:

- [`traffic/`](traffic/) — Table 4: concurrency sweep c ∈ {32, 512, 2048}.
- [`non_stationary/`](non_stationary/) — Table 5: distribution shift and
  concurrency shift scenarios.

See the per-subdirectory READMEs.
