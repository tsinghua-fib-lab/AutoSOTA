# Final Report: paper-2669

- Title: Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search
- Primary metric: `mean_frequency` (higher)
- Records: 8
- Generated: 2026-07-09T11:44:47Z

## Best Result

- Iteration: 7
- Idea: 2669-004 — clf_alpha=20, temp=2.0, max_len=5 -- BEST
- Primary metric: 0.6263
- Commit: `43b1c238523d0bd7117336ab06ea7f3afdae73c5`
- Notes: BEST CONFIG: Fixed temperature=1000 bug + clf_alpha=20 + sampling_temp=2.0 + max_length=5 + sd_batch=5. mean_frequency +12.9% vs baseline, perplexity -22.1%. Best trade-off across all three metrics.
