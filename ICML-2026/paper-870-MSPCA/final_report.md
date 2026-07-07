# Final Report: paper-870

- Title: Mean-Shift PCA by Knockoff Mean
- Primary metric: `Alignment` (higher)
- Records: 9
- Generated: 2026-07-05T16:59:08Z

## Best Result

- Iteration: final
- Idea: BEST — Best state verification (50 trials)
- Primary metric: 99.54
- Commit: `4e172d1cd92f6e581b9babae8dc92c86c70596c8`
- Notes: Final verification of best state (iter-4: svds + max_k_r=3 + early-break). 50-trial eval at pi_1=0.15: ALIGNMENT=99.54%, RUNTIME=23.88ms. Confirms +1.86pp improvement over baseline (97.68%). Pareto dominant on both metrics.
