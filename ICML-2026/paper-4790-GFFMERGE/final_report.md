# Final Report: paper-4790

- Title: GFFMERGE: Efficient Merging of Graph Neural Force Fields and Beyond
- Primary metric: `test_Force_MAE_kcal_per_mol_per_angstrom` (lower)
- Records: 9
- Generated: 2026-07-12T07:41:34Z

## Best Result

- Iteration: 7
- Idea: PARAM-01 — Higher LR=2e-4 + Cosine LR + Grad Clip + Early Stop + 12 epochs
- Primary metric: 0.7131
- Commit: `67886ff58e9e06d1d507245f2e21a011bceead92`
- Notes: Higher learning rate (2e-4 vs 1e-4 baseline) with cosine annealing to 2e-6 gives best results. Force MAE -22.0% vs baseline, Energy MAE -16.0% vs baseline. Both metrics strongly improved.
