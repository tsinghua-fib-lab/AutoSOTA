# Final Report: paper-5224

- Title: Learning to Execute Graph Algorithms Exactly with Graph Neural Networks
- Primary metric: `case_accuracy` (higher)
- Records: 13
- Generated: 2026-07-13T05:48:47Z

## Best Result

- Iteration: 12
- Idea: I-08+I-06 — Float32 + 3500 epochs + hidden_dim=k*2000 (re-verify, BEST)
- Primary metric: 1.0
- Commit: `c4c7333c5d924157153adaa9cb2b1699b6c58331`
- Notes: Re-verification of float32+3500+k*2000 config. 22 models (57% reduction from baseline 51). Total training cost: 22*3500=77K epoch-models vs baseline 51*7000=357K (78% reduction). Float32 improves model diversity; half epochs preserves stochasticity. Most robust and efficient configuration found.
