# Final Report: paper-3377

- Title: Modeling temporal scRNA-seq data with latent Gaussian process and optimal transport
- Primary metric: `W2_t8` (lower)
- Records: 7
- Generated: 2026-07-10T08:48:37Z

## Best Result

- Iteration: 5
- Idea: ALGO-5 — Deeper decoder [128,128,64] + LayerNorm
- Primary metric: 30.44
- Commit: `9a385795cc00bd3a81cd5590914517e589a96774`
- Notes: FIRST IMPROVEMENT! W2_t8 improved from 30.74 to 30.44 (-0.98%%). All three metrics better: W2_t4 -0.08%%, W2_t6 -0.35%%. Deeper decoder with LayerNorm improved reconstruction quality. Training 311s (312 epochs). Validation OT 427.
