# Final Report: paper-4082

- Title: GenUnfold: Rapidly Predict Protein Mechanical Unfolding Trajectory via a Physics-Guided Diffusion Model
- Primary metric: `FID` (lower)
- Records: 7
- Generated: 2026-07-11T03:59:09Z

## Best Result

- Iteration: 7
- Idea: ALGO-7d — Maximum bandpass enhancement [5,60) sigma=0.008
- Primary metric: 0.0084
- Commit: `ac4307dd5ece5beabc1d34fd3ffbd4915e432acf`
- Notes: Maximum safe bandpass [5,60) with sigma=0.008. FID improved 20.9% (0.0106->0.0084). Force-JSD +4.9% (at tolerance boundary). Rel_l2 +1.7%. All guardrails within 5% tolerance. Further sigma increase would risk Force-JSD violation. Best result: 20.9% FID reduction.
