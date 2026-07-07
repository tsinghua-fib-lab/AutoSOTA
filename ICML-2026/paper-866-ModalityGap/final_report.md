# Final Report: paper-866

- Title: Respecting Modality Gap in Post-hoc Out-of-distribution Detection with Pre-trained Vision-Language Models
- Primary metric: `AUROC` (higher)
- Records: 13
- Generated: 2026-07-05T20:07:12Z

## Best Result

- Iteration: 10
- Idea: PARAM-1 — blend_factor=0.98 + prototype_lr=0.2
- Primary metric: 99.59
- Commit: `296833b459251ddec1309c7f60888c77881076b0`
- Notes: Parameter combo: blend_factor=0.98 (more visual trust) + prototype_lr=0.2 (faster adaptation). Significant improvement over baseline: AUROC 99.59 (vs 99.26, +0.33%), FPR95 1.01 (vs 2.51, -59.8%). Big gains on harder datasets: Textures 99.23/1.52 (was 98.67/4.04), Places 99.45/1.64 (was 98.98/3.85).
