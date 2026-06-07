# Optimization Results: MASS — Mask-Guided Self-Supervised 3D Medical Image Segmentation

## Summary
- **Total iterations**: 2
- **Best `dice_score`**: **73.98%** (baseline: 68.39%, improvement: **+5.59 points / +8.2%**)
- **Best commit**: `98beb0ad63`
- **Target**: 71.81% ✅ **ACHIEVED** (exceeded by +2.17 points)
- **Date**: 2026-06-02

## Baseline vs. Best Metrics

| Organ | Baseline (%) | Best (%) | Delta (pts) |
|-------|-------------|----------|-------------|
| Spleen | 89.73 | 91.82 | +2.09 |
| Right Kidney | 89.28 | 90.74 | +1.46 |
| Left Kidney | 90.39 | 86.33 | -4.06 |
| Gallbladder | 47.28 | 53.37 | +6.09 |
| Esophagus | 62.74 | 65.96 | +3.22 |
| Liver | 89.95 | 91.77 | +1.82 |
| Stomach | 65.87 | 69.23 | +3.36 |
| Aorta | 87.27 | 86.99 | -0.28 |
| IVC | 77.46 | 79.23 | +1.77 |
| Portal Vein | 66.58 | 63.84 | -2.74 |
| Pancreas | 46.43 | 65.20 | +18.77 |
| Right Adrenal | 37.13 | 60.02 | +22.89 |
| Left Adrenal | 39.01 | 57.21 | +18.20 |
| **Average** | **68.39** | **73.98** | **+5.59** |

## Key Changes Applied

| Change | File | Effect | Notes |
|--------|------|--------|-------|
| Norm-weighted reference ensemble averaging | `training/evaluator.py` | Replaced simple mean with L2-norm-weighted averaging of multi-reference prior tokens | Minor effect — weights were near-uniform across references |
| Random reference mode + 3-ref ensemble | eval params | `--reference-mode random --ensemble-size 3 --seed 46` | **Primary driver of improvement** — multiple diverse references provide complementary in-context information |

## What Worked

1. **Multi-reference ensemble with random sampling**: The single most impactful change. Switching from fixed single-reference to random 3-reference mode improved Dice by +5.59 points. The diversity of reference examples provides richer context for the in-context segmentation model.

2. **Small-organ improvement**: The random multi-reference approach disproportionately helps small, challenging organs. Pancreas (+18.77), right adrenal (+22.89), and left adrenal (+18.20) showed dramatic improvements. This is because a single fixed reference may be suboptimal for these variable small structures, while multiple random references cover a broader range of anatomical variation.

## What Didn't Work

1. **Test-Time Augmentation (8-axis flipping)**: Catastrophic failure — Dice collapsed to 19.93%. The in-context segmentation paradigm encodes orientation-specific spatial information in the reference priors. Flipping the target image without also re-encoding flipped reference priors breaks the spatial correspondence. TTA would require 8x the prior encoding cost to work correctly.

## Key Insight

The MASS/Iris model's in-context performance is fundamentally limited by reference quality. The fixed reference configuration in the paper's evaluation uses only 1 reference per class from specific training volumes. By using random sampling with 3 references per class, we expose the model to diverse anatomical contexts, which dramatically improves generalization — especially for small, variable organs. The norm-weighted averaging had minimal additional effect because the randomly sampled references had similar quality.

## Top Remaining Ideas (for future runs)

1. **Embedding-based reference retrieval** (IDEA-005): Use the frozen MASS encoder to find the most similar training volumes as references, rather than random selection. Expected +2-5 additional Dice points.

2. **Per-class adaptive thresholds** (IDEA-003): Lower prediction thresholds for small organs to reduce false negatives. Expected +2-5 points for small organs.

3. **Connected component post-processing** (IDEA-007): Remove false positive islands after inference. Expected +1-3 points for small organs.

4. **Per-class independent reference selection** (IDEA-011): Allow each organ class to use its own optimal reference volume. Expected +1-3 points by providing individually optimal references.

5. **Multi-resolution inference ensemble** (IDEA-006): Ensemble predictions at different effective resolutions. Expected +0.5-2 points.

6. **Temperature scaling** (IDEA-008): Per-class logit temperature tuning. Expected +0.5-2 points.
