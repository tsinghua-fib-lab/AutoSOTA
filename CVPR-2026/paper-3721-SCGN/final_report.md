# Optimization Results: Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging

## Summary
- Total iterations: 12 (plus baseline)
- Best PSNR: **27.0843** (baseline: 26.5503, improvement: **+2.01%**)
- Best SSIM: **0.9750** (baseline: 0.9723, improvement: **+0.28%**)
- Best IoU: **0.7436** (baseline: 0.7330, improvement: **+1.45%**)
- Best commit: `8db0856`

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| PSNR | 26.5503 | 27.0843 | +0.534 (+2.01%) |
| SSIM | 0.9723 | 0.9750 | +0.0027 (+0.28%) |
| IoU | 0.7330 | 0.7436 | +0.0106 (+1.45%) |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| 4x Flip Ensemble (H+V flips) in Net.forward() | PSNR +0.51, SSIM +0.0027, IoU +0.0097 | Flip operations are exact pixel rearrangements — no interpolation artifacts |
| Weighted averaging (0.4 original + 0.2 each flip) | PSNR +0.024 over equal weights | Giving more weight to the original prediction reduces ensemble variance |

## What Worked
1. **Flip ensemble (H+V)**: The single most effective optimization. Averaging predictions from horizontally and vertically flipped inputs reduces noise variance without introducing interpolation artifacts. This is because flips are exact pixel-level operations — no bilinear/nearest-neighbor sampling needed.
2. **Weighted averaging**: Giving the original (un-flipped) prediction higher weight (0.4 vs 0.2 per flip) consistently outperformed equal weighting, suggesting the original orientation produces the most reliable predictions.
3. **Avoiding interpolation**: Any operation involving interpolation (rotation, scaling, multi-scale) degraded performance. The model's features are orientation- and scale-dependent, making exact pixel rearrangements superior.

## What Didn't Work
1. **Rotation TTA (8x via kornia.rotate)**: Bilinear interpolation artifacts hurt SSIM (-0.005) despite modest PSNR gain (+0.13)
2. **Median filter post-processing**: Destroyed TEM atomic features — PSNR dropped to 17.63 (-33%)
3. **Variance-matching calibration**: Amplified noise because noisy inputs have higher variance than clean outputs
4. **Mean bias correction**: Model is well-calibrated — forcing output mean to match input mean degraded all metrics
5. **Transpose ensemble**: Model is orientation-dependent — transpose fundamentally changes spatial layout
6. **Multi-scale ensemble (0.5x down/upsample)**: Bilinear interpolation at half-resolution destroyed quality (-2.35 PSNR)

## Key Findings
- TEM image features are extremely sensitive to any form of interpolation or filtering
- The flip ensemble is the only inference-time technique that preserved image structure while reducing noise
- Without access to training data (`tem_data4` with 1000 images is unavailable), retraining-based optimizations could not be attempted
- The target PSNR of 27.8778 could likely be achieved through retraining with improved loss functions (L1+SSIM), modern optimizers (AdamW+cosine annealing), and data augmentation

## Top Remaining Ideas (for future runs)
1. **Retraining with combined best practices**: AdamW + CosineAnnealing + L1+SSIM loss + data augmentation — expected +0.5-1.5 dB
2. **Extended training (200 epochs)**: Current 100 epochs is short by modern standards
3. **Increased model capacity**: ch=96 or ch=128 could help
4. **Frequency-domain loss**: Add FFT-domain supervision to leverage FBGW module
5. **Stochastic Weight Averaging**: Multi-checkpoint averaging during training
6. **GELU activations**: Replace ReLU for smoother gradients (requires retraining)
