# Optimization Results: HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models

## Summary
- **Total iterations**: 8 (+ final)
- **Best `avg_accuracy`**: 80.41% (self-consistency) / 80.25% (flip augmentation, reproducible)
- **Baseline**: 79.62%
- **Improvement**: +0.63% to +0.79%
- **Target**: 83.77% — **NOT reached**
- **Best commit**: `06477b0` (iter-5, self-consistency) / `0c2f9e7` (iter-1, flip augmentation)

## Baseline vs. Best Metrics

| Metric | Paper Baseline | Our Baseline | Best (Iter 1/5) | Delta |
|--------|---------------|-------------|-----------------|-------|
| Avg Accuracy | 79.78% | 79.62% | 80.25% / 80.41% | +0.63% / +0.79% |
| Width Accuracy | 69.92% | 69.92% | 69.92% | 0% |
| Height Accuracy | 85.50% | 85.50% | 86.26% | +0.76% |
| Direct Distance Accuracy | 80.27% | 80.27% | 79.59% / 82.31% | -0.68% / +2.04% |
| Horizontal Distance Accuracy | 86.07% | 86.07% | 87.70% | +1.63% |
| Vertical Distance Accuracy | 77.14% | 76.19% | 78.10% / 77.14% | +1.91% / +0.95% |

## Key Changes Applied

| Iter | Change | Effect | Notes |
|------|--------|--------|-------|
| 1 | Flip Augmentation XYZ Averaging | +0.63% avg | Best single change. Run MoGe on original + flipped image, average XYZ. |
| 2 | Increase max_new_tokens 100→200 | 0% | No effect — model outputs short answers under 100 tokens. |
| 3 | Per-Category Prompt Hints | -1.73% | Harmful. "Output only the number" confused the model. |
| 4 | Depth Anomaly Filtering | 0% | No effect — MoGe depth is already clean enough. |
| 5 | Self-Consistency N=5 (temp=0.5) | +0.16% | Marginal. Direct distance +2.72pp but width -1.5pp. |
| 6 | CoT Prefix | 0% | No effect. Simple spatial reasoning hint doesn't help. |
| 7 | Reference Object Calibration (LEAP) | -4.62% | Major regression. Explicit object sizes confused the model. |
| 8 | Simplified Reference Context (HP1) | -0.50% | Partial recovery but below best. |

## What Worked

1. **Flip Augmentation XYZ Averaging**: The only robust improvement (+0.63%). Running MoGe on both original and horizontally flipped images and averaging the XYZ point clouds reduces monocular depth estimation noise. This is a direct application of test-time augmentation principles.

2. **Self-Consistency Decoding**: Marginal improvement (+0.16%) at 5x inference cost. Generating 5 samples with temperature=0.5 and taking the median improved direct distance by 2.72pp but degraded width estimation.

## What Didn't Work

1. **Prompt Engineering (Category Hints, CoT, Reference Objects)**: ALL prompt modifications either had no effect or caused severe regressions. The HiSpatial model was trained with a specific prompt format, and deviating from it confuses the model. This is consistent with findings that naive prompt modification harms spatial reasoning.

2. **Parameter Tuning (max_new_tokens)**: Increasing token budget from 100 to 200 had zero effect because the model outputs concise answers well under 100 tokens.

3. **Post-Processing (Depth Filtering)**: Median-based anomaly filtering of depth estimates had no effect because MoGe produces sufficiently clean depth for the VLM.

4. **Reference Object Calibration**: Explicitly providing typical object sizes as scale references severely degraded performance (-4.62pp). The model couldn't effectively use this information.

## Root Cause Analysis

The fundamental limitation is that **HiSpatial's spatial reasoning accuracy is determined by its training, not by inference-time parameters**. Inference-time optimizations can only make marginal improvements because:

1. The model learns a specific mapping from XYZ coordinates to spatial answers during training
2. Changing the XYZ preprocessing (flip augmentation) helps by reducing depth noise, but the effect is limited by the model's trained tolerance
3. The model's prompt processing is calibrated to its training format — prompt modifications cause distribution shift
4. Width estimation (69.92%) is fundamentally limited by monocular depth ambiguity in the lateral dimension

## Top Remaining Ideas (for future runs)

1. **Multi-Scale Depth Fusion**: Run MoGe at multiple image scales (336, 448, 560), fuse XYZ by averaging. Could provide better depth than single-scale + flip.
2. **MoGe Model Ensemble**: Combine MoGe ViT-L with DepthAnything V2 or MetricAnything for complementary depth estimation.
3. **Training-Time Improvements**: Fine-tune with width-focused data augmentation, curriculum learning, or spatial reward functions (SpatialThinker approach).
4. **Architecture Changes**: Add attention-based depth confidence weighting in the XYZ-to-VLM pipeline.
5. **Higher Resolution**: Use PaliGemma 2 896px variant with checkpoint surgery — requires retraining but could improve fine-grained width estimation.
