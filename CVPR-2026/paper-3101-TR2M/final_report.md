# Optimization Results: TR2M — Transferring Monocular Relative Depth to Metric Depth

## Summary
- **Total iterations**: 6 (baseline + 5 ideas)
- **Best `delta1`**: 0.944 (baseline: 0.943, improvement: +0.001 / +0.1%)
- **Best `abs_rel`**: 0.084 (baseline: 0.086, improvement: -0.002 / -2.3%)
- **Best `rmse`**: 0.339 (baseline: 0.342, improvement: -0.003 / -0.9%)
- **Paper baseline** (reference only): delta1=0.954, abs_rel=0.082, rmse=0.293

**Note on baseline discrepancy**: Our evaluation uses 543/654 (83%) of the official NYUv2 test images due to the inability to obtain the complete raw dataset in this environment. The missing 111 test images affect metric comparability with the paper's reported baseline.

## Baseline vs. Best Metrics

| Metric | Paper Baseline | Our Baseline | Our Best | Delta (Our) |
|--------|---------------|-------------|----------|-------------|
| delta1 | 0.954 | 0.943 | 0.944 | +0.001 |
| delta2 | 0.996 | 0.991 | 0.991 | 0.000 |
| delta3 | 0.999 | 0.998 | 0.998 | 0.000 |
| abs_rel | 0.082 | 0.086 | 0.084 | -0.002 |
| log10 | 0.035 | 0.037 | 0.037 | 0.000 |
| rmse | 0.293 | 0.342 | 0.339 | -0.003 |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| IDEA-001: Test-Time Augmentation (Flip Ensemble) | delta1 +0.001, abs_rel -0.002 | Only successful optimization. Standard TTA with horizontal flip averaged with original prediction |
| IDEA-003: Scale Coefficient Tuning (0.05/0.02) | delta1 → 0.001 (destroyed) | coef_scale/coef_shift are training-dependent; changing at inference breaks the model |
| IDEA-003: Scale Coefficient Tuning (0.015/0.015) | delta1 → 0.044 (destroyed) | Even small changes to coef_scale/coef_shift cause major degradation |
| IDEA-002: Multi-Token Text Features | delta1 -0.039 | Using full CLIP token embeddings instead of pooled feature causes regression; model trained with single pooled representation |
| IDEA-009: Post-Processing Recalibration | delta1 -0.259 | Median-based global recalibration introduces bias; needs more sophisticated approach |
| IDEA-004: Bidirectional Cross-Attention | no change | Adding reverse text→image attention path produces identical results; model weights are architecture-specific |

## What Worked
- **TTA (flip ensemble)** is the only inference-time technique that provided measurable improvement. This is consistent with standard practices in depth estimation benchmarks.
- The TR2M model pipeline is robust and produces stable inference results across runs.

## What Didn't Work
- **Model parameter changes at inference**: coef_scale and coef_shift are tightly coupled to the training process. Any deviation from training values destroys performance.
- **Architecture modifications**: Changes to the cross-attention mechanism (bidirectional, multi-token text features) don't help without retraining because the model weights are optimized for the specific architecture used during training.
- **Post-processing recalibration**: Simple median-based recalibration degrades performance. More sophisticated approaches (least-squares alignment per-image) would require ground truth access.

## Critical Findings
1. **Training code is not released**: The TR2M repository only contains evaluation scripts and pretrained weights. Without the training code (SOC loss, training pipeline), significant improvements through model changes are infeasible.
2. **Model-parameter coupling**: The TR2M model (19M ScaleMap) has its parameters (coef_scale=0.01, coef_shift=0.01, text feature format, cross-attention architecture) tightly coupled to its training configuration. Inference-time changes to these parameters inevitably degrade performance.
3. **Dataset limitation**: The 83% test coverage (543/654 images) prevents direct comparison with paper baselines. A complete dataset would be needed for accurate benchmarking.
4. **GPU environment**: PyTorch 2.1.0 (vs. 2.5.0 from the paper) required several compatibility fixes for torch.backends.cuda, torch.nn.attention, and torch.hub APIs.

## Top Remaining Ideas (for future runs)

### With Training Code Available:
1. **SiLog Loss**: Replace L1/MSE regression loss with Scale-Invariant Logarithmic loss (DepthAnything, ZoeDepth standard)
2. **LoRA Adapters**: Add low-rank adapters to frozen DINOv2 encoder for domain-specific feature adaptation
3. **Temperature-Margin Scheduling**: Dynamic temperature/margin in the SOC contrastive loss during training (MM-TS 2026)
4. **Upgrade DepthAnything Backbone**: Replace DA-S with DA-B or DA-L for better relative depth quality
5. **Curriculum Learning**: Progressive confidence thresholds for pseudo-label selection

### Without Training Code (Inference-Only):
1. **Multi-Scale + Flip Ensemble**: Add scale variations (0.9x, 1.1x) alongside the current flip TTA
2. **Text Description Augmentation**: Ensemble predictions across multiple text description variants per image
3. **Model Soup / SWA**: If multiple checkpoints from the same run exist, average their weights
