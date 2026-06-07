# Optimization Results: CoFiDA-M

## Summary

- Total iterations: 5 (0-4) evaluated across multiple container runs
- Best `auroc`: **0.8733** (baseline: 0.8689, improvement: **+0.51%**)
- Target was 0.9123 (+5.0%), achieved +0.51%
- Best configuration: `img_size=384` (inference-only change, no retraining required)
- Best commit: `443e7770bde288569f4e0c43fe405ba54207f8f4`

## Baseline vs. Best Metrics

| Metric | Baseline (288) | Best (384) | Delta |
|--------|---------------|------------|-------|
| AUROC | 0.8689 | 0.8733 | +0.0044 (+0.51%) |
| Balanced Acc @ opt | 0.8182 | 0.8240 | +0.0058 |
| Accuracy @ opt | 0.7529 | 0.7443 | -0.0086 |
| Optimal threshold | 0.300 | 0.303 | +0.003 |

## Key Changes Applied

| Iter | Change | AUROC | Effect | Notes |
|------|--------|-------|--------|-------|
| 0 | Baseline (img_size=288) | 0.8689 | - | Paper baseline confirmed |
| 1 | img_size=384 | 0.8733 | +0.51% | **Best result** |
| 2 | img_size=456 | 0.8694 | -0.39% vs 384 | Too large, degraded |
| 3 | Multi-scale 288+384 | 0.8712 | -0.21% vs 384 | Dilution effect |
| 4 | TTA (11 augmentations) | 0.8726 | -0.07% vs 384 | Augmentations hurt |

## What Worked

1. **Higher resolution inference (288→384)**: The single most effective change. EfficientNet-B2 benefits from seeing more detail. Gained +0.51% AUROC without any retraining.
2. **Resolution sensitivity analysis**: 384 is the sweet spot — both lower (288) and higher (456) give worse results, forming an inverted-U relationship.

## What Didn't Work

1. **Test-Time Augmentation (TTA)**: Multiple augmentations (flips, rotations, brightness/contrast) consistently degraded AUROC. The model isn't robust to these transformations, suggesting the training augmentations were limited.
2. **Multi-Scale Ensemble**: Averaging predictions from different resolutions hurt performance, because the lower-resolution predictions dilute the higher-quality 384-scale predictions.
3. **MC Dropout**: Implementation attempts failed due to Docker environment constraints on model import.
4. **Retraining**: Not feasible — the MONET CSV metadata required for teacher-guided student training is not available in the container.

## Why Improvements Were Limited

1. **Eval-only optimization ceiling**: The student model is already well-trained. Without retraining, only input preprocessing changes (resolution, augmentations) can be explored, which have limited impact (~0.5% max).
2. **No MONET CSV**: The most promising optimization avenue — retraining the student with better hyperparameters (class-weighted KD, deeper edit MLP, cosine feature alignment, label smoothing) — requires the MONET concept annotations that are not available in the container.
3. **Docker environment constraints**: Docker exec stdout capture and file persistence issues significantly slowed the iteration cycle.

## Top Remaining Ideas (for future runs with training data)

1. **Class-Weighted Knowledge Distillation**: Weight melanoma-class KD loss 3-5× higher to force better melanoma-specific feature alignment.
2. **Deeper Edit MLP**: Replace the simple 2-layer edit MLP (1408→512→1408) with a deeper residual network (1408→512→256→512→1408).
3. **Cosine Feature Alignment Loss**: Replace MSE with cosine similarity for directional feature alignment.
4. **Label Smoothing in KD**: Apply 0.05-0.1 label smoothing to teacher logits for softer, more informative distillation targets.
5. **Attention-Gated Feature Fusion**: Replace simple residual fusion with learned attention gating.
6. **EMA Decay Schedule**: Cosine-anneal EMA decay from 0.9999 to 0.995 during teacher training.
