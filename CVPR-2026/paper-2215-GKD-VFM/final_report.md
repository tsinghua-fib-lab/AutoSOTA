# Optimization Results: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation

## Summary
- **Total iterations**: 2 (target achieved early)
- **Baseline `mIoU_cityscapes`**: 52.02%
- **Best `mIoU_cityscapes`**: **54.72%** (+2.70 / +5.19%)
- **Target**: 54.621% — **ACHIEVED** ✓ (+0.10 above target)
- **Best commit**: `3e25066`

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| mIoU | 52.02 | **54.72** | **+2.70** |
| aAcc | 90.74 | 91.95 | +1.21 |
| mAcc | 68.42 | 69.52 | +1.10 |

## Per-Class IoU Comparison

| Class | Baseline | Best | Delta |
|-------|----------|------|-------|
| road | 92.55 | 93.49 | +0.94 |
| sidewalk | 55.09 | 59.45 | +4.36 |
| building | 88.02 | 89.44 | +1.42 |
| wall | 50.24 | 49.00 | -1.24 |
| fence | 40.26 | 44.00 | +3.74 |
| pole | 38.75 | 49.22 | **+10.47** |
| traffic light | 44.79 | 54.40 | **+9.61** |
| traffic sign | 28.28 | 36.14 | +7.86 |
| vegetation | 87.46 | 89.63 | +2.17 |
| terrain | 44.68 | 46.30 | +1.62 |
| sky | 88.85 | 90.57 | +1.72 |
| person | 65.18 | 70.61 | +5.43 |
| rider | 32.19 | 27.26 | -4.93 |
| car | 87.71 | 89.97 | +2.26 |
| truck | 40.46 | 44.75 | +4.29 |
| bus | 76.56 | 81.08 | +4.52 |
| train | 0.40 | 0.24 | -0.16 |
| motorcycle | 26.84 | 24.12 | -2.72 |
| bicycle | 0.00 | 0.00 | 0.00 |

**Key observation**: Multi-scale inference dramatically improved small/thin classes (pole +10.5, traffic light +9.6, traffic sign +7.9, person +5.4) by providing higher-resolution detail. Minor regressions occurred on a few classes (wall, rider, motorcycle, train) where the averaged predictions across scales introduced ambiguity. The net effect is strongly positive (+2.70 mIoU).

## Key Changes Applied

| Change | File | Effect | Notes |
|--------|------|--------|-------|
| Multi-scale model wrapper | `rein/models/segmentors/msi_wrapper.py` (new, 48 lines) | +2.70 mIoU | Custom `MultiScaleModel` class that runs inference at 3 scales [0.75x, 1.0x, 1.25x] and averages logits |
| Module registration | `rein/models/segmentors/__init__.py` | Enables config-based loading | Registered `MultiScaleModel` in mmseg registry |
| Config change | `eval_cityscapes_config.py` | Switches to multi-scale | Changed model type to `MultiScaleModel` with `ms_scales=[0.75, 1.0, 1.25]` |

**Total diff**: 3 files changed, 52 insertions, 2 deletions.

## What Worked

1. **Multi-scale inference with tight scale range [0.75, 1.0, 1.25]** — This was the single most impactful change, providing +2.70 mIoU. The tight range around 1.0x preserved detail while capturing multi-scale context. Wider ranges like [0.5, 1.0, 1.5] gave +1.54 mIoU, suggesting that extreme scales introduce artifacts.
2. **Scale selection matters** — [0.75, 1.0, 1.25] outperformed [0.5, 1.0, 1.5] by +1.16 mIoU. Too-small scales (0.5x) lose critical detail; too-large scales (1.5x) amplify noise.
3. **Model wrapper approach** — Creating a `MultiScaleModel` as a subclass of the existing `FrozenBackboneEncoderDecoder` was clean, maintainable, and required minimal config changes. The approach preserves all existing mmseg evaluation infrastructure.

## What Didn't Work

1. **Flip TTA alone** — Only +0.29 mIoU (52.02→52.31). While positive, the gain was too small to justify the doubled inference time.
2. **Flip TTA combined with multi-scale** — Timed out at 900s. Combined approach (6 forward passes per image) was too slow for the evaluation budget.
3. **MultiScaleFlipAug pipeline approach** — Incompatible with newer mmseg data format (`PackSegInputs`). Required custom model wrapper instead.
4. **CRF post-processing** — `pydensecrf` could not be installed due to proxy restrictions. This technique remains untested.

## Optimization Trajectory

```
Baseline: 52.02
  → Iter 1 (MSI [0.5,1.0,1.5]): 53.56 (+1.54)
  → Iter 2 (MSI [0.75,1.0,1.25]): 54.72 (+2.70) 🎯 TARGET MET
```

## Top Remaining Ideas (for future runs)

1. **DenseCRF post-processing** (IDEA-003) — Estimated +0.5–1.5 mIoU on top of current best. Requires installing pydensecrf.
2. **Class-aware CRF** (IDEA-004) — Different CRF parameters per class to avoid over-smoothing thin structures. Estimated +0.5–1.0 mIoU over uniform CRF.
3. **Confidence-guided selective CRF** (IDEA-008) — Apply CRF only to low-confidence pixels. Estimated +0.3–0.8 mIoU.
4. **Larger test crop size** (IDEA-007) — Using 768×768 or 1024×512 test crops. Estimated +0.5–2.0 mIoU.
5. **Per-class logit bias** (IDEA-009) — Boost rare class predictions to address train/bicycle/motorcycle failures.
6. **Backbone upgrade to DeiT-B** — The paper's own results show DINOv2-L→DeiT-B reaching 54.2 mIoU without TTA. Would require re-running the full distillation pipeline (not feasible without training data).
