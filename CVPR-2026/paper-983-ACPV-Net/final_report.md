# Optimization Results: ACPV-Net — All-Class Polygonal Vectorization

## Summary
- **Total iterations**: 7 (stopped early — target exceeded)
- **Best `ciou_building`**: **75.78%** (baseline: 71.92%, **+5.37% improvement**)
- **Target**: 75.516% (5% above baseline) — **EXCEEDED ✓**
- **Best commit**: `c1d8850a81` (iter 7)

## Baseline vs. Best Metrics

| Metric | Baseline | Best (Final) | Delta | Direction |
|--------|----------|-------------|-------|-----------|
| **ciou_building** | 71.92 | **75.78** | **+3.86** | ↑ +5.37% |
| iou_building | 79.03 | **82.06** | +3.03 | ↑ +3.84% |
| biou_building | 76.27 | **79.51** | +3.24 | ↑ +4.25% |
| nratio_building | 0.87 | **1.11** | +0.24 | ↑ +27.6% |
| polis_building | 2.00 | **1.83** | -0.17 | ↓ 8.5% |
| ciou_road | 63.06 | **65.39** | +2.33 | ↑ +3.70% |
| ciou_unvegetated | 57.83 | **59.46** | +1.63 | ↑ +2.82% |
| ciou_vegetation | 64.45 | **68.53** | +4.08 | ↑ +6.33% |
| ciou_water | 57.01 | **57.00** | -0.01 | ≈ stable |

## Key Changes Applied

Only one source file was modified: `tools/extract_vertices_from_heatmap.py`

| Change | Effect | Iteration |
|--------|--------|-----------|
| **Heatmap channel fusion: mean → max** | Preserved stronger per-channel peaks, improved n-ratio 0.87→0.89, building ciou +0.13 | Iter 2 |
| **Vertex threshold 0.1→0.05** | No standalone effect (topk bottleneck), but established for later iterations | Iter 3 |
| **Topk 300→500** | **Critical breakthrough**: n-ratio 0.89→1.00, ciou 72.05→75.36 (+3.31!) | Iter 4 |
| **Adaptive threshold (Leap)** | Produced same result as fixed 0.05 — validated optimal threshold | Iter 6 |
| **Topk 500→750** | **Final push**: n-ratio 1.00→1.11, ciou 75.36→75.85, target exceeded! | Iter 7 |
| **Pipeline flags: VSS+DP-fix** | Enabled in polygonization (structure change but no aggregate gain alone) | Iter 1 |

## What Worked

1. **Vertex count optimization is the #1 lever for CIoU improvement**: CIoU = IoU × ps, where ps penalizes vertex count mismatch. The baseline n-ratio of 0.87 was the primary bottleneck. Increasing topk from 300→750 delivered the majority of gains (+3.93 ciou, +3.03 iou).

2. **Max pooling > mean pooling for multi-channel heatmap fusion**: The 3-channel KL-f4 latent decoder produces complementary vertex activations across channels. Max pooling preserves strong peaks in any channel, while mean pooling averages them away.

3. **VSS+DP-fix changes structure but not aggregate**: The paper's vertex-guided subset selection redistributes vertices within images but doesn't change mean metrics when vertex counts are already optimized.

4. **Simple parameter changes can yield major gains**: The entire +5.37% improvement came from inference-time parameter adjustments — no retraining, no architectural changes, no dataset modifications.

## What Didn't Work

1. **Lowering vertex threshold alone** (iter 3): With topk=300, the 300th candidate was already above threshold, so lowering it had no effect.

2. **Tighter PSLG snap distance** (iter 5): dist_thresh=3.0 excluded some valid snapped vertices, reducing n-ratio and ciou slightly.

3. **Adaptive threshold** (iter 6): The adaptive computation (95th percentile × 0.3) converged to roughly the same 0.05 value the fixed threshold was already using.

## Optimization Trajectory

```
Iter 0: 71.92 (baseline)
Iter 1: 71.92 (VSS+DP-fix, no gain)
Iter 2: 72.05 (max pooling, +0.13)
Iter 3: 72.05 (threshold 0.05, no gain — topk bottleneck)
Iter 4: 75.36 (topk=500, +3.31 — BREAKTHROUGH)
Iter 5: 75.30 (dist_thresh=3.0, regression)
Iter 6: 75.36 (adaptive threshold LEAP, same as iter 4)
Iter 7: 75.85 (topk=750, +0.49 — TARGET EXCEEDED)
Final: 75.78 (full pipeline confirmation)
```

## Top Remaining Ideas (for future runs)

1. **Increase DDIM steps beyond 200**: More denoising steps could improve vertex heatmap quality further, potentially pushing building ciou above 77.

2. **Per-class topk optimization**: Different classes need different vertex budgets — buildings benefit from high topk, while vegetation might need less.

3. **Multi-scale vertex consensus**: Extracting vertices at multiple NMS kernel sizes and keeping only consensus vertices could reduce noise while preserving true detections.

4. **Confidence-weighted vertex selection**: Use heatmap activation strength + segmentation probability to weight vertices, filtering noise in uncertain regions.

5. **Curvature-guided polygon simplification**: Weight DP simplification by local boundary curvature to preserve sharp corners more effectively.

6. **Alternative diffusion sampler (DPM++)**: Could match or exceed DDIM quality at fewer steps, enabling higher effective step counts within same inference budget.
