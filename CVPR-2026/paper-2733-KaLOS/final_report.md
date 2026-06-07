# Optimization Results: KaLOS finds Consensus: A Meta-Algorithm for Evaluating Inter-Annotator Agreement in Complex Vision Tasks

## Summary
- Total iterations: 16
- Best `mean_alpha`: **0.942825** (baseline: 0.807823, improvement: **+16.71%**)
- Best `global_alpha`: **0.965824** (baseline: 0.825843, improvement: **+16.95%**)
- Target: 0.8482 — **exceeded by 11.2%**

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| mean_alpha | 0.807823 | 0.942825 | +0.135002 (+16.71%) |
| global_alpha | 0.825843 | 0.965824 | +0.139981 (+16.95%) |
| method | greedy | greedy | - |
| similarity_threshold | 0.5 | 0.5 | - |
| threshold_func | bbox_iou_similarity | bbox_iou_centroid_fusion | changed |
| cost_func | negative_score | negative_score | - |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| Added `bbox_iou_centroid_fusion()` similarity function | +16.7% mean_alpha | Fuses IoU (20%) + centroid similarity (80%) — the breakthrough change |
| Registered new function in THRESHOLD_FUNCTIONS | Required for pipeline integration | correspondence_algorithms.py |
| Added function to eval.py argparse choices | Required for CLI access | eval.py |

### The `bbox_iou_centroid_fusion` Function

```python
def bbox_iou_centroid_fusion(ann1, ann2, iou_weight=0.2):
    """Fuses IoU and centroid similarity into a single score."""
    iou = bbox_iou_similarity(ann1, ann2)
    cent = centroid_similarity(ann1, ann2)
    return iou_weight * iou + (1.0 - iou_weight) * cent
```

**Why it works**: The pure IoU similarity function produces 0.0 for non-overlapping bounding boxes, losing all spatial information. By fusing IoU with centroid similarity (which normalizes centroid distance by bbox diagonal), the fused function provides:
1. Higher scores for true positive matches (annotations of the same object)
2. Better ordering in greedy matching (true matches get lower cost and are processed first)
3. Rescue of borderline cases where IoU drops below 0.5 but centroid proximity still indicates a match

**Weight tuning results**:
| IoU Weight | Centroid Weight | mean_alpha |
|------------|-----------------|------------|
| 0.6 | 0.4 | 0.934485 |
| 0.5 | 0.5 | 0.939295 |
| 0.4 | 0.6 | 0.941167 |
| 0.3 | 0.7 | 0.942825 |
| 0.2 | 0.8 | 0.942825 |
| 0.1 | 0.9 | 0.941678 |
| 0.0 (pure centroid) | 1.0 | 0.941678 |
| 1.0 (pure IoU/baseline) | 0.0 | 0.807823 |

Optimal weight: **0.2 IoU / 0.8 centroid**

## What Worked
- **Multi-metric fusion (IoU + centroid)**: The single most impactful change. Combining geometric overlap with positional proximity dramatically improves correspondence matching quality.
- **Weight tuning**: Fine-tuning the fusion weights found the optimal balance at 20% IoU, 80% centroid.

## What Didn't Work
- **Spatial proximity gate**: Adding a distance-based filter to the pairwise score precomputation had no effect on synthetic data (annotations are already well-separated).
- **Alternative matching methods (SHM, MGM, AHC)**: All produced lower alpha than greedy matching on this synthetic data. Greedy with well-ordered pairs (from the fused similarity) is optimal.
- **Category-lenient cost function**: No effect since all annotators assign the same categories to objects in synthetic data.
- **Centroid tiebreaking**: No effect since fused similarity scores are already unique continuous values.
- **Similarity threshold tuning**: The default 0.5 threshold works well with the fused function; lower (0.3) gives same result, higher (0.7) reduces alpha.

## Top Remaining Ideas (for future runs)
- **IDEA-006**: Hungarian refinement pass on greedy output clusters for local optimality
- **IDEA-011**: Two-stage matching (centroid coarse + IoU fine) for hybrid precision
- **IDEA-015**: Adaptive cost penalty discouraging borderline matches (could help with more complex data)
- **IDEA-008**: Implement bbox_giou_similarity for bounding boxes (handles non-overlapping annotations better)
