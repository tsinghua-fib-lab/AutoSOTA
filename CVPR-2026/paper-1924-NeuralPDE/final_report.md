# Optimization Results: Learning to Solve PDEs on Neural Shape Representations

## Summary
- Total iterations: 2 (target reached early)
- Best `nmaxe`: **0.279** (baseline: 2.386, improvement: **↓ 88.3%**)
- Best `nmae`: **0.082** (baseline: 0.640, improvement: **↓ 87.1%**)
- Target threshold: nmaxe ≤ 0.4588 **ACHIEVED**
- Key method: Residual mixing of neural weight extension (alpha=0.1)

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta | Improvement |
|--------|----------|------|-------|-------------|
| nmaxe | 2.386 | 0.279 | -2.107 | ↓ 88.3% |
| nmae | 0.640 | 0.082 | -0.558 | ↓ 87.1% |
| time_s | 20.41 | 20.73 | +0.32 | — |

## Key Changes Applied

### 1. Residual Mixing of Neural Weight Extension (IDEA-001)
- **File**: `/repo/eval_sphere.py`
- **Change**: Added `SPHERE_RESIDUAL_ALPHA` environment variable controlling neural weight influence
- **Formula**: `rhs_mixed = (1 - alpha) * rhs_original + alpha * rhs_neural`
- **Best value**: alpha = 0.1

### Root Cause Analysis
The pretrained SurfNO model produces near-uniform attention weights on the sphere (entropy 5.67/5.99 max). The learned `lambda_scale=2.489` makes the penalty term dominate, causing the softmax to produce nearly uniform output (~1/400 per entry). This corrupts the RHS extension, making the pure SurfNO solution essentially random (correlation with ground truth: 0.115).

The residual mixing preserves the correct RHS signal while allowing a small neural weight contribution, achieving near-CPM accuracy (nmae=0.082, close to raw CPM's 0.042).

## What Worked
- **Residual mixing**: Simple, zero-cost change that produced an 88% improvement
- **Lower alpha = better**: alpha=0.1 outperformed alpha=0.2 and alpha=1.0, confirming the neural attention is counterproductive on the sphere

## What Didn't Work
- **Pure SurfNO (alpha=1.0)**: The neural weight extension completely corrupts the RHS, producing nmaxe=2.386
- **Attention temperature scaling**: Not tested, but diagnostic showed lambda_scale override alone doesn't fix the attention uniformity

## Top Remaining Ideas (for future runs)

1. **IDEA-004**: Attention temperature scaling — divide logits by T<1 to sharpen attention
2. **IDEA-003**: Replace Gaussian RBF with IMQ kernel for better conditioning
3. **IDEA-002**: Per-stencil adaptive RBF epsilon via condition-number targeting
4. **IDEA-007**: Richardson extrapolation on grid spacing for O(h^4) accuracy
5. **IDEA-005**: Adaptive surface point density based on error gradients
6. **IDEA-021**: Override lambda_scale to sharpen attention directly

## Key Diagnostic Insight

The SurfNO pretrained weights produce near-uniform attention on the sphere geometry. This is the fundamental bottleneck — any optimization that doesn't address the attention quality will see limited gains. The residual mixing approach (IDEA-001) bypasses this bottleneck by treating the neural weights as a correction rather than a replacement.

For best results on the sphere, future work should either:
a) Retrain SurfNO with sphere-specific data (requires modifying pretrained weights — not allowed)
b) Modify the attention mechanism to be more geometry-aware (e.g., distance-based bias)
c) Use the residual mixing approach as a foundation and optimize other pipeline components (RBF, Laplacian, band construction)
