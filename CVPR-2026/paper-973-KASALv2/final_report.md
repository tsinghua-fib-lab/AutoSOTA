# Optimization Results: KASALv2 — Fully Automatic 3D Rotational Symmetry Classification and Axis Localization

## Summary
- **Total iterations**: 13 (plus baseline)
- **Best `eadi_d`**: **0.002988** (baseline: 0.003073, improvement: **-2.8%**)
- **Best runtime**: 19.5s (baseline: 9.7s — runtime was not the optimization target)
- **Best commit**: `c522b0f` (iter-7)
- **Target**: ≤ 0.0029 — **NOT reached** (0.002988 vs 0.0029 target, gap: 3.0%)

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta | % Change |
|--------|----------|------|-------|----------|
| eADI/d (mean) | 0.003073 | 0.002988 | -0.000085 | -2.8% |
| C(=1) Circular | 0.003221 | 0.003219 | -0.000002 | -0.1% |
| C(>1) Cylindrical | 0.003162 | 0.003167 | +0.000005 | +0.2% |
| D(=1) Pyramidal | 0.002531 | 0.002518 | -0.000013 | -0.5% |
| D(>1) Prismatic | 0.003175 | 0.003012 | -0.000163 | -5.1% |
| P(20) Icosahedral | 0.004134 | 0.004130 | -0.000004 | -0.1% |
| P(4) Tetrahedral | 0.003554 | 0.003531 | -0.000023 | -0.6% |
| P(8) Octahedral | 0.001954 | 0.001954 | 0.000000 | 0.0% |

### Per-Object Improvements (best vs baseline)

Most improved objects:
| Object | Baseline | Best | Delta |
|--------|----------|------|-------|
| obj_000006 (D(>1) Prismatic) | 0.004364 | 0.002950 | **-32.4%** |
| obj_000018 (D(>1) Prismatic) | 0.003727 | 0.003591 | -3.6% |
| obj_000014 (D(>1) Prismatic) | 0.001857 | 0.001845 | -0.6% |
| obj_000013 (D(>1) Prismatic) | 0.001714 | 0.001682 | -1.9% |

## Key Changes Applied

| Change | File | Effect | Notes |
|--------|------|--------|-------|
| **Point-to-plane ICP** | `kasal/geometry/o3d_icp.py` | -2.7% eADI/d | The ONLY meaningful improvement. Normals provide additional geometric constraints. |
| Per-type parameters | `eval_final.py` | 0.0% | Negligible effect but included in best state |
| Exact dis_ang | `kasal/keyaxis/keyaxis.py` | -0.03% | Marginal. Original quantization was reasonable. |
| fpsample_num 1500→3000 | `eval_final.py` | -0.03% | Doubled FPS points; marginal benefit at 2× runtime cost |

## What Worked

1. **Point-to-plane ICP** was the single most impactful change. It reduced eADI/d by 2.7% overall and dramatically improved specific objects (obj_000006 dropped 32% from 0.004364 to 0.002950). The improvement is concentrated in D(>1) prismatic objects (-5.1%).

2. The KASAL algorithm's axis search (Fibonacci sphere) is saturated at ~5000 directions — increasing `sample_num`, adding local refinement, or using two-stage search provides no additional benefit.

3. ICP refinement is essential — disabling it causes a 70% regression (0.005103 vs 0.002989).

## What Didn't Work

1. **Multi-candidate axis averaging** (IDEA-001): Caused catastrophic regression for platonic solids because top-K candidates belong to different axis families.
2. **Conditional ICP quality gating** (IDEA-007): The validation function used the same flawed dis_ang quantization that doesn't correlate with true eADI/d.
3. **Normal-aware scoring** (IDEA-006): Adding normal consistency to the KDTree scoring didn't change axis rankings.
4. **Two-stage coarse-to-fine search** (IDEA-004): Coarse search with local refinement produced the same results as exhaustive.
5. **All parameter tuning** (sample_num, fpsample_num, half_sphere, ICP convergence, voxel size): No meaningful effect.

## Why Target Was Not Reached

The KASALv2 paper reports eADI/d = 0.00212 at 1.46s using:
1. **GDG (Geometric Degeneration Guidance)** — automatic symmetry type classification
2. **Efficient 2-stage sampling** — 128 coarse + local refinement

The public repository only contains **KASAL v1** code (TIP 2025), which uses exhaustive Fibonacci sphere search with ground-truth symmetry types. The v1 algorithm has a fundamental accuracy ceiling of ~0.003. Without the KASALv2 classification and sampling code (not publicly released), matching the paper-reported 0.00212 is not feasible.

## Top Remaining Ideas (for future runs)

1. **Implement KASALv2 GDG classification**: Automatic symmetry type detection would enable per-type optimized pipelines and potentially bridge the remaining gap to 0.00212.
2. **Per-object learned axis refinement**: A small MLP trained on the DSRSTO dataset could learn to correct systematic errors in the geometric axis search.
3. **Robust ICP with Huber/Tukey loss**: Replace the Open3D point-to-plane ICP with a custom implementation using robust kernels for outlier rejection.
4. **Multi-view consistency validation**: Render the object from multiple views, apply symmetry transforms, and validate consistency in image space.
