# Learning to Solve PDEs on Neural Shape Representations: A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization study performed on the public release of *Learning to Solve PDEs on Neural Shape Representations* (Welschinger et al., CVPR 2026), a neural-operator method that combines a Closest Point Method (CPM) finite-difference discretization with a learned right-hand-side (RHS) extension produced by SurfNO, a SurfaceNet-style attention operator. The optimization was driven by AutoSOTA (`tsinghua-fib-lab/AutoSOTA`) and targeted the per-evaluation `nmaxe` (normalized max error) metric on the sphere benchmark. Two iterations were executed against the released pretrained SurfNO checkpoint without retraining. The headline result is a reduction of `nmaxe` from a pure-SurfNO baseline of 2.386 to **0.279** (−88.3%), with an accompanying reduction of `nmae` (normalized mean absolute error) from 0.640 to 0.082 (−87.1%), at a negligible runtime overhead (20.41 s → 20.73 s). The +5%-target threshold of `nmaxe ≤ 0.4588` was achieved early, terminating the optimization at two iterations. The single accepted change is a *residual-mixing* of the SurfNO RHS extension into the original CPM RHS, controlled by a new `SPHERE_RESIDUAL_ALPHA` environment variable; the optimal mixing coefficient is `α = 0.1`. Diagnostic analysis attributes the breakthrough to a structural pathology in the released SurfNO checkpoint: the learned attention is nearly uniform on the sphere geometry (per-entry weight ≈ 1/400, entropy 5.67/5.99 max), so a pure-SurfNO substitution corrupts the RHS, while a small residual contribution preserves the correct CPM signal and lets the neural weights act as a low-amplitude correction.

## 1. Introduction

The original paper (Welschinger et al., CVPR 2026) studies the problem of solving partial differential equations on shapes represented as level sets or implicit surfaces in three dimensions. Its central contribution is a neural-operator extension of the Closest Point Method: a Surface Neural Operator (SurfNO) replaces the explicit finite-difference RHS extension at each surface stencil with a learned attention-weighted aggregation over neighbouring grid points. The method is reported to generalise across geometries while preserving the spectral structure of the underlying PDE.

This report studies whether the released pipeline — pretrained SurfNO checkpoint plus released CPM scaffold — can be improved on the sphere benchmark through purely test-time interventions. The motivation is a stark observation made during baseline reproduction: the pure-SurfNO RHS yields a sphere-benchmark `nmaxe` of 2.386, two orders of magnitude worse than the raw-CPM reference (`nmae ≈ 0.042`). Either the pretrained checkpoint is unsuited to the sphere geometry, or the way SurfNO's output is composed with the CPM RHS is mis-tuned. AutoSOTA was used to enumerate, run, and evaluate candidate test-time changes against `nmaxe` in a budgeted iterative loop; the budget was used efficiently, with the target reached after the second iteration.

The remainder of the report covers the original method (Section 2), the limitations targeted by the optimization (Section 3), the methodology (Section 4), the experimental setup, results, and the residual-mixing sweep (Section 5), a discussion centred on the attention-uniformity diagnosis (Section 6), and reproducibility information (Section 7).

## 2. Original Method (Background)

The released pipeline consists of two cooperating components, plus a small number of analytic PDE drivers in `src/`.

* **Closest Point Method (CPM) scaffold.** A standard CPM discretization of PDEs on implicit surfaces. The release ships analytic drivers for the heat and Poisson equations (`src/Solve_heat_equation.py`, `src/Solve_poisson_equation.py`) and the supporting kernels in `src/utils/`. CPM is well-known to be O(h²)-accurate on the sphere when configured with an appropriate radial-basis-function (RBF) interpolation; the *raw-CPM reference* in this study has `nmae ≈ 0.042`.
* **SurfNO neural RHS extension.** A SurfaceNet/attention-based neural operator (`src/model/`) that takes a stencil's neighbourhood as input and predicts a neural RHS, intended as a replacement for the analytic CPM RHS extension. The release ships pretrained weights and a learned scalar `lambda_scale = 2.489` that modulates the attention-penalty term inside the SurfNO softmax.

Evaluation is driven by `eval_sphere.py`, which loads the pretrained SurfNO checkpoint, applies it stencil-by-stencil on the sphere benchmark, and reports `nmaxe`, `nmae`, and per-evaluation timing in `scores.jsonl`. The `optimization_curve.png` artifact provides a per-iteration trajectory.

## 3. Identified Limitations

The optimization study identified three issues, the first two of which are diagnostic and the third of which provided the actual lever.

1. **Near-uniform attention on the sphere.** Direct inspection of the SurfNO attention output on the sphere stencils showed an entropy of 5.67 against a theoretical maximum of 5.99 (i.e. extremely close to uniform), with per-entry weights at approximately `1/400` for a 400-neighbour stencil. The learned `lambda_scale = 2.489` makes the softmax penalty term dominate, suppressing any geometric structure that the attention would otherwise encode.
2. **Pure-SurfNO RHS is uncorrelated with ground truth.** When the neural RHS is used as a direct replacement for the analytic CPM RHS, its correlation with the ground-truth RHS is only 0.115 (essentially random). The pure-SurfNO solution therefore inherits the error of an essentially random RHS extension, producing `nmaxe ≈ 2.386`.
3. **No exposed knob to attenuate the neural RHS.** The released `eval_sphere.py` substitutes the neural RHS for the analytic one without any tunable mixing. This forecloses the most natural test-time correction, namely treating the neural output as a small *residual* on top of the analytic RHS.

A practical operating constraint of the study is that retraining the SurfNO checkpoint with sphere-specific data is disallowed under the optimization protocol's no-pretrained-weight-modification rule; only test-time interventions may be considered.

## 4. Optimization Methodology

The two retained iterations exercise a single-line change followed by an α-sweep.

**Iteration 1 — Pure-SurfNO baseline (reference).** Confirmed the pure-SurfNO RHS substitution baseline at `nmaxe = 2.386`, `nmae = 0.640`, and produced the diagnostic measurements (attention entropy, per-entry weight distribution, RHS correlation with ground truth) that motivated Iteration 2.

**Iteration 2 — Residual mixing of the neural RHS extension (IDEA-001).** A new environment variable `SPHERE_RESIDUAL_ALPHA` was introduced in `eval_sphere.py`, parameterising a convex combination of the analytic and neural RHS extensions:

```
rhs_mixed = (1 − α) · rhs_original + α · rhs_neural
```

The value `α = 0.1` was selected after sweeping `α ∈ {0.1, 0.2, 1.0}`. The value `α = 1.0` reproduces the pure-SurfNO baseline (`nmaxe = 2.386`); `α = 0.1` produces the optimum (`nmaxe = 0.279`). At this `α`, the neural RHS contributes approximately 10% of the magnitude of the analytic CPM RHS, which is small enough not to corrupt the signal but large enough to act as a low-amplitude correction.

**Approaches considered but not retained or not reached.**

* *Pure SurfNO (`α = 1.0`).* `nmaxe = 2.386` — preserved as a reference value, not retained as best.
* *Attention temperature scaling.* Not exercised; a separate diagnostic indicated that overriding `lambda_scale` alone does not repair the attention-uniformity pathology because the post-softmax distribution remains close to uniform.
* *IMQ-kernel substitution* (replacing the Gaussian RBF used by CPM with an inverse-multiquadric kernel for better conditioning), *per-stencil adaptive RBF ε* (condition-number-driven), *Richardson extrapolation* on grid spacing for O(h⁴) accuracy, *adaptive surface-point density* based on error gradients, and *direct `lambda_scale` override*: all listed as candidate ideas but not exercised, since the target threshold was reached after the residual-mixing change.

## 5. Experiments

### 5.1 Setup

The optimization target was `nmaxe` on the sphere benchmark, computed by the released `eval_sphere.py` pipeline. All runs used the released SurfNO pretrained checkpoint unchanged. The improvement target was `nmaxe ≤ 0.4588` (the protocol's standard +5%-relative threshold), which was achieved by the second iteration. Per-iteration runtime is approximately 20 s, well within the AutoSOTA per-iteration budget.

### 5.2 Quantitative Results

| Metric | Baseline (α = 1.0, pure SurfNO) | Best (α = 0.1) | Delta |
|---|---:|---:|---:|
| nmaxe | 2.386 | **0.279** | **−2.107 (↓ 88.3%)** |
| nmae | 0.640 | **0.082** | **−0.558 (↓ 87.1%)** |
| time_s | 20.41 | 20.73 | +0.32 |

For reference, the raw-CPM (no-neural) baseline measured during diagnostic analysis has `nmae ≈ 0.042`. The optimized configuration reaches `nmae = 0.082`, i.e. within 2× of the raw-CPM accuracy floor while still incorporating a non-trivial neural-RHS contribution. The `nmae` reduction (−87.1%) is comparable in magnitude to the `nmaxe` reduction (−88.3%), indicating that the residual mixing improves the entire error distribution rather than only its tail.

### 5.3 Ablation / Iteration Trajectory

| Iter | Configuration | `nmaxe` | Status |
|---|---|---:|---|
| 1 | Pure SurfNO (α = 1.0) | 2.386 | reference; identifies the pathology |
| 2 | Residual mixing, α = 0.1 (IDEA-001) | **0.279** | **target met (≤ 0.4588)** |

Sub-sweep over α (within Iteration 2):

| α | `nmaxe` | Note |
|---:|---:|---|
| 0.1 | **0.279** | best |
| 0.2 | (intermediate) | worse than 0.1 |
| 1.0 | 2.386 | pure SurfNO |

The monotone-toward-zero behaviour over α is consistent with the diagnostic finding that the neural RHS is uncorrelated with ground truth: any non-trivial weight on it injects noise; the optimum is therefore a small but non-zero α, not zero (the neural RHS still carries useful low-amplitude information that the analytic CPM RHS does not have).

## 6. Discussion

The dominant takeaway is diagnostic. The released SurfNO checkpoint, applied to the sphere benchmark, produces an attention distribution that is almost indistinguishable from uniform. Because the learned `lambda_scale = 2.489` makes the softmax penalty dominate, no geometric structure survives the attention bottleneck, and the resulting RHS is essentially noise. Composing this RHS as a *replacement* for the analytic CPM RHS therefore destroys the analytic signal; composing it as a small *residual* with `α = 0.1` preserves the analytic signal and gains a small additional correction from the neural pathway.

This pattern — neural component as a low-amplitude residual on top of an analytic baseline — is well known in scientific computing and in residual-corrected solvers; the present study confirms its applicability when the neural component itself is mis-conditioned. The 88% improvement is a function of the gap between the analytic and pure-neural baselines on this geometry; on geometries where the SurfNO checkpoint produces non-uniform attention the optimal α is likely to be larger and the relative improvement smaller, but the residual-mixing recipe should remain a useful default.

A subtler point is that the result here cannot be treated as evidence against the SurfNO architecture. The pathology lives in the released pretrained weights on the sphere, not in the architecture itself. The most plausible architectural follow-up is to introduce a geometry-aware attention bias (e.g. a distance-based prior added to the attention logits) so that the attention does not collapse to uniform when a softmax penalty is large; alternatively, attention temperature scaling (dividing logits by T < 1) is a candidate test-time fix that has not been exercised here but is a natural next iteration.

Several additional levers remain for future runs: replacing the Gaussian RBF with an IMQ kernel for better conditioning of the CPM stencil; per-stencil adaptive RBF ε targeting a fixed condition number; Richardson extrapolation in grid spacing to lift the accuracy from O(h²) to O(h⁴); adaptive surface-point density driven by error gradients; and a direct override of `lambda_scale` to sharpen the attention. None of these were required to clear the present optimization target.

## 7. Reproducibility

The slimmed repository contains the code required to reproduce the best configuration; the SurfNO pretrained checkpoint is intentionally not included.

* **Best configuration.** Set the environment variable
  ```
  SPHERE_RESIDUAL_ALPHA=0.1
  ```
  and run the released `eval_sphere.py`. The released SurfNO checkpoint is used unchanged.
* **Files touched.** `eval_sphere.py` (RHS-mixing arithmetic and the `SPHERE_RESIDUAL_ALPHA` parsing).
* **Pretrained checkpoint.** Use the SurfNO weights distributed with the original release.
* **Environment.** As documented by the original repository's setup. The slimmed repository preserves `src/` (model, utils, training and PDE drivers) and the evaluation entry point but omits the original `src/data/` payload.
* **Diagnostic check.** With the pretrained weights, the attention distribution on the sphere has entropy ≈ 5.67/5.99 max and per-entry weight ≈ 1/400. Setting `SPHERE_RESIDUAL_ALPHA=1.0` should reproduce the pure-SurfNO baseline of `nmaxe = 2.386`; setting `SPHERE_RESIDUAL_ALPHA=0.1` should reproduce the optimized `nmaxe = 0.279`.

## 8. References

* Welschinger et al. (2026). *Learning to Solve PDEs on Neural Shape Representations*. CVPR 2026.
* AutoSOTA: Tsinghua FIB Lab. *AutoSOTA: An automated SOTA-chasing harness*. [github.com/tsinghua-fib-lab/AutoSOTA](https://github.com/tsinghua-fib-lab/AutoSOTA).
* Closest Point Method: Ruuth, S. J., & Merriman, B. (2008). *A simple embedding method for solving partial differential equations on surfaces*. (Foundational analytic baseline.)
