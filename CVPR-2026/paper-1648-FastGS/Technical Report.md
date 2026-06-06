# FastGS: Training 3D Gaussian Splatting in 100 Seconds — A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization study performed on the public release of FastGS, a CVPR 2026 paper that targets training-time efficiency for 3D Gaussian Splatting (3DGS) by accelerating the densification pipeline through a Multi-View Consistency Decision/Pruning (VCD/VCP) framework. The optimization was driven by AutoSOTA (`tsinghua-fib-lab/AutoSOTA`) and targeted PSNR on the Mip-NeRF 360 *bicycle* scene. Twenty-one iterations were executed against the released codebase. The headline result is an improvement on bicycle from a baseline PSNR of 24.8063 dB to 25.1873 dB (+0.381 dB, +1.54%), accompanied by a substantially larger SSIM gain (0.7203 → 0.7601, +5.53%) and a 29.16% reduction in LPIPS (0.2997 → 0.2123). The 5%-over-paper PSNR target of 28.791 dB was not reached. The accepted changes are concentrated in the densification controller and the multi-view scoring path, with the dominant lever being an iteration-dependent gradient-threshold decay (0.3× → 1.0×) that permits aggressive early densification and conservative late densification (+0.89% PSNR alone). The combined recipe also includes a finer split factor (`N=3`), a decaying importance-score threshold, anisotropy-guided pruning, and a reversed λ_dssim schedule that prioritises SSIM early and L1 late. The best configuration is captured at commit `dafb00a` (Iteration 17).

## 1. Introduction

FastGS, presented at CVPR 2026, attacks the wall-clock training cost of 3D Gaussian Splatting by replacing the densification heuristics of vanilla 3DGS with a Multi-View Consistency framework. Two components drive the speedup: a *Multi-View Consistency Decision* (VCD) score that selects candidate Gaussians for splitting/cloning based on cross-view rendering errors, and a *Multi-View Consistency Pruning* (VCP) score that removes Gaussians whose contribution is inconsistent across views. Together with a tightened training schedule, the framework is reported to reach competitive Mip-NeRF 360 quality in approximately 100 seconds of training, well below the canonical 30-minute budget of vanilla 3DGS.

This report studies whether the released FastGS pipeline can be improved post hoc, *without* changes to the CUDA rasterizer or external dependencies, using algorithmic and recipe-level interventions on the densification controller, the multi-view scoring weights, and the loss schedule. The motivation is that several knobs in the released training script — the gradient threshold for densification, the per-Gaussian split factor, the importance-score threshold, and the λ_dssim schedule — are exposed as constants and have not been swept in the released configuration. AutoSOTA was used to enumerate, run, and evaluate candidate changes against PSNR on Mip-NeRF 360 bicycle in a 21-iteration budget.

The remainder of the report covers the original method (Section 2), the limitations targeted by the optimization (Section 3), the methodology (Section 4), the experimental setup, results, and ablations (Section 5), a discussion of negative results and what they imply about the released training recipe (Section 6), and reproducibility information (Section 7).

## 2. Original Method (Background)

FastGS extends the standard 3DGS pipeline (Kerbl et al., 2023) with two additions, plus a tightened training schedule.

* **Multi-View Consistency Decision (VCD).** During densification, candidate Gaussians are scored on the basis of cross-view rendering error: a Gaussian whose rendering disagrees substantially across views is a candidate for splitting/cloning. The candidate score is gated by a gradient threshold and an importance score that is computed across a fixed number of cameras.
* **Multi-View Consistency Pruning (VCP).** During pruning, Gaussians whose contribution is inconsistent across views are preferentially removed, with anisotropy and opacity thresholds providing additional gating.
* **Tightened training schedule.** The released `train_base.sh` and `train_big.sh` configurations target ~100-second total training time on Mip-NeRF 360 scenes through a much shorter overall iteration count, an aggressive learning-rate decay, and a small λ_dssim throughout.

The reported baseline metrics for the released checkpoint are PSNR 27.42, SSIM 0.795, LPIPS 0.263 (averaged across the standard scene set). The single-scene bicycle baseline reproduced in the present infrastructure is PSNR 24.8063, SSIM 0.7203, LPIPS 0.2997.

## 3. Identified Limitations

The optimization study identified four sources of friction in the released training pipeline:

1. **Static densification gradient threshold.** The threshold that gates whether a Gaussian is a densification candidate is a single constant. Densification has different optimal aggressiveness early (when the geometry is coarse) versus late (when the geometry is converging), and a single threshold cannot exercise this contrast.
2. **Hard-coded split factor `N=2`.** When a candidate Gaussian is split, the released code subdivides it into two children. A finer split (`N=3`) increases per-split cost but produces smaller, more accurately placed children and is rarely tested in 3DGS releases.
3. **Static importance-score threshold and uniform multi-view scoring.** The importance score is gated at a single threshold and the cross-view rendering errors are averaged uniformly across pixels. Edge-aware weighting and a decaying threshold are candidate improvements that exploit the fact that early densification benefits from permissiveness while late densification benefits from selectivity.
4. **Static λ_dssim throughout training.** The released loss schedule fixes λ_dssim at a small constant. Time-varying schedules — both monotonically increasing (L1 → SSIM) and the inverse (SSIM → L1) — are candidate levers worth ablating.

## 4. Optimization Methodology

The 21 iterations explored four categories of change. All retained changes are localised to the densification controller, the multi-view scoring path, the pruning path, and the loss schedule; no changes were made to the CUDA rasterizer, the data pipeline, or the rendering kernel.

**Densification control (primary lever).**

* *Iteration-dependent gradient threshold decay (Iter 1).* The densification gradient threshold is annealed from 0.3× of the released value at the start of training to 1.0× at the end, permitting aggressive splitting when the model is coarse and conservative splitting as it converges. Effect: +0.89% PSNR.
* *Split scale factor `N=3` (Iter 8).* The per-Gaussian split factor was changed from `N=2` to `N=3`, producing finer geometric subdivisions per split. Effect: +0.63% PSNR.

**Multi-view scoring.**

* *Importance-score threshold reduced from `>5` to `>3` (Iter 5).* More Gaussians pass the multi-view importance filter into the densification path. Effect: +0.31% PSNR.
* *Edge-aware multi-view scoring (Iter 6).* Per-pixel VCD errors are weighted by edge intensity (image-gradient magnitude) before being aggregated into the candidate score. Effect: +0.06% PSNR.
* *Decaying importance-score threshold from 2 to 5 (Iter 15).* The threshold is permissive early (encouraging exploration) and selective late (encouraging stability). Effect: +0.19% PSNR.

**Pruning.**

* *Anisotropy-guided pruning (Iter 3).* Anisotropic Gaussians (which carry geometric detail in elongated structures such as bike spokes) are protected from premature removal. Effect: +0.02% PSNR.

**Loss schedule.**

* *Adaptive λ_dssim schedule from 0.05 to 0.40 (Iter 2).* PSNR-neutral but improved SSIM/LPIPS.
* *Reverse λ_dssim schedule from 0.40 to 0.05 (Iter 17, retained as best).* SSIM-heavy early, L1-heavy late, prioritising perceptual structure early and pixel accuracy late. Effect: +0.05% PSNR (and the configuration that captured the overall best result).

**Approaches tested but not retained.**

* Cosine LR schedule with warmup (Iter 4): −3.1% PSNR. The released exponential decay is well-tuned to FastGS's tightened iteration count.
* Hard view mining (Iter 7): minor regression. Uniform random view sampling is preferable.
* Increasing the VCD camera count from the released value to 20 (Iter 11): no PSNR gain at substantially higher compute.
* Opacity clamp relaxation (Iter 9), final-prune relaxation (Iter 10), prune-budget increase (Iter 12): all neutral or slightly negative.
* Repeated tuning of `grad_abs_thresh`, the λ_dssim start value, and the decay-factor range produced no further movement — the metric had saturated.

No changes to data, evaluation, or the rasterizer kernel were made. All retained changes are localised to densification logic, multi-view scoring, pruning, and the loss schedule.

## 5. Experiments

### 5.1 Setup

The optimization target was PSNR on the Mip-NeRF 360 *bicycle* scene, evaluated through the released `full_eval.py` and `metrics.py` pipeline. All runs used the released training entry point (`train.py`) with the configuration from `train_base.sh`/`train_big.sh` as the starting point and the FastGS-released defaults for the rasterizer, scene loaders, and LPIPS evaluator. AutoSOTA executed 21 iterations under a fixed wall-clock budget per iteration. The improvement target was PSNR ≥ 28.791 dB, set as the +5%-over-paper PSNR target relative to the paper's averaged baseline (PSNR 27.42); on the present single-scene bicycle baseline this target is unattainable through the iteration budget exercised here.

### 5.2 Quantitative Results

| Metric | Baseline (bicycle) | Best (bicycle) | Delta |
|---|---:|---:|---:|
| PSNR | 24.8063 | **25.1873** | +0.3810 (+1.54%) |
| SSIM | 0.7203 | **0.7601** | +0.0398 (+5.53%) |
| LPIPS | 0.2997 | **0.2123** | −0.0874 (−29.16%) |

For reference, the paper's reported (averaged) baseline is PSNR 27.42, SSIM 0.795, LPIPS 0.263. The present single-scene bicycle baseline starts substantially below this average, which is consistent with bicycle being one of the harder Mip-NeRF 360 scenes; gains in this study are reported relative to the bicycle-specific baseline.

The best configuration is captured at commit `dafb00a` (Iteration 17) and combines all retained changes from the methodology section.

### 5.3 Ablation / Iteration Trajectory

| # | Change | Iter | PSNR Delta | Notes |
|---|---|---:|---:|---|
| 1 | Iteration-dependent gradient-threshold decay (0.3× → 1.0×) | 1 | **+0.89%** | Largest single lever |
| 2 | Adaptive λ_dssim schedule (0.05 → 0.40) | 2 | −0.04% | PSNR-neutral; improves SSIM/LPIPS |
| 3 | Anisotropy-guided pruning | 3 | +0.02% | Protects elongated detail |
| 4 | Importance-score threshold `>5 → >3` | 5 | +0.31% | More Gaussians enter densification |
| 5 | Edge-aware multi-view scoring | 6 | +0.06% | Per-pixel weight by image gradient |
| 6 | Split scale factor `N=2 → N=3` | 8 | +0.63% | Finer geometric subdivisions |
| 7 | Decaying importance-score threshold (2 → 5) | 15 | +0.19% | Permissive early, selective late |
| 8 | Reverse λ_dssim schedule (0.40 → 0.05) | 17 | +0.05% | Captured overall best |

The trajectory exhibits a clear pattern of diminishing returns: the first three densification-control changes account for roughly two thirds of the total gain, after which subsequent changes add tenths of a percent each. A second pattern is that PSNR and perceptual metrics decouple: the λ_dssim schedule changes are PSNR-neutral or slightly negative but materially improve SSIM and LPIPS.

## 6. Discussion

The most informative finding is that the released FastGS training recipe is well-calibrated *outside* the densification controller. Every attempt to perturb the optimizer schedule (cosine LR with warmup), view sampling (hard view mining), or pruning budget (relaxation, larger budget) was neutral or negative. By contrast, every change inside the densification path — gradient-threshold decay, finer split factor, lower or decaying importance threshold, anisotropy-guided pruning — yielded a positive contribution. This localizes the remaining headroom in the released recipe to the multi-view consistency machinery, which is also the headline contribution of the paper.

A second observation is that PSNR and perceptual metrics moved together up to a point and then decoupled. The first six changes lifted all three metrics; the λ_dssim schedule changes (forward 0.05 → 0.40 and reverse 0.40 → 0.05) explicitly traded structural similarity against pixel accuracy. The +5.53% SSIM and −29.16% LPIPS gains are large relative to the +1.54% PSNR gain and indicate that the released recipe under-weights perceptual structure on bicycle.

The 5%-over-paper PSNR target of 28.791 dB was not approached, and several of the most promising remaining ideas require modifications outside the scope of this study: AbsGS-style homodirectional gradient accumulation in the CUDA backward pass (estimated +0.3–0.6 dB), Pixel-GS distance-based gradient scaling (+0.3–0.6 dB), depth regularization via Depth-Anything V2 (+0.2–0.8 dB), exposure compensation (+0.5–2.2 dB), FreGS frequency regularization (+0.2–0.5 dB), a dynamic densification interval, and a Lion optimizer swap. These were deprioritised because they require either CUDA-kernel changes or external dependencies that exceeded the available iteration budget.

## 7. Reproducibility

The slimmed repository contains the code required to reproduce the best configuration; pretrained checkpoints and Mip-NeRF 360 datasets are intentionally not included.

* **Best commit.** `dafb00a` (Iteration 17).
* **Best configuration.** All retained changes in the methodology section: gradient-threshold decay (0.3× → 1.0×), `N=3` split, importance-score threshold `>3`, decaying threshold (2 → 5), edge-aware multi-view scoring, anisotropy-guided pruning, reverse λ_dssim schedule (0.40 → 0.05). All other hyperparameters at their released values.
* **Datasets.** Mip-NeRF 360 scenes, single-scene evaluation on *bicycle* in this study. The full evaluation across all Mip-NeRF 360 scenes is supported by `full_eval.py`.
* **Environment.** As documented in the original `README.md`. The `submodules/` directory is required for the rasterizer and is preserved in the slimmed repository; do not run a `git clean -fd` after fresh setup.
* **Entry points.** `train.py` (training), `render.py` (rendering), `metrics.py` and `full_eval.py` (evaluation), `convert.py` (data conversion). Reference configs in `train_base.sh` and `train_big.sh`.

## 8. References

* FastGS (CVPR 2026): *Training 3D Gaussian Splatting in 100 Seconds*. (Original `README.md` should be consulted for the canonical citation.)
* Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. (Foundation method.)
* AutoSOTA: Tsinghua FIB Lab. *AutoSOTA: An automated SOTA-chasing harness*. [github.com/tsinghua-fib-lab/AutoSOTA](https://github.com/tsinghua-fib-lab/AutoSOTA).
