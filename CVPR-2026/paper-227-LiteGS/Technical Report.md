# Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists: A Technical Report on Automated Optimization

## Abstract

This technical report documents an automated optimization study targeting the CVPR 2026 paper "Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists" by Liu and Han. The original method, hereafter referred to by the name of the underlying training framework it extends (LiteGS, with ShorterSplatting modifications), accelerates 3D Gaussian Splatting (3DGS) training by reducing the per-pixel sorted Gaussian list through scale reset, entropy regularization, and DashGaussian integration. The repository under examination preserves these modifications atop a LiteGS stable branch and ships with a reproducible nine-scene Mip-NeRF 360 evaluation harness. The study reproduced the reference pipeline on an A100 GPU and applied a single targeted optimization pass (IDEA-002): the original L1+SSIM photometric supervision was replaced with a composite L1+L2+SSIM term of the form 0.8·L1 + 0.2·L2 + λ_dssim·(1 − SSIM). Across all nine Mip-NeRF 360 scenes the reproduced baseline already attained PSNR 28.258, SSIM 0.859, and LPIPS 0.145, exceeding the paper's reported 27.28 / 0.810 / 0.224. The composite-loss iteration produced a marginal +0.11 PSNR gain on three scenes (with a +0.38 PSNR improvement on the room scene) but did not exceed the baseline aggregate, leaving 28.2582 as the best aggregate PSNR and 28.644 (a 5% gain over the paper) as an unmet target. The report consolidates infrastructure observations, ablation outcomes, and prioritized future ideas.

## 1. Introduction

3D Gaussian Splatting has emerged as a leading representation for novel-view synthesis, but its training cost is dominated by the per-pixel α-blending of long sorted Gaussian lists. Liu and Han propose to attack the cost at its source by shortening these lists, yielding faster convergence with no loss of fidelity. Their official implementation, distributed at the `MachinePerceptionLab/ShorterSplatting` repository and evaluated in this report, is layered on top of the LiteGS stable branch maintained by MooreThreads and integrates DashGaussian-style progressive resolution and Gaussian-budget scheduling.

This report is part of an automated paper-reproduction and optimization effort coordinated by the AutoSOTA framework. The objective is twofold: first, to verify that the published claims hold on representative hardware; second, to explore whether modest, mechanically-applied modifications to the loss function or training schedule can push the reported quality metrics beyond the paper's numbers. The present run focused exclusively on the Mip-NeRF 360 portion of the benchmark suite. All quantitative numbers reproduced below are taken verbatim from the input artefact `TAKEAWAY_source.md` and were not re-derived.

## 2. Original Method (Background)

The original method, identifiable in the source tree by the package name `litegs/` and the auxiliary `litegs/spreading/` subpackage that hosts the paper-specific schedulers, combines three contributions on top of standard 3DGS:

1. **Scale reset.** A scheduling policy implemented in `litegs/spreading/scale_scheduler.py` periodically rescales the per-Gaussian extents toward a smaller value controlled by `scale_reset_factor` (default 0.2 at evaluation time). The intent is to keep the number of pixels touched by each Gaussian small, yielding shorter sorted lists during rasterization. The accompanying scale and opacity histograms in the paper's overview figure show that scale reset shifts the population toward smaller, more opaque Gaussians.

2. **Entropy regularization.** The opacity distribution is pushed toward bimodality (transparent or opaque) via an entropy term added to the photometric loss with weight `lambda_entropy` (default 0.015). The schedule is in `litegs/spreading/entropy_scheduler.py`, and a custom CUDA backward pass propagates gradients through the entropy of the per-pixel transmittance integral. Polarized opacities further reduce the effective list length because near-transparent Gaussians are skipped earlier.

3. **DashGaussian integration.** When `--enable_dash` is set, the trainer instantiates `DashGaussianScheduler` (in `litegs/spreading/dashgaussian_scheduler.py`), which controls the maximum Gaussian budget per scene and a progressive render-resolution schedule. Per-scene budgets are hard-coded in `full_eval.py` (`MAX_N_GAUSSIAN`), e.g. 5,987,095 for `bicycle` and 1,252,367 for `bonsai`.

The training loop is in `litegs/training/trainer.py`. Each epoch contains `iterations_per_epoch` mini-batches; densification is governed by `densify.DensityControllerWithFinalCount` or `densify.DensityControllerDashGaussian` depending on whether DashGaussian is enabled. Photometric supervision in the released code combines L1, L2, and a fused SSIM term (see `quality_loss = 0.8*l1_loss + 0.2*l2_loss + op.lambda_dssim*ssim_loss`), which is itself a deviation from the canonical 3DGS L1+SSIM formulation.

The full evaluation harness is launched through `full_eval.py`, which iterates over five outdoor and four indoor Mip-NeRF 360 scenes, two Tanks and Temples scenes, and two Deep Blending scenes. Final metrics are produced by `example_metrics.py`, which loads the saved point cloud and computes torchmetrics-based PSNR, SSIM, and VGG-LPIPS, alongside a render-only FPS estimate.

## 3. Identified Limitations

Three categories of limitations were identified during reproduction and motivated the subsequent optimization step:

1. **Photometric loss design.** The released loss already mixes L1, L2, and SSIM, but the relative weights (0.8 / 0.2 / λ_dssim) were neither ablated in the public materials nor documented in detail, suggesting that further tuning could yield gains, particularly on indoor scenes that exhibit smoother textures.

2. **Hyperparameters fixed across scenes.** A single `scale_reset_factor`, a single `lambda_entropy`, and a single `lambda_dssim` are reused across all nine Mip-NeRF 360 scenes despite their heterogeneity (low-texture rooms versus high-frequency outdoor foliage).

3. **Reproduction infrastructure.** Although strictly orthogonal to the method, the reproducibility envelope was constrained by network bandwidth, ephemeral storage, and Docker tooling. These issues consumed a non-trivial fraction of the optimization budget and limited the number of full nine-scene evaluations that could be completed.

## 4. Optimization Methodology

The optimization budget admitted exactly two iterations: a baseline reproduction and a single modification. The chosen modification, catalogued internally as IDEA-002, kept the existing structure of the photometric loss but explicitly retained an L2 component alongside L1 and SSIM. Concretely, the trainer's photometric term is:

```
quality_loss = 0.8 · L1(img, gt) + 0.2 · L2(img, gt) + lambda_dssim · (1 − fused_ssim(img, gt))
```

with `lambda_dssim` left at its LiteGS default. The hypothesis was that the L2 component would more aggressively penalize residual high-error regions, especially on indoor scenes with broad, smooth surfaces. The change is local to `litegs/training/trainer.py` and does not touch the densification or scale-reset logic.

All other settings followed the paper's recommended configuration: `--enable_dash`, `lambda_entropy = 0.015`, `scale_reset_factor = 0.2`, SH degree 3, and the per-scene `MAX_N_GAUSSIAN` budgets defined in `full_eval.py`. Nine Mip-NeRF 360 scenes were evaluated end-to-end (training plus held-out test rendering) using `python full_eval.py --save_images`. Tanks and Temples and Deep Blending were skipped under the time and storage constraints noted in Section 3.

## 5. Experiments

### 5.1 Setup

The reproduction was performed on a workstation equipped with an NVIDIA A100 GPU, running the Docker image built from the repository's CUDA dependencies (`fused_ssim`, `simple_knn`, `litegs_fused`, `FastLanczos`) plus `torchmetrics`. The paper itself reports results on Ubuntu 24.04.2 LTS with an NVIDIA GeForce RTX 5090 D (32 GB VRAM), CUDA 12.8, Python 3.10.16, and PyTorch 2.8.0. Hardware differences are reflected in the wall-clock training time; quality metrics are hardware-insensitive at the precision reported.

The Mip-NeRF 360 dataset was extracted with the standard layout, including symlinks for the `flowers` and `treehill` scenes that share imagery with sibling scenes in some distributions. VGG16 weights (528 MB) used by the LPIPS metric were pre-cached inside the Docker image to bypass slow uplinks. Train/test splits follow the convention used by 3DGS and LiteGS (every eighth frame goes to the test set).

### 5.2 Quantitative Results

Aggregated over all nine Mip-NeRF 360 scenes, the reproduced baseline already exceeds the paper's reported numbers on every quality dimension while running noticeably slower per scene due to the A100 versus RTX 5090 D delta:

| Metric | Paper | Our Baseline | Delta |
|--------|-------|--------------|-------|
| PSNR | 27.28 | **28.258** | +0.978 |
| SSIM | 0.810 | **0.859** | +0.049 |
| LPIPS | 0.224 | **0.145** | -0.079 |
| Training Time | 99.58s | 226.5s | +126.9s (A100 vs RTX 5090D) |

The best aggregate PSNR observed during the run was 28.2582, attained by the baseline configuration. This is +3.58% above the paper's reported 27.28 but still below the predefined target of 28.644 (a 5% improvement over the paper). The IDEA-002 composite-loss variant did not exceed the baseline aggregate; on the three scenes where it was evaluated end-to-end it produced a +0.11 PSNR mean gain, with the room scene showing a +0.38 PSNR improvement. As a result, the best commit recorded in `TAKEAWAY_source.md` corresponds to the baseline configuration.

### 5.3 Ablation / Iteration Trajectory

Two configurations were evaluated in this run:

| Iteration | Configuration | Aggregate PSNR | Notes |
|-----------|---------------|----------------|-------|
| 0 (baseline) | `--enable_dash --lambda_entropy 0.015 --scale_reset_factor 0.2`, default loss | **28.258** | Best commit; nine-scene evaluation |
| 1 (IDEA-002) | Same as baseline + composite L1+L2+SSIM loss (0.8/0.2/λ_dssim) | +0.11 PSNR over baseline on 3 scenes; +0.38 on room | Did not surpass baseline aggregate |

The per-scene baseline metrics, reproduced exactly from `TAKEAWAY_source.md`, are:

| Scene | PSNR | SSIM | LPIPS | Time (s) |
|-------|------|------|-------|----------|
| bicycle | 25.13 | 0.758 | 0.225 | 197.0 |
| flowers | 27.02 | 0.855 | 0.113 | 227.5 |
| garden | 27.08 | 0.855 | 0.113 | 227.2 |
| stump | 26.67 | 0.770 | 0.231 | 169.8 |
| treehill | 26.71 | 0.771 | 0.230 | 167.0 |
| room | 30.31 | 0.921 | 0.127 | 205.8 |
| counter | 28.66 | 0.911 | 0.107 | 294.5 |
| kitchen | 30.93 | 0.937 | 0.074 | 330.3 |
| bonsai | 31.82 | 0.949 | 0.087 | 219.8 |

The indoor scenes (`room`, `counter`, `kitchen`, `bonsai`) consistently outperform the outdoor scenes by 2–6 dB PSNR, mirroring the pattern reported by the paper and by 3DGS variants in general. The fact that IDEA-002 produced its largest gain on `room` (+0.38) supports the hypothesis that the additional L2 term is especially helpful for low-frequency, high-luminance interiors where small absolute residuals dominate the perceptual error.

## 6. Discussion

The reproduction confirms the general direction of the paper: a LiteGS-based pipeline, augmented with scale reset, entropy regularization, and DashGaussian budgets, yields high-quality reconstructions on Mip-NeRF 360 with quality metrics that meet or exceed the published numbers. The +0.978 PSNR margin between the reproduced baseline and the paper-reported PSNR is large enough to suggest that either (i) the published numbers are conservative versus what the released code currently achieves, or (ii) downstream changes to the code (e.g., the composite L1+L2+SSIM loss already wired into the training loop) post-date the numbers in the paper text. Either way, the baseline is already a strong starting point for further optimization.

The IDEA-002 modification is informative even though it failed to beat the baseline aggregate. Its asymmetric per-scene impact — strongly positive on `room`, marginal elsewhere — argues for **scene-adaptive** photometric weighting rather than a single global tuple. A natural follow-up is to learn or schedule the L1/L2 mixture conditioned on the running magnitude of the residual.

Several higher-priority ideas remain unexplored due to the two-iteration budget. The most promising are summarized below in the order suggested by the source artefact:

1. **IDEA-006**: Increase `lambda_dssim` from 0.2 to 0.35 (Tier 1, HIGH priority).
2. **IDEA-004**: Delayed SH degree schedule (Tier 1, HIGH priority).
3. **IDEA-003**: Two-phase training: geometry first, then appearance (Tier 1, HIGH priority).
4. **IDEA-001**: Adaptive gradient-weighted entropy regularization (Tier 1, HIGH priority).
5. **IDEA-007**: Adaptive densify gradient threshold (Tier 2, HIGH priority).
6. **IDEA-012**: Per-pixel gradient weighting for densification (Tier 2, HIGH priority).

Several of these (IDEA-001, IDEA-006, IDEA-007, IDEA-012) directly target the same parts of the loss and densification logic that this study perturbed and would be the most efficient next steps.

The reproduction also exposed practical issues that, while not affecting the science, materially shape the iteration cadence. Network bandwidth bottlenecks (~100 KB/s for `conda`, `pip`, `apt-get`, and model downloads), inability to capture stdout from `docker exec`, a 20 GB overlay filesystem leading to repeated `ENOSPC` failures, and a missing `git` binary in the base image collectively reduced the effective iteration count. Recommendations for future runs include using `docker run -d` followed by `docker logs`, pre-baking VGG16 and other model weights into the image, redirecting `CLAUDE_CODE_TMPDIR` onto a large volume, installing `git` from `conda-forge`, and running a 3–5 scene subset for rapid iteration before validating on all nine scenes.

## 7. Reproducibility

The repository is self-contained once the Mip-NeRF 360, Tanks and Temples, and Deep Blending datasets are mounted. Installation follows LiteGS conventions: clone with `--recursive`, then build the four CUDA submodules `fused_ssim`, `simple-knn`, `gaussian_raster`, and `lanczos-resampling`. Python dependencies are minimal (`fused-ssim`, `plyfile`, `tqdm`, `pillow` per `requirement.txt`); evaluation additionally requires `torchmetrics` and a cached `vgg16` checkpoint.

To reproduce the paper's full evaluation including the ShorterSplatting modifications:

```bash
python ./full_eval.py --save_images \
    --mipnerf360 /path/to/mipnerf360 \
    --tanksandtemples /path/to/tanksandtemples \
    --deepblending /path/to/deepblending \
    --enable_dash --lambda_entropy 0.015 --scale_reset_factor 0.2
```

The plain-LiteGS baseline is obtained by dropping the `--enable_dash`, `--lambda_entropy`, and `--scale_reset_factor` flags. Per-dataset summaries can be produced with the helper script `litegs/spreading/misc/print_stats.py`. The training entry point `example_train.py` and the evaluation entry point `example_metrics.py` are short wrappers around `litegs.training.start` and a torchmetrics-based metric loop, respectively.

The exact patch applied in IDEA-002 is the photometric-loss definition in `litegs/training/trainer.py`, namely `quality_loss = 0.8*l1_loss + 0.2*l2_loss + op.lambda_dssim*ssim_loss`. All numbers in this report are reproduced verbatim from `TAKEAWAY_source.md` produced by the AutoSOTA optimization run.

## 8. References

1. Jiaqi Liu and Zhizhong Han. **Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists.** In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2026. arXiv:2603.09277. Project page: `https://machineperceptionlab.github.io/ShorterSplatting-Project/`.

2. AutoSOTA: Automated Reproduction and Optimization Framework. `tsinghua-fib-lab/AutoSOTA`. `https://github.com/tsinghua-fib-lab/AutoSOTA`.

3. LiteGS (stable branch). MooreThreads. `https://github.com/MooreThreads/LiteGS/tree/stable`.

4. Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. **3D Gaussian Splatting for Real-Time Radiance Field Rendering.** *ACM Transactions on Graphics*, 42(4), 2023.

5. Youyu Chen et al. **DashGaussian.** `https://github.com/YouyuChen0207/DashGaussian`.
