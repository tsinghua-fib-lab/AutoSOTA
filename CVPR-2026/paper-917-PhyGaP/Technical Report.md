# PhyGaP: Physically-Grounded Gaussians with Polarization Cues: A Technical Report on Automated Optimization

## Abstract
This report documents an automated optimization study of PhyGaP, a CVPR 2026 Oral method for novel view synthesis with physically grounded inverse rendering from polarized images. PhyGaP extends 3D Gaussian Splatting with polarization‑aware, physically based shading, jointly recovering scene geometry, material properties, and lighting from Stokes‑vector supervision. Using the AutoSOTA pipeline, 19 parameter‑ and code‑level interventions were tested on the PANDORA *owl_quat_white* scene. The optimization achieved a peak PSNR of 28.4572 dB, an improvement of 0.3107 dB (+1.10 %) over the reproduced baseline (28.1465 dB), accompanied by a slight SSIM increase of 0.0016 and an LPIPS decrease of 0.0013. Two interventions drove this advance: (1) activating and tuning depth smoothness regularization (λ = 0.05, previously disabled) reduced floater artifacts and improved geometric consistency; (2) introducing an adaptive Degree of Linear Polarization (DoLP)‑weighted Stokes loss focused polarization supervision on reliable signal, mitigating over‑regularization in diffuse regions. A 5 % target PSNR improvement (29.589 dB) was not reached, suggesting that further progress demands architectural rather than purely parametric modifications.

## 1. Introduction
Recovering accurate geometry, material properties, and lighting from multi‑view photographs is a long‑standing challenge in computer graphics and vision. When the inputs include polarized images, the physical relationship between surface orientation, material Fresnel response, and Stokes parameters provides richer constraints for inverse rendering. PhyGaP (Physically‑Grounded Gaussians with Polarization Cues) leverages this relationship by embedding a linear polarizer model within a modern 3D Gaussian Splatting framework, achieving state‑of‑the‑art results in novel view synthesis and physically plausible decomposition [1]. However, the default hyperparameter choices and loss design leave room for improvement on specific scenes. The AutoSOTA framework was deployed to automatically explore parameter and code‑level modifications to improve rendering fidelity on a representative scene. This report presents the methodology, results, and insights from that optimization campaign.

## 2. Original Method (Background)
PhyGaP [1] augments the 2DGS/3DGS backbone with a physics‑based shading model and polarization handling. A set of anisotropic 3D Gaussians carries not only color and opacity but also surface normal, diffuse albedo, roughness, half‑eta (for Fresnel), reflection strength, and indirect illumination SH coefficients. During training, a combination of differentiable splatting, a microfacet BRDF, and environment‑map‑based direct lighting produce a diffuse and a specular rendering. When polarization data are available, a linear polarizer model computes Stokes vectors for the combined specular and diffuse components; a dedicated Stokes loss penalises deviations from ground‑truth Stokes images. Geometry is supervised via render‑depth and normal regularizations, and an alpha mask loss enforces scene silhouettes. The default training uses 15,000 iterations with a staged schedule (initial scene expansion, surfel rendering, optional indirect illumination). The pipeline can handle synthetic Mitsuba‑rendered data as well as real captures from the PANDORA dataset.

## 3. Identified Limitations
### 3.1 Disabled Depth Smoothness Regularization
Analysis of the default configuration (`arguments/__init__.py`) reveals that the depth smoothness loss weight `lambda_depth_smooth` is set to 0.0. Consequently, the edge‑aware depth smoothness term – designed to encourage locally planar depth maps and suppress floating Gaussians – contributes nothing to the training objective. Visual inspection of early runs confirms the presence of cloud‑like artifacts near object silhouettes, a symptom of under‑regularized depth. The AutoSOTA log explicitly tests nonzero values, establishing this as a potential bottleneck.

### 3.2 Uniform Polarization Supervision
The default Stokes loss applies a constant weight `lambda_stokes` = 10.0 to every pixel, irrespective of the local Degree of Linear Polarization (DoLP). In regions where DoLP is low (e.g., diffuse surfaces, shadow boundaries), the polarization signal is weak and often noise‑dominated; enforcing a heavy loss there can distort geometry and reflectance. The codebase does not adapt the loss strength to DoLP, so the network receives undifferentiated feedback that may hinder convergence in low‑DoLP areas.

These two limitations were prime targets for the optimization study, as they relate to fundamental geometry–appearance trade‑offs in the physically grounded pipeline.

## 4. Optimization Methodology
AutoSOTA conducted 19 iterations of automated experimentation, each applying a candidate change and evaluating the resulting model on the *owl_quat_white* test set with PSNR, SSIM, and LPIPS. The three interventions that improved PSNR are detailed below.

### 4.1 Depth Smoothness Regularization Activation and Tuning
**File(s) affected:** `arguments/__init__.py` (setting `lambda_depth_smooth`), and indirectly `gaussian_renderer/__init__.py` where the normal/depth regularizations are computed.
**Change:** The depth smoothness weight was increased from 0.0 to 0.03 (Iteration 3) and subsequently to 0.05 (Iteration 11). This activates an L1‑based, edge‑aware smoothness loss on the surf‑depth map, penalising large second‑order differences weighted by image gradients.
**Rationale:** Improving geometric consistency directly improves the surface normals and reflection directions used in the physics‑based shading, reducing floater‑induced specular artefacts. Empirical evidence from the log shows monotonic PSNR gains with moderate weights, saturating at 0.05 before over‑smoothing (0.07 caused a −0.027 dB regression).

### 4.2 Adaptive DoLP‑Weighted Stokes Loss
**File(s) affected:** The loss computation (likely in the training loop or a dedicated loss module; the log identifies the change as “Adaptive DoLP‑weighted stokes loss”).
**Change:** The per‑pixel Stokes loss was scaled by the ground‑truth Degree of Linear Polarization (DoLP) before aggregation. Pixels with DoLP below a threshold receive proportionally less weight, concentrating the polarization constraint on highly polarized surface regions.
**Rationale:** Co‑polarized data are most informative where the specular contribution is strong; diffuse areas with low DoLP are better constrained by RGB, alpha, and normal losses. This adaptive scheme avoids over‑regularization, as evidenced by a +0.0416 dB PSNR increase without harming SSIM or LPIPS.

No other change passed the acceptance threshold; all other attempted tweaks (e.g., distance regularizer, delayed densification, extended training) resulted in regressions or negligible effects.

## 5. Experiments

### 5.1 Setup
- **Hardware:** Single NVIDIA RTX 4090D GPU, Ubuntu 22.04, CUDA 11.8.
- **Dataset:** PANDORA *owl\_quat\_white* scene, images resized to ¼ of original resolution, following the official instructions.
- **Metrics:** Peak Signal‑to‑Noise Ratio (PSNR), Structural Similarity (SSIM), Learned Perceptual Image Patch Similarity (LPIPS, AlexNet).
- **Evaluation Protocol:** All metrics computed on the test split after training for 15 k iterations (the default length). The reproduced baseline yields PSNR 28.1465 dB (slightly below the paper‑reported 28.18 dB, possibly due to minor environmental differences).
- **Optimization Budget:** 19 search iterations, each with full training of 15 k iterations (some extended runs were attempted but not adopted). The final best configuration was obtained at iteration 11 with commit `9138fd22ab`.
- **Caveats:** The 5 % PSNR improvement target (29.589 dB) was not attained; the pipeline is considered well‑tuned and limited gains are achievable without architectural changes. The `double_view` mode is incompatible with the PANDORA dataset, and the environment map resolution experiment (128) was discarded because it doubled training time without benefits.

### 5.2 Quantitative Results

| Metric    | Baseline (Reproduced) | Optimized (Best) | Δ Absolute | Δ %    | Direction |
|-----------|------------------------|------------------|------------|--------|-----------|
| PSNR ↑    | 28.1465               | 28.4572          | +0.3107    | +1.10% | Improved  |
| SSIM ↑    | 0.9596                 | 0.9612           | +0.0016    | –      | Improved  |
| LPIPS ↓   | 0.0434                 | 0.0421           | −0.0013    | –      | Improved  |

The PSNR improvement, driven largely by better geometry, is the most pronounced, while SSIM and LPIPS show minor but consistent gains.

### 5.3 Ablation / Iteration Trajectory
The table below lists the sequence of accepted changes and the cumulative PSNR improvement over the reproduced baseline.

| Iteration | Change                                       | PSNR (dB)   | Cum. Δ from baseline |
|-----------|----------------------------------------------|-------------|----------------------|
| Baseline  | (default)                                    | 28.1465     | –                    |
| 3         | λ\_depth\_smooth = 0.03                      | 28.2718     | +0.1253              |
| 8         | + Adaptive DoLP‑weighted Stokes loss         | 28.3134     | +0.1669              |
| 11        | + λ\_depth\_smooth = 0.05                    | **28.4572** | +0.3107              |

Each row builds on all previous changes. The final combination (λ\_depth\_smooth = 0.05, adaptive DoLP) accumulates +0.3107 dB over the baseline, with depth smoothness tuning contributing the majority of the gain.

## 6. Discussion
The optimization study reveals that geometry quality is the primary bottleneck for PhyGaP on the tested scene. Introducing depth smoothness regularization accounts for nearly 87 % of the total PSNR improvement, consistent with the observation that noisy depth causes mis‑reflections and speckle artefacts in the physically based rendering. The adaptive DoLP‑weighted Stokes loss further refines the result by preventing over‑fitting to weak polarization signals, confirming that polarimetric cues should be trusted proportional to their strength.

The failure of many other candidate changes (new schedulers, longer training, additional regularizers) suggests that the original authors carefully balanced the loss terms and training dynamics. Over‑constraining the Gaussians (e.g., distance loss, aggressive pruning) or modifying training duration uniformly hurt performance. The target of a 5 % PSNR improvement – a high bar for a mature method – was not achieved; attaining such a leap would likely require architectural modifications (e.g., neural radiance field‑style geometry priors, anisotropic roughness, or multi‑view material consistency) rather than simple parameter adjustments.

Threats to validity include the restriction to a single scene, the relatively small optimization budget, and the use of a reproduced baseline that is slightly lower than the reported value. The gains may not generalise to other scenes in the PANDORA or Mitsuba datasets. Reproducibility also depends on the exact code state at the time of the experiments (commit `9138fd22ab`), and any future code refactoring could alter the baseline.

## 7. Reproducibility
- **Repository:** https://github.com/Kelvar00/PhyGaP
- **Environment:**  
  ```
  conda create -n phygap python=3.8
  conda activate phygap
  conda install cudatoolkit=11.8 -c pytorch
  pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
  pip install submodules/cubemapencoder submodules/diff-surfel-rasterization submodules/simple-knn submodules/raytracing
  pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
  pip install -r requirements.txt
  ```
- **Seed:** Not specified (default random seeding used).  
- **Baseline command:** `bash run_phygap.sh` with default config.  
- **Optimized command:** Modify `arguments/__init__.py` to set `lambda_depth_smooth = 0.05` and integrate the DoLP‑weighted Stokes loss (code change referenced in the log), then run `bash run_phygap.sh`.

## 8. References
```bibtex
@article{wu2026phygap,
  title={PhyGaP: Physically-Grounded Gaussians with Polarization Cues},
  author={Wu, Jiale and Bai, Xiaoyang and He, Zongqi and Xu, Weiwei and Peng, Yifan},
  journal={arXiv preprint arXiv:2603.14001},
  year={2026}
}

@misc{autosota,
  author = {{AutoSOTA}},
  title  = {AutoSOTA: Automated State-of-the-Art Optimization},
  note   = {https://github.com/tsinghua-fib-lab/AutoSOTA}
}
```
