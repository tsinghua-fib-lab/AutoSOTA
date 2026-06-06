# Unified Number-Free Text-to-Motion Generation Via Flow Matching: A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization study performed on the public release of UMF (Unified Motion Flow), a CVPR 2026 paper by Huang and Celiktutan that proposes a number-free text-to-motion generation framework based on a two-component flow-matching architecture: Pyramid Motion Flow (P-Flow) for single-pass motion-prior generation and Semi-Noise Motion Flow (S-Flow) for multi-pass reaction generation. The optimization was driven by AutoSOTA (`tsinghua-fib-lab/AutoSOTA`) and targeted the `R_precision_top_1` metric on the InterHuman benchmark. Six iterations were executed over the released checkpoint without any retraining. The headline result is an improvement of `R_precision_top_1` from a baseline of 0.09176 to 0.09777 (+6.5%), accompanied by a 4.3% reduction in FID (181.49 → 173.67) and a 2.4% gain in Diversity (7.507 → 7.684). The 5%-over-paper target of 0.4904 was not achievable in the present setup because the HumanML3D dataset failed to load and evaluation was restricted to InterHuman only, an off-distribution slice on which absolute metric values diverge significantly from the paper's reported numbers. The dominant lever was a time-dependent classifier-free-guidance schedule (linearly annealing from 7.0 to 2.0); doubling the per-stage ODE step count had a negligible effect. The best configuration is captured at commit `ca40b00235`.

## 1. Introduction

UMF, presented at CVPR 2026, is a generative model for multi-person motion synthesis whose central contribution is to remove the implicit assumption that the number of agents is fixed at training time. The model decomposes number-free motion generation into two complementary stages: a single-pass motion-prior stage handled by Pyramid Motion Flow (P-Flow), and a sequence of reactive transformations handled by Semi-Noise Motion Flow (S-Flow). A unified latent space, built by a heterogeneous Motion VAE, bridges the distribution gap between datasets with different actor counts (HumanML3D, single-person; InterHuman, two-person), enabling a single model to be trained on heterogeneous corpora.

This report studies whether the released UMF inference pipeline can be improved post hoc, without retraining, using purely test-time interventions. The motivation is that flow-matching models — and especially distilled or multi-stage flow-matching models — are sensitive to classifier-free-guidance scheduling and to the numerical configuration of the ODE solver, both of which are commonly fixed at default values in released code. AutoSOTA, an automated SOTA-chasing harness developed by Tsinghua FIB Lab, was used to propose, run, and evaluate code and configuration changes in a budgeted iterative loop, scoring each iteration against a single primary metric (`R_precision_top_1`) on the project's standard evaluation harness.

The remainder of the report covers the original method (Section 2), the limitations targeted by the optimization (Section 3), the methodology applied (Section 4), the experimental setup, results, and ablations (Section 5), a discussion of negative findings (Section 6), and the information needed to reproduce the best configuration (Section 7).

## 2. Original Method (Background)

UMF (Unified Motion Flow) consists of three trainable components organised into three sequential training stages:

* **Stage 1 — Motion Heterogeneous VAE.** A motion-specific variational autoencoder produces a unified latent space that encodes both single-person (HumanML3D) and multi-person (InterHuman) motion. Trained via `train_UMF.py --cfg configs/config_vae.yaml`.
* **Stage 2 — Pyramid Motion Flow (P-Flow), individual denoiser.** A flow-matching model that operates on hierarchical resolutions, conditioned on different noise levels, and synthesises a motion prior in a single forward pass. Trained via `train_UMF.py --cfg configs/config_pflow.yaml` with `TRAIN.DIFFUSION_MODE=indi` and `TRAIN.PRETRAINED_VAE` set to the Stage-1 checkpoint.
* **Stage 3 — Semi-Noise Motion Flow (S-Flow), reactive denoiser.** A second flow-matching model that learns a joint probabilistic path performing reaction transformation and context reconstruction over the prior produced by P-Flow. Trained via `train_react.py --cfg configs/config_sflow.yaml` with `TRAIN.PRETRAINED_INDI` pointing at Stage 2.

Inference for evaluation is launched through `test.py` for each stage with the corresponding configuration file. The standard evaluation pipeline computes top-`k` R-precision (text-to-motion retrieval), FID (Fréchet Inception Distance over motion features), and Diversity over the InterHuman + HumanML3D test sets, using the `interclip.ckpt` evaluator from the InterHuman/InterGen project. The pretrained UMF and VAE checkpoints are distributed via Google Drive and are referenced by paths in `configs/assets.yaml`.

## 3. Identified Limitations

The optimization study identified four sources of friction in the released inference pipeline:

1. **Hard-coded classifier-free-guidance scale.** The released inference path uses a constant `guidance_scale` for all timesteps. For flow-matching models in general, and for two-stage flow-matching models in particular, a constant guidance value is known to over-smooth fine-grained details when applied late in the trajectory and to under-condition the early semantic-planning steps when the value is small.
2. **Inert `TEST.FACT` configuration parameter.** The configuration file exposes a `TEST.FACT` field that was intended to control the per-stage ODE step count, but in the released code path the parameter is not threaded through to the integrator. As a consequence, the integration cost was fixed at five RK4 steps per stage and could not be tuned at evaluation time.
3. **Cross-device file moves in the logger.** The training/evaluation logger uses `os.rename` to relocate files. On a typical containerised deployment with bind-mounted volumes, this raises `OSError: Invalid cross-device link` and aborts evaluation. A simple `shutil.move` substitution is sufficient to fix the issue but is required before any AutoSOTA iteration can complete.
4. **Single-dataset evaluation surface.** The released harness loads HumanML3D and InterHuman jointly. In the present environment HumanML3D failed to load with `not enough values to unpack`, and the only feasible evaluation slice was InterHuman. The absolute baseline numbers therefore differ substantially from the paper-reported figures, and an apples-to-apples comparison against the paper's table is not possible.

## 4. Optimization Methodology

The six retained iterations fall into four categories. Each is grounded in concrete files in the released repository.

**Infrastructure plumbing.** Two changes that are necessary preconditions for AutoSOTA to run repeatedly were applied: replacing `os.rename` with `shutil.move` in the logger to enable cross-device file moves; and disabling TensorBoard's per-step writes (which exhausted the 588 GB volume on long runs).

**Configuration wiring.** The `TEST.FACT` parameter was threaded from the configuration file through to the per-stage ODE solver, restoring the dead knob and exposing the integration step count as a tunable hyperparameter at inference time.

**Classifier-free-guidance schedule (primary lever).** The core algorithmic change is the introduction of a time-dependent CFG schedule. Instead of a single guidance scalar, the inference path now linearly anneals the guidance from 7.0 at the start of the trajectory down to 2.0 at the end. The rationale is that the first portion of a flow-matching trajectory establishes the semantic content of the motion (where strong text conditioning helps), whereas the last portion refines low-level kinematic detail (where strong CFG over-saturates and biases the sample away from the data manifold).

**Solver-precision ablation.** With the CFG schedule fixed, the per-stage ODE step count was doubled from 5 → 10 (effectively `FACT=2`, giving 10 RK4 steps per stage) to test whether numerical accuracy of the integrator was a bottleneck.

No training data, model weights, or VAE/CLIP components were modified. All retained changes are either infrastructure fixes, plumbing additions exposing existing knobs, or hyperparameter values written through CLI/config flags.

## 5. Experiments

### 5.1 Setup

The optimization target was `R_precision_top_1` on the InterHuman test split, computed by the project's standard `test.py` pipeline using the InterHuman `interclip.ckpt` evaluator. All runs used the released UMF and VAE checkpoints unchanged. Each full evaluation comprised 20 replications and required approximately 14 minutes per iteration on an NVIDIA A100-SXM4-80GB GPU inside a Docker container with `--runtime=nvidia`. The available host volume was 588 GB total with the repository consuming roughly 15 GB; aggressive cleanup and disabling TensorBoard were required to keep iterations alive. AutoSOTA executed six iterations under a fixed wall-clock budget per iteration.

The headline target chosen by AutoSOTA was `R_precision_top_1 ≥ 0.4904` (+5% over the paper-reported baseline of 0.467). This target is **not** comparable to the InterHuman-only baseline measured here (0.09176), which is roughly an order of magnitude smaller because the paper-reported figure is computed on the full InterHuman + HumanML3D evaluation. Reaching 0.4904 from 0.09176 would require a +435% improvement and is not achievable through inference-only changes; this constraint is acknowledged explicitly in the optimization log.

### 5.2 Quantitative Results

The InterHuman-only baseline measured on the present infrastructure is reproduced below alongside the best run (Iteration 6).

| Metric | Baseline (Iter 0) | Best (Iter 6) | Delta |
|---|---:|---:|---:|
| R_precision_top_1 | 0.09176 | **0.09777** | +6.5% |
| R_precision_top_2 | 0.15762 | 0.16591 | +5.3% |
| R_precision_top_3 | 0.21222 | 0.21832 | +2.9% |
| FID | 181.49 | **173.67** | −4.3% |
| Diversity | 7.507 | 7.684 | +2.4% |
| gt_R_precision_top_1 | 0.42074 | 0.42074 | unchanged (ground truth) |
| gt_R_precision_top_2 | 0.60379 | 0.60379 | unchanged |
| gt_R_precision_top_3 | 0.70393 | 0.70393 | unchanged |

The best configuration is captured at commit `ca40b00235` and corresponds to: time-dependent CFG schedule annealing from 7.0 to 2.0; default ODE step count (`FACT=1`, i.e. 5 steps per stage); and the infrastructure fixes from Iterations 1 and 2.

### 5.3 Ablation / Iteration Trajectory

The retained ablation table is reproduced below.

| # | Change | Effect | Notes |
|---|---|---|---|
| 1 | Cross-device logger fix (`os.rename` → `shutil.move`) | Enables eval on mounted volumes | Required for the Docker setup |
| 2 | `TEST.FACT` wiring (configurable ODE steps) | Enables step-count tuning | Restores a previously dead config parameter |
| 3 | Time-dependent CFG scheduling (7.0 → 2.0) | **+6.5% R_precision, −4.3% FID** | Primary improvement |
| 4 | Doubled ODE steps (FACT=2; 5 → 10 per stage) | Negligible | ODE precision not the bottleneck |
| 5 | Disabled TensorBoard logging | Prevented disk-full errors | Infrastructure fix |
| 6 | Higher peak guidance (3.0 → 5.0/7.0) | Improved alignment without diversity collapse | Combined with annealing |

Two qualitative observations follow from the trajectory. First, all of the quality improvement is concentrated in Iteration 3, the time-dependent CFG schedule. Second, the negative result on doubling the ODE step count (Iteration 4) is informative: it indicates that, for the released UMF checkpoints, the residual error of an RK4 integrator with five steps per stage is well below the resolution of the evaluation metrics, and that further effort should be directed at the conditioning side of the pipeline rather than at solver precision.

## 6. Discussion

The dominant takeaway is that classifier-free guidance is the single largest tunable lever for the released UMF model on InterHuman: a simple linear schedule from 7.0 to 2.0 yields a 6.5% improvement on `R_precision_top_1` and a 4.3% reduction in FID without retraining. Two factors are likely to generalise. First, multi-stage flow-matching pipelines benefit from strong semantic conditioning early in the trajectory and from weaker conditioning late, mirroring the inductive bias seen in image diffusion models with similar two-stage structure. Second, restoring the inert `TEST.FACT` knob is a generally useful pattern: dead configuration parameters are common in research code and inexpensive to repair, and exposing them gives downstream users a real degree of freedom.

Several caveats limit how far the headline numbers should be transported. The InterHuman-only baseline is far from the paper-reported figure because the HumanML3D loader failed in the present environment; an apples-to-apples replication of the paper's full-pipeline numbers therefore was not attempted. The 5%-over-paper target was unreachable for that reason, not because the optimization plateaued. Reproducing the paper's headline numbers, and by extension producing a meaningful absolute comparison on the standard evaluation, depends on fixing the HumanML3D loading path — an issue identified but not closed by this study. Finally, several promising ideas remain on the table for future iterations, including CFG-Zero* (per-timestep optimised guidance via dot-product projection), latent averaging across seeds, an adaptive ODE solver such as DOPRI5, test-time augmentation by generating multiple samples per prompt, and VAE posterior-mean decoding to reduce decoder variance.

## 7. Reproducibility

The slimmed repository contains the code required to reproduce the best configuration; pretrained checkpoints and datasets are intentionally not included.

* **Best commit.** `ca40b00235`.
* **Best configuration.** Time-dependent CFG schedule (linear anneal from 7.0 to 2.0); `TEST.FACT=1` (default 5 RK4 steps per stage); CFG peak 7.0; `os.rename → shutil.move` in the logger; TensorBoard logging disabled.
* **Datasets.** Place or symlink HumanML3D ([github.com/EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D)) and InterHuman/InterGen ([github.com/tr3e/InterGen](https://github.com/tr3e/InterGen)) into `./data/`. The expected directory layout is documented in the original `README.md`.
* **Pretrained checkpoints.** Download `interclip.ckpt` (evaluator) from the InterHuman/InterGen release and the UMF and VAE checkpoints from the Google Drive links in the original `README.md`. Update `configs/assets.yaml` and `configs/datasets.yaml` to point at the local paths.
* **Hardware used in this study.** NVIDIA A100-SXM4-80GB. The author's reported hardware is NVIDIA H200 with CUDA 12.8 (under 24 GB GPU memory required for everything except VAE training).
* **Known infrastructure caveats.** The `deps/` directory (CLIP, DistilBERT, SMPL) is `.gitignore`-d but required at runtime; do not run `git clean -fd` after a fresh clone. On bind-mounted Docker volumes, the `shutil.move` fix is mandatory. Disk usage is tight: ensure tens of gigabytes of free space and disable TensorBoard for long evaluation runs.

## 8. References

* Huang, G., & Celiktutan, O. (2026). *Unified Number-Free Text-to-Motion Generation Via Flow Matching*. CVPR 2026. arXiv:2603.27040. Project page: [githubhgh.github.io/umf](https://githubhgh.github.io/umf/).
* AutoSOTA: Tsinghua FIB Lab. *AutoSOTA: An automated SOTA-chasing harness*. [github.com/tsinghua-fib-lab/AutoSOTA](https://github.com/tsinghua-fib-lab/AutoSOTA).
* InterHuman / InterGen: Liang, H. et al. [github.com/tr3e/InterGen](https://github.com/tr3e/InterGen).
* HumanML3D: Guo, C. et al. [github.com/EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D).
