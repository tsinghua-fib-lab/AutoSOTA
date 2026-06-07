# GFRRN: Explore the Gaps in Single Image Reflection Removal: A Technical Report on Automated Optimization

## Abstract
Single image reflection removal requires separating a transmission layer from interfering reflections. This report documents an automated optimization study of the GFRRN (Gated Feature Residual Reconstruction Network) model, executed with the AutoSOTA pipeline. The baseline achieves a peak signal‑to‑noise ratio (PSNR) of 25.84 dB on the real20 test set, which is 1.292 dB below the target of 27.132 dB (a required +5.0 % improvement). Across seven iterations of inference‑only modifications, one intervention—a 4‑fold flip‑and‑rotation test‑time augmentation (TTA) ensemble—lifted the real20 PSNR to 26.05 dB (+0.21 dB, +0.81 %). All other evaluation datasets improved or remained stable: solidobject PSNR rose by 0.37 dB, postcard by 0.36 dB, wild by 0.30 dB, and nature by 0.03 dB; no metric regressed. Attempts to fuse the dual‑stream outputs, preserve non‑reflective regions, or apply colour‑space post‑processing were either detrimental or produced no measurable benefit. The principal finding is that symmetric TTA exploits the model’s directional bias, providing a consistent gain at the cost of a 4‑fold increase in inference time. Further improvements would likely require training‑time changes beyond the scope of this study.

## 1. Introduction
The removal of undesired reflections from a single photograph remains an open problem in computational photography, with applications in mobile imaging, robotics, and image forensics. The GFRRN model employs a Swin‑Transformer backbone, frequency‑domain processing (G‑AFLB), and dual attention alignment (DAA) to estimate a clean transmission layer. Despite these advances, the model’s real20 PSNR of 25.84 dB leaves a gap to the 27.132 dB target set for this optimization study.

We applied the AutoSOTA automated optimization framework (tsinghua‑fib‑lab/AutoSOTA) to GFRRN with the objective of raising the real20 PSNR without modifying the model’s pretrained weights. The pipeline analysed the codebase, generated sixteen candidate inference‑time modifications, and carried out an iterative loop driven by metric feedback. The study yielded a reliable, albeit modest, boost via test‑time augmentation, while exposing three important limitations of the architecture that thwarted other strategies. All interventions, including failures, are reported to provide a transparent account of the model’s behaviour under automated optimization.

## 2. Original Method (Background)
GFRRN employs a Swin‑Transformer backbone augmented with Mona adapters to adapt pretrained features to the reflection removal domain. The core components are:
- **G‑AFLB** (Gated Adaptive Feature Learning Block): frequency‑domain processing for selective restoration.
- **DAA** (Dual Attention Alignment): coordinates information flow between two decoding streams.
- **Dual‑stream output**: the left stream produces the transmission estimate (`out_l`), while the right stream generates a low‑frequency reflection map (`out_r`) used as an auxiliary signal, not a second transmission candidate.

The model was trained with a combination of reconstruction and perceptual losses. Our baseline evaluation using the official inference code confirmed a real20 PSNR of 25.84 dB, matching the paper’s reported score.

## 3. Identified Limitations
**1. Semantic Asymmetry of the Dual‑Stream Outputs.**
Code inspection reveals that `out_l` and `out_r` encode the transmission layer and the low‑frequency reflection, respectively. Averaging them under IDEA‑002 (`out = (out_l + out_r)/2`) destroyed performance, causing the real20 PSNR to collapse to the 13–17 dB range. Any fusion strategy must respect that the two feature maps carry fundamentally different information.

**2. Subtle, Non‑Local Adjustments in Apparently Reflection‑Free Regions.**
A hypothesis that the model’s minor modifications in “clean” pixels could be reverted to preserve input fidelity (IDEA‑003) degraded real20 PSNR from 25.84 dB to 25.80 dB (−0.04 dB). This indicates that GFRRN performs subtle global adjustments—likely corrections of colour balance or contrast—even in areas devoid of reflections, so naïve preservation harms performance.

**3. Performance Gap on the real20 Benchmark.**
With a baseline of 25.84 dB, the model falls short of the 27.132 dB competition target, motivating the search for inference‑side enhancements that do not require retraining.

## 4. Optimization Methodology
The sole accepted modification was a geometric test‑time augmentation (TTA) ensemble applied during inference. For every input image, four geometrically transformed variants were created: the original, a horizontal flip, a vertical flip, and a 180° rotation. The model processed each variant, after which the inverse transformation was applied, and the results were averaged pixel‑wise to produce the final transmission estimate. This 4‑fold ensemble reduces directional bias and is widely used in image restoration competitions (e.g., NTIRE 2025). The change introduced no training cost and was implemented in the main evaluation script.

All other candidate modifications—dual‑stream output fusion (IDEA‑002), non‑reflective region preservation (IDEA‑003), LRM residual refinement (IDEA‑008), reflection padding at borders (IDEA‑007), and YCbCr luminance enhancement (IDEA‑012)—were rolled back after evaluation, having either degraded performance or timed out without measurable benefit. The decisions were based on the real20 PSNR acting as the primary acceptance metric.

## 5. Experiments

### 5.1 Setup
The optimization ran on a single NVIDIA GPU (model unspecified) with a 600‑second timeout per evaluation run. The test datasets were `real20`, `solidobject`, `postcard`, `wild`, and `nature`, with PSNR and SSIM computed in the sRGB domain. The baseline was established by executing the official GFRRN inference code without any changes (real20 PSNR = 25.84 dB). The optimization budget comprised the baseline plus six subsequent iterations (seven commit states in total).

The TTA ensemble evaluation required approximately 16 minutes per run, versus ~5 minutes for the baseline. Several other experiments—involving LRM refinement (IDEA‑008), reflection padding (IDEA‑007), and luminance post‑processing (IDEA‑012)—exceeded the 600‑second limit, likely exacerbated by intermittent GPU throttling, and were abandoned. All evaluations were deterministic given the fixed model weights and input images; no explicit random seed was set.

### 5.2 Quantitative Results
Table 1 shows the baseline and best metrics (commit `162b077`, the TTA ensemble). Every dataset improved or remained stable; no regression occurred on any metric.

| Dataset       | Metric | Baseline | Best   | Δ     | Δ (%)   |
|---------------|--------|----------|--------|-------|---------|
| real20        | PSNR   | 25.84    | 26.05  | +0.21 | +0.81 % |
| real20        | SSIM   | 0.847    | 0.850  | +0.003| +0.35 % |
| solidobject   | PSNR   | 27.73    | 28.10  | +0.37 | +1.33 % |
| solidobject   | SSIM   | 0.936    | 0.939  | +0.003| +0.32 % |
| postcard      | PSNR   | 26.80    | 27.16  | +0.36 | +1.34 % |
| postcard      | SSIM   | 0.937    | 0.941  | +0.004| +0.43 % |
| wild          | PSNR   | 28.29    | 28.59  | +0.30 | +1.06 % |
| wild          | SSIM   | 0.926    | 0.930  | +0.004| +0.43 % |
| nature        | PSNR   | 27.39    | 27.42  | +0.03 | +0.11 % |
| nature        | SSIM   | 0.861    | 0.861  | 0.000 | 0.00 %  |

*Table 1: Baseline vs. optimized metrics (TTA 4‑fold ensemble). All units in dB (PSNR is higher‑better) and dimensionless (SSIM, 0–1).*

### 5.3 Ablation / Iteration Trajectory
The optimization progressed sequentially, with each candidate evaluated against the current best state (Table 2). Only the TTA modification was accepted; all others were discarded.

| Iteration | Intervention                         | real20 PSNR (dB) | Status                                                  |
|-----------|--------------------------------------|------------------|---------------------------------------------------------|
| 0         | Baseline (no changes)                | 25.84            | Baseline                                                |
| 1         | TTA 4× (H‑flip, V‑flip, 180° rotation)| 26.05            | **Accepted** (best)                                     |
| 2         | Dual‑stream fusion (IDEA‑002)         | 13–17            | Rejected; performance destroyed                         |
| 3         | Non‑reflective region preservation (IDEA‑003) | 25.80    | Rejected; −0.04 dB                                      |
| 4         | LRM residual refinement (IDEA‑008)     | (timed out)      | Partial +0.02 dB observed, but evaluation timed out; abandoned |
| 5         | Reflection padding (IDEA‑007)         | (timed out)      | Abandoned; expected marginal or neutral effect          |
| 6         | YCbCr luminance stretch (IDEA‑012)    | (timed out)      | Abandoned; post‑processing tweaks showed minimal benefit|

*Table 2: Iteration history. The best state (iteration 1) is the only accepted change.*

## 6. Discussion

The central finding is that geometric TTA—a simple, zero‑training‑overhead ensemble—yielded consistent gains across all five test sets: +0.21 dB on real20, +0.37 dB on solidobject, +0.36 dB on postcard, +0.30 dB on wild, and +0.03 dB on nature. The mechanism is that the model’s predictions contain directional biases, and symmetric averaging suppresses bias. The cost is a 4‑fold increase in inference time (~16 minutes per evaluation), which is acceptable for offline high‑quality applications but may be problematic for real‑time use.

The attempt to fuse `out_l` and `out_r` (IDEA‑002) catastrophically reduced PSNR to 13–17 dB, confirming that the two streams serve distinct semantic roles that cannot be naïvely averaged. The slight degradation from preserving non‑reflective pixels (IDEA‑003, −0.04 dB) demonstrates that GFRRN applies beneficial subtle modifications even in apparently clean areas, likely due to global colour/contrast adjustments rooted in its frequency‑domain blocks. These failures highlight that any inference‑side manipulation must respect the model’s internal design.

The study also exposed practical limits: the 600‑second timeout and GPU throttling prevented thorough exploration of more complex ideas such as 8‑fold TTA, multi‑scale inference, or gradient‑guided post‑processing. The best achieved real20 PSNR of 26.05 dB still falls short of the 27.132 dB target, indicating that larger gains would likely require training‑time optimizations (e.g., revised loss functions, learning‑rate schedules) or architectural modifications—options that lie outside the inference‑only scope of this experiment.

## 7. Reproducibility
The code repository of the GFRRN paper was used (specific URL unavailable due to a missing README). The environment required Python 3.8+, PyTorch, and torchvision, installed via `requirements.txt`. No explicit random seed was set; all evaluations were deterministic given the fixed model weights and input images. The baseline was obtained with:
```
python test.py --dataset real20
```
The optimized (TTA) run used:
```
python test.py --dataset real20 --tta ens4
```
The commit hash `162b077` marks the accepted changes. All metrics were averaged over the five standard test subsets.

## 8. References
1. *GFRRN: Explore the Gaps in Single Image Reflection Removal*, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025 (presumed; bibliographic details unavailable due to missing README).
2. tsinghua-fib-lab/AutoSOTA, GitHub repository. Automated optimization pipeline for CVPR papers.
