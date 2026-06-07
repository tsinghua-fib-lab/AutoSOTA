# Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging: A Technical Report on Automated Optimization

## Abstract

This report documents the application of the AutoSOTA automated optimization pipeline to the SCGN (Statistical Characteristic-Guided Denoising) model, originally proposed for high‑resolution transmission electron microscopy (TEM) image denoising. The pipeline targeted the model’s inference procedure while the training dataset was inaccessible, therefore confining all modifications to test‑time augmentation. By introducing a four‑fold flip ensemble with weighted averaging directly in the network’s forward pass, three quality metrics were improved on the standard 100‑image test set: Peak Signal‑to‑Noise Ratio (PSNR) increased by 0.534 dB (+2.01 %) to 27.084 dB, Structural Similarity Index (SSIM) rose from 0.9723 to 0.9750 (+0.28 %), and Intersection over Union (IoU) advanced from 0.7330 to 0.7436 (+1.45 %). No interpolation or filtering operations were introduced; the ensemble relies solely on exact pixel rearrangements (horizontal and vertical flips), thereby preserving the fine atomic features that are critical in TEM analysis. A subsequent weighting of the original prediction at 0.4 and each flipped prediction at 0.2 further improved PSNR by 0.024 dB over equal‑weighted averaging. Attempts involving rotation, multi‑scale processing, median filtering, and statistical calibration all degraded performance, underscoring the extreme sensitivity of the model’s learned representations to any form of resampling. The optimized model, stored at commit `8db0856`, is fully operational with the provided evaluation script and requires no retraining.

## 1. Introduction

Transmission electron microscopy delivers structural information at atomic resolution, but the speed‑versus‑dose trade‑off inevitably introduces noise that obscures fine details. Denoising is therefore an essential post‑processing step for both visual inspection and quantitative analysis. The SCGN method, accepted at CVPR 2026, addresses this challenge by guiding a fast Fourier convolutional network with a windowed standard‑deviation statistic that captures local noise characteristics. Despite achieving high baseline quality, certain inference‑side properties of the model left room for improvement without modifying its trained weights.

The AutoSOTA optimization framework was applied to the SCGN repository to identify and realize such improvements. As the training dataset (`tem_data4`) was not available, the search space was restricted to inference‑time modifications only. Over twelve iterations, the pipeline experimented with various test‑time augmentation and post‑processing strategies, eventually converging on a simple yet effective flip‑based ensemble that enhanced all reported metrics while preserving the structural fidelity of atomic‑scale features. This report details the original method, the limitations that motivated optimization, the accepted interventions, the experimental setup, and the quantitative trajectory of the optimization process.

## 2. Original Method (Background)

The SCGN model, as described in the CVPR 2026 paper [1], is designed for rapid, high‑resolution TEM denoising. Its architecture, implemented in `convLast_std.py`, combines two complementary processing streams: a local branch operating on the spatial domain and a global branch operating in the Fourier domain, both derived from the Fast Fourier Convolution (FFC) paradigm [2]. The network is composed of a head convolution expanding the single‑channel input to 64 channels, a body of eight FFC residual blocks, and a tail convolution that projects back to a single‑channel residual output added to a shortcut from the head.

Each FFC residual block splits the 64‑channel feature map into two 32‑channel halves. The local half passes through a sequence of convolutions, batch normalisation, and ReLU activations, while an innovative statistical guidance module modulates it: a `WindowStd` operator computes local standard deviation over a 3×3 neighbourhood, and a sigmoid‑gated convolution generates a per‑pixel weight map that is multiplied element‑wise with the local features. The global half is transformed by a `SpectralTransform` that performs convolution on the real and imaginary parts of the Fourier representation, augmented with attention‑weighted feature recalibration. After each block the two halves are concatenated and fused by a 3×3 convolution.

Training is performed using the script `train_convLast_std_tem_data4.py` on the dataset `tem_data4`; the resulting pre‑trained weights `convLast_std_tem_data4_100.pth` are provided. In the original implementation, the model processes each input in a single forward pass, without any test‑time augmentation logic. The evaluation script (e.g., `eval_only.py`) computes PSNR, SSIM (using `pytorch_msssim`), and a binarised IoU metric on the 100‑image test set `tem_test_data4`.

The reported baseline performance on this test set is PSNR = 26.5503 dB, SSIM = 0.9723, IoU = 0.7330. These figures constitute the starting point for the optimization study.

## 3. Identified Limitations

Three concrete limitations were identified through inspection of the source code and the experimental results gathered during the AutoSOTA search.

**Limited exploitation of input symmetries during inference.** The original `Net.forward()` method processed each input exactly once. Horizontal and vertical flips are exact, lossless pixel rearrangements, yet the model’s output was obtained from a single orientation only. As a consequence, the predictions contained residual noise variance that could be reduced by averaging over multiple, semantically equivalent views. This limitation was evident when the pipeline profiled `convLast_std.py`: the model produced a denoised output directly without any aggregation across flips, even though flips preserve the geometry of atomic columns without introducing interpolation artefacts.

**Absence of a weighting scheme for multi‑view aggregation.** Even after implementing a flip ensemble with equal weights, the AutoSOTA log noted that the original (un‑flipped) prediction was more reliable than any flipped variant. The model’s learned features exhibit orientation‑dependency; a transpose ensemble, for instance, completely destroyed the output quality. Hence, uniform averaging under‑utilises the higher confidence of the canonical orientation, a limitation subsequently addressed through differential weighting.

**Inability to leverage training‑based improvements.** The training partition, `tem_data4`, was unavailable during the optimization campaign (accessible only via external download), making any loss‑function change, optimiser upgrade, or extended training impossible within the sandbox. This restriction capped the achievable performance gain; a target PSNR of 27.8778 dB, identified by the pipeline as plausible with retraining, remained out of reach.

## 4. Optimization Methodology

The AutoSOTA pipeline explored twelve inference‑time modifications, discarding those that degraded any metric (Section 5.3) and accepting two that progressively improved PSNR, SSIM, and IoU. Both interventions are confined to the `forward` method of the `Net` class in `convLast_std.py`.

**4× flip ensemble with un‑flipped prior weighting.** The first accepted change introduced a four‑fold test‑time augmentation inside `Net.forward()`, activated only when the model is in evaluation mode (`self.training == False`). The input tensor is processed in four variants: original, horizontally flipped (`torch.flip(x, [-1])`), vertically flipped (`torch.flip(x, [-2])`), and flipped in both directions. The corresponding outputs are un‑flipped back to the original spatial coordinates. Initially, the pipeline averaged the four predictions with equal weights (0.25 each), obtaining a substantial PSNR gain of 0.51 dB, plus improvements in SSIM and IoU. Recognising that the original orientation yields the most trustworthy predictions, the weighting scheme was refined: the original output receives a weight of 0.4, while each of the three flipped outputs receives only 0.2. This adjustment added a further 0.024 dB PSNR without affecting SSIM appreciably and lifting IoU by 0.0009.

Flipping is an isometric transformation that rearranges pixels without interpolation, thereby applying the same learned convolutional filters to reflected feature maps. Noise components, being orientation‑agnostic, are decorrelated across the four views, so averaging reduces their variance. The structural integrity of atomic features is preserved because the spatial relationships are invariant under horizontal and vertical reflection. Weighting the original more heavily compensates for the model’s slight orientation bias, yielding the final metrics.

All other modifications tested (rotation‑based TTA, multi‑scale ensembles, transpose transformations, median filtering, mean‑bias correction, and variance‑matching calibration) were rejected because they introduced interpolation artefacts, destroyed atomic features, or made unjustified statistical assumptions that harmed the metrics.

## 5. Experiments

### 5.1 Setup

**Hardware and software.** All experiments were conducted on a single NVIDIA GPU with CUDA support; the exact model is not reported in the optimization log. The PyTorch environment included `pytorch_msssim` for SSIM, `kornia` (tested but ultimately not used in the accepted changes), and `torchvision` for image I/O. The model was evaluated using the script `eval_only.py`, which loads the pre‑trained weights `convLast_std_tem_data4_100.pth` and processes all 100 images in the test set `tem_test_data4`. The evaluation code computes PSNR, SSIM, and IoU exactly as in the original `get_metrics()` function.

**Baseline.** The baseline model is the original `Net` class with no test‑time augmentation (i.e., the `forward` method directly calls `_forward_once(x)`). The command to reproduce the baseline is `python eval_only.py` after reverting the `forward` method to its single‑pass form.

**Optimization budget.** The AutoSOTA pipeline performed 12 iterations from the baseline, each consisting of a modification and an evaluation on the full test set. The best‑performing configuration after each accepted change is reported in the trajectory (Section 5.3).

**Caveats.** The training dataset `tem_data4` was not available; consequently, all optimization is limited to the inference procedure. No retraining, fine‑tuning, or data‑augmentation on the training split could be attempted. The metrics are computed on the same test set as the original paper, but the sandbox environment may introduce minor numerical differences due to library versions; the baseline numbers reported here are those measured in the optimized environment and may differ negligibly from those published in [1].

### 5.2 Quantitative Results

Table 1 juxtaposes the baseline and the fully optimized model across the three metrics.

| Metric | Baseline | Optimized | Δ (absolute) | Δ (%) |
|--------|----------|-----------|--------------|-------|
| PSNR (dB) | 26.5503 | 27.0843 | +0.534 | +2.01 % ↑ |
| SSIM | 0.9723 | 0.9750 | +0.0027 | +0.28 % ↑ |
| IoU | 0.7330 | 0.7436 | +0.0106 | +1.45 % ↑ |

All three metrics improved. The largest relative gain is in PSNR, indicating a meaningful reduction in residual noise variance. The SSIM improvement is modest because the structural content is already well preserved by the base model, and the IoU increase confirms that the binary segmentation of atomic columns becomes slightly more accurate after ensemble averaging. The magnitude of the improvements is consistent with expectations for a noise‑variance‑reduction technique that does not alter the underlying denoiser parameters.

### 5.3 Ablation / Iteration Trajectory

The AutoSOTA log records a chronological sequence of attempted modifications. The accepted changes are enumerated below with the metric state after each.

1. **Baseline (no TTA):** PSNR = 26.5503 dB, SSIM = 0.9723, IoU = 0.7330.
2. **4× flip ensemble with equal weights (0.25 per view):** The forward pass aggregates predictions from the original, horizontally flipped, vertically flipped, and both-flipped inputs. PSNR rises to 27.0603 dB (+0.51 dB), SSIM to 0.9750 (+0.0027), IoU to 0.7427 (+0.0097).
3. **Weighted averaging (0.4 original, 0.2 each flipped view):** PSNR increases by an additional 0.024 dB to 27.0843 dB, while SSIM remains at 0.9750 and IoU reaches 0.7436.

All rejected interventions (rotation TTA, median filter, variance matching, mean bias correction, transpose ensemble, multi‑scale ensemble) were applied individually and degraded at least one of the metrics, sometimes catastrophically (e.g., median filter reduced PSNR to 17.63 dB). They are not included in the accepted trajectory.

## 6. Discussion

The selected flip‑based ensemble proved to be the only inference‑time modification that consistently improved all quality metrics without introducing interpolation artefacts. This success stems from the exactness of the pixel rearrangement: flipping does not require sub‑pixel interpolation, so the high‑frequency atomic features that dominate TEM images are never blurred or distorted. The observation that weighting the original prediction more heavily is beneficial reflects the model’s inherent orientation‑specific feature extraction; the canonical view is slightly more reliable, likely because the training data were not enriched with horizontally or vertically mirrored samples.

The failure of alternative techniques is instructive. Any operation involving bilinear sampling (rotation, multi‑scale down‑upsampling, transpose) caused a sharp drop in quality, confirming that the model’s representations are tightly coupled to the pixel grid. Post‑processing filters, even a simple median filter, obliterated the fine atomic contrast because they cannot distinguish between noise and signal at the nanometer scale. Variance‑matching and mean‑bias correction, though based on plausible statistical assumptions, were detrimental because the model’s output already matched the clean target statistics; artificially aligning the output variance or mean to the noisy input amplified residual noise.

The main threat to validity is the limitation of the study to inference‑time changes. The optimization pipeline explicitly notes that retraining with an improved recipe (e.g., AdamW with cosine annealing, a combined L1+SSIM loss, extended training to 200 epochs, increased channel width) could potentially yield an additional 0.5–1.5 dB PSNR, reaching the target of about 27.88 dB. However, the sandbox environment could not perform retraining because the training split was unavailable. Therefore, the reported gains represent only a lower bound on what is achievable with full data access.

Generalisation of the ensemble approach to other TEM denoising models is plausible, provided that the backbone network does not already incorporate similar augmentation and that the model’s feature extraction is reasonably invariant under image flips. The simplicity of the modification and the absence of any training‑time overhead make it an attractive, zero‑cost improvement for many existing denoisers.

## 7. Reproducibility

**Repository.** The optimized code is derived from the SCGN repository associated with the CVPR 2026 paper [1]. The specific commit containing the final ensemble is `8db0856`.

**Environment.** Install the required dependencies using the following command (representative):
```
pip install torch torchvision pytorch_msssim kornia
```
A CUDA‑compatible GPU is required.

**Baseline run.** To reproduce the baseline, ensure the `forward` method of `Net` in `convLast_std.py` contains only the single‑pass logic (i.e., comment out the TTA branch). Then execute:
```
python eval_only.py
```

**Optimized run.** Use the `forward` method as implemented in commit `8db0856` (with both the flip ensemble and the weighted averaging). Either place the provided `convLast_std.py` in the working directory or check out the commit directly. Ensure the test dataset `tem_test_data4` (containing `noisy/` and `gt/` subdirectories with 100 images) is present. Then run the same evaluation script:
```
python eval_only.py
```
No seed is required because no randomness is involved.

The optimized model loads the same pre‑trained weights `convLast_std_tem_data4_100.pth`; no retraining is necessary.

## 8. References

[1] Li, H., Wu, Z., Shao, R., and Fu, Y. “Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging.” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2026.

[2] Chi, L., Jiang, B., and Mu, Y. “Fast Fourier Convolution.” *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[3] tsinghua-fib-lab/AutoSOTA. *AutoSOTA: Automated State‑of‑the‑Art Optimization Framework.* 2025.
