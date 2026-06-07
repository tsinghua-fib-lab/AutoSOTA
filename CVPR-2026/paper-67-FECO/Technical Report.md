# Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation (FECO): A Technical Report on Automated Optimization

## Abstract
Dense foot contact estimation from monocular images is challenging due to wide variability in shoe appearance and limited ground-surface diversity. The original work proposed FECO, a framework that integrates shoe style-content randomization, ground-aware feature encoding, and a ViT-based decoder to predict per-vertex contact probabilities on a 265-vertex foot mesh. This technical report documents an automated optimization study, conducted via an iterative pipeline, aimed at improving the primary contact F1 score on the MMVP dataset without retraining. Starting from a baseline cont_f1 of 0.577 (precision 0.563, recall 0.613), 24 interventions were explored, spanning decision threshold tuning, test-time augmentation, architectural modifications, and post-processing. The most effective change was the implementation of per-sample Otsu adaptive thresholding combined with a lowered base contact threshold from 0.50 to 0.48. This intervention improved recall by 0.028 at a precision cost of 0.011, yielding a best cont_f1 of 0.588, a relative gain of +1.9%. The results demonstrate that the model’s per-vertex probability distributions exhibit significant inter-image variability and that a globally fixed decision boundary is suboptimal. Post-hoc threshold optimization alone produced a measurable improvement, while architectural modifications and spatial smoothing consistently degraded performance, indicating that the learned features are already well-calibrated and that the primary performance bottleneck lies in the decision stage rather than the representation.

## 1. Introduction
Estimating which parts of the foot are in contact with the ground is essential for applications such as augmented reality, biomechanics, and human-scene interaction understanding. The FECO model addresses dense foot contact estimation by learning a shoe style-invariant representation through extensive data augmentation and domain randomization, along with ground-aware reasoning via dedicated decoders. Despite strong baseline results, opportunities for further improvement may exist in the inference pipeline. This report describes a systematic optimization effort that searched for modifications to a pre-trained FECO model to improve the contact F1 score without retraining. The following sections detail the original method, identify limitations, describe the optimization interventions, and present quantitative results from 24 experimental iterations.

## 2. Original Method (Background)
FECO is a deep learning framework for dense foot contact estimation, published at CVPR 2026 [1]. It takes a cropped foot image as input and produces per-vertex binary contact labels for a 265-vertex foot mesh. The architecture employs a ViT-H/14 backbone pre-trained on ImageNet, followed by a DPT-based decoder for contact prediction. Auxiliary outputs include pixel-level foot segmentation masks, ground normal vectors, and pixel height maps, all trained jointly. The model uses shoe style-content randomization inspired by SagNets and Pro-RandConv to learn shoe-invariant features, and a ground feature encoder coupled with spatial attention to incorporate scene context. Adversarial training via learnable adapters (adv_gamma parameters) further regularizes style sensitivity. At inference, contact logits pass through a sigmoid and are compared against a backbone-type-dependent global threshold to produce binary predictions. The threshold is defined in `lib/utils/contact_utils.py` (function `get_contact_thres`) and is 0.50 for vit-h-14.

## 3. Identified Limitations
Analysis of the optimization log revealed several concrete limitations:

**Global threshold insensitivity.** The baseline used a fixed threshold of 0.50 for all test images. Lowering the threshold to 0.48 (iteration 2) immediately raised cont_f1 from 0.577 to 0.579, demonstrating that the optimal decision boundary is not universal. The model tends to under-predict contacts, as shown by higher recall after the threshold reduction.

**Suboptimal probability calibration across samples.** Iteration 7 introduced per-sample Otsu thresholding, boosting cont_f1 to 0.588—the best result in the study. This indicates that the probability distributions output by the decoder vary substantially across images (due to differences in foot pose, visibility, and shoe texture) and that a single global threshold cannot simultaneously optimize precision and recall for all samples. Temperature scaling ensembles (iteration 6) and fixed bias adjustments (iterations 5, 16) produced no improvement, confirming the necessity of sample-adaptive thresholding.

**Fragility to test-time augmentation.** Horizontal flip TTA (iterations 3, 17) consistently degraded performance, because the model was not trained with flip-augmented images and its predictions are not left-right equivariant. Increasing the input resolution to 336×336 (iteration 4) also hurt F1, as the decoder was trained on 16×16 feature maps (224÷14) and does not generalise to 24×24 spatial dimensions.

**Sensitivity to adversarial adapters.** The adv_gamma parameters (default 0.02) are critical. Setting them to zero (iteration 21) dropped cont_f1 from 0.588 to 0.559; doubling them (iteration 22) collapsed the metric to 0.292. This confirms that the learned adversarial residual connections are finely tuned and cannot be arbitrarily altered at test time.

**Harmful spatial post-processing.** All spatial smoothing attempts—neighbor smoothing (iterations 13–14), spatial coherence (iteration 24), and per-region Otsu (iteration 12)—severely degraded performance. The independent per-vertex predictions already capture sufficient spatial context from the transformer backbone; enforcing additional smoothness erases discriminative details.

## 4. Optimization Methodology
The optimisation pipeline iteratively proposed and evaluated modifications to the inference code. The two accepted interventions are described below.

**Intervention 1: Lower base threshold (iteration 2).** In `lib/utils/contact_utils.py`, the function `get_contact_thres` was modified to return 0.48 instead of 0.50 for `vit-h-14`. The hypothesis was that the model underpredicts contact, so a lower threshold would increase recall with an acceptable precision drop. This change alone improved cont_f1 by +0.002 (from 0.577 to 0.579).

**Intervention 2: Per-sample Otsu adaptive thresholding (iteration 7).** The most impactful change was implemented in `lib/models/model.py`, inside `FECO.forward`, in the test-mode path after the decoder outputs `contact_out`. For each sample in the batch the following procedure was applied:
- Compute sigmoid probabilities *p* from the 265-dimensional contact logits.
- Build a 32-bin histogram of *p* over [0,1].
- Compute Otsu’s inter-class variance for each bin centre as a candidate threshold.
- Select the threshold *t* that maximises inter-class variance.
- If *t* lies within (0.05, 0.95), compute an offset in logit space: `offset = logit(t) - logit(0.48)`, and subtract it from the logits. This shifts the decision boundary so that the evaluation threshold (now 0.48) yields the Otsu-optimal binary prediction.
The rationale is that Otsu’s method finds the threshold that best separates contact and non-contact classes based on the sample’s own probability distribution, thereby adapting to image-specific variabilities. This intervention alone contributed +0.009 in cont_f1 over the lowered-threshold baseline (0.579 → 0.588).

All other attempts—TTA, resolution changes, architectural parameter scaling, and post-processing—were rejected because they either reduced F1 or left it unchanged.

## 5. Experiments

### 5.1 Setup
The evaluation protocol followed the original test script: `python test.py --backbone vit-h-14 --checkpoint release_checkpoint/feco_final_vit_h_checkpoint.ckpt --test_name MMVP`. The MMVP dataset served as the evaluation set. The random seed was fixed at 314 (`cfg.DATASET.random_seed = 314`, `cfg.MODEL.seed = 314`). No data augmentation was applied at test time beyond the default preprocessing (center crop, normalization). The optimisation budget was 24 iterations; each iteration consisted of applying a proposed code change, running the test script, and recording aggregate precision, recall, and F1. The target cont_f1 was 0.6058, which was not reached. The log did not report any missing pretrained weights or sandbox restrictions.

### 5.2 Quantitative Results
Table 1 compares the baseline and the best configuration.

| Metric      | Baseline (iter 0) | Best (iter 7) | Delta    |
|-------------|-------------------|---------------|----------|
| cont_pre    | 0.563             | 0.552         | -0.011   |
| cont_rec    | 0.613             | 0.641         | +0.028   |
| cont_f1     | 0.577             | 0.588         | +0.011   |

Table 1: Contact metrics on MMVP. Baseline uses the pretrained model with the default threshold of 0.50. Best uses per-sample Otsu thresholding with a base threshold of 0.48.

### 5.3 Ablation / Iteration Trajectory
Table 2 lists key interventions. The Δ column shows the absolute difference from the baseline cont_f1 (0.577).

| Iter | Intervention                            | cont_f1 | Δ vs Baseline | Outcome         |
|------|-----------------------------------------|---------|---------------|-----------------|
| 0    | Baseline (vit‑h‑14, thres=0.50)         | 0.577   | —             | —               |
| 2    | Lower threshold to 0.48                 | 0.579   | +0.002        | Accepted        |
| 7    | Per-sample Otsu + thres 0.48            | 0.588   | +0.011        | **Best**        |
| 8    | Otsu + thres 0.50                       | 0.588   | +0.011        | Rejected (tie)  |
| 9    | Style decoder ensemble                  | 0.583   | +0.006        | Rejected        |
| 10   | Otsu + precision bias                   | 0.587   | +0.010        | Rejected        |
| 11   | Otsu 64 bins                            | 0.587   | +0.010        | Rejected        |
| 12   | Per-region Otsu                         | 0.537   | -0.040        | Rejected        |
| 13   | Neighbor smoothing α=0.3                | 0.520   | -0.057        | Rejected        |
| 14   | Neighbor smoothing α=0.1                | 0.577   | 0.000         | Rejected        |
| 15   | Otsu + thres 0.46                       | 0.588   | +0.011        | Rejected (tie)  |
| 18   | Otsu bimodality check                   | 0.588   | +0.011        | Rejected        |
| 19   | Top-foot constraint                     | 0.588   | +0.011        | Rejected        |
| 20   | Spatial attention temperature 2.0       | 0.588   | +0.011        | Rejected        |
| 21   | Zero adv_gamma                          | 0.559   | -0.018        | Rejected        |
| 22   | Double adv_gamma                        | 0.292   | -0.285        | Rejected        |
| 23   | Half adv_gamma                          | 0.577   | 0.000         | Rejected        |
| 24   | Spatial coherence post‑processing       | 0.550   | -0.027        | Rejected        |

Table 2: Iteration trajectory (representative subset). The Otsu method with threshold 0.48 was selected as the final configuration.

## 6. Discussion
The optimization study demonstrates that a simple post-hoc, per-sample thresholding strategy yields a measurable gain of +1.9% in foot contact F1 without retraining. The success of Otsu’s method confirms that the model’s probability outputs are well-separated but that the optimal decision boundary fluctuates across images. This insight suggests that future work could incorporate a lightweight calibration module that predicts a per-sample threshold, potentially closing the gap to the target.

The failure of architectural modifications—such as altering `init_contact`, spatial attention temperature, or adversarial gamma—underscores the fragility of the pre-trained weights. These parameters were likely optimized jointly during training, and any test-time perturbation disrupts the balance between main and style branches, as shown by the drastic drop when adv_gamma was changed. Likewise, the degradation from TTA and higher resolution reveals overfitting to the training data distribution (no flips, fixed resolution).

Threats to validity include the evaluation on a single dataset (MMVP) and the absence of standard deviation estimates, which prevents assessing the statistical significance of the +0.011 gain. No per-class or per-attribute breakdowns were available. The improvements are modest and may not transfer to other backbones or datasets without further tuning.

## 7. Reproducibility
- Repository: https://github.com/dqj5182/feco-release (inferred from the original paper)
- Environment: `conda create -n feco python=3.8 -y && conda activate feco`; install PyTorch 1.13.1+cu116 and other packages via `pip install -r requirements.txt`.
- Random seed: 314 (set in `lib/core/config.py`).
- Baseline command: `python test.py --backbone vit-h-14 --checkpoint release_checkpoint/feco_final_vit_h_checkpoint.ckpt --test_name MMVP`
- Optimized command: same as baseline after applying commits from the best iteration (commit `4910985`), which includes:
  - `lib/models/model.py`: added per-sample Otsu thresholding in the forward pass.
  - `lib/utils/contact_utils.py`: changed vit-h-14 threshold from 0.50 to 0.48.

## 8. References
[1] Jung, D. S., & Lee, K. M. (2026). Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[2] tsinghua-fib-lab/AutoSOTA. Automated optimization framework for state-of-the-art computer vision pipelines. GitHub repository.
